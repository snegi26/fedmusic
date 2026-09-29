"""Local training: flow-matching loss on the client's own preprocessed tensors.

The step mirrors ACE-Step's corrected trainer (``acestep.training_v2.fixed_lora_module``):
logit-normal timesteps from the model config, CFG dropout onto the null condition
embedding, and MSE against the flow ``x1 - x0``. Both adapters train jointly.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedlora_music.adapters import GLOBAL, PERSONAL, is_param_of
from fedlora_music.config import TrainSpec
from fedlora_music.model import Runtime

_TENSOR_KEYS = (
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
)


def build_loader(tensor_dir: Path, batch_size: int) -> DataLoader:
    from acestep.training.data_module import (
        PreprocessedTensorDataset,
        collate_preprocessed_batch,
    )

    if not tensor_dir.is_dir():
        raise FileNotFoundError(
            f"No local data at {tensor_dir}. Run `fedlora-prepare` on this client first."
        )
    dataset = PreprocessedTensorDataset(str(tensor_dir))
    if len(dataset) == 0:
        raise ValueError(f"{tensor_dir} contains no preprocessed tensors")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_preprocessed_batch,
        num_workers=0,
        drop_last=False,
    )


def flow_matching_loss(
    rt: Runtime, batch: dict[str, torch.Tensor], cfg_ratio: float
) -> torch.Tensor:
    from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps

    model = rt.model
    use_autocast = rt.device.type in ("cuda", "xpu", "mps") and rt.dtype != torch.float32
    ctx = (
        torch.autocast(device_type=rt.device.type, dtype=rt.dtype)
        if use_autocast
        else nullcontext()
    )
    with ctx:
        t_in = {k: batch[k].to(rt.device, dtype=rt.dtype, non_blocking=True) for k in _TENSOR_KEYS}
        x0 = t_in["target_latents"]
        ehs = t_in["encoder_hidden_states"]
        null_emb = getattr(model, "null_condition_emb", None)
        if null_emb is not None and cfg_ratio > 0.0:
            ehs = apply_cfg_dropout(ehs, null_emb, cfg_ratio=cfg_ratio)

        x1 = torch.randn_like(x0)
        t, _ = sample_timesteps(
            batch_size=x0.shape[0],
            device=rt.device,
            dtype=rt.dtype,
            data_proportion=rt.data_proportion,
            timestep_mu=rt.timestep_mu,
            timestep_sigma=rt.timestep_sigma,
            use_meanflow=False,
        )
        tt = t.view(-1, 1, 1)
        xt = tt * x1 + (1.0 - tt) * x0
        out = model.decoder(
            hidden_states=xt,
            timestep=t,
            timestep_r=t,
            attention_mask=t_in["attention_mask"],
            encoder_hidden_states=ehs,
            encoder_attention_mask=t_in["encoder_attention_mask"],
            context_latents=t_in["context_latents"],
        )
        loss = F.mse_loss(out[0], x1 - x0)
    return loss.float()


def _batches(loader: DataLoader, steps: int) -> Iterator[dict[str, torch.Tensor]]:
    """Yield exactly ``steps`` batches, reshuffling on every pass over the data."""
    produced = 0
    while produced < steps:
        for batch in loader:
            yield batch
            produced += 1
            if produced == steps:
                return


def train_local(rt: Runtime, loader: DataLoader, spec: TrainSpec) -> float:
    """Run ``spec.local_steps`` optimiser steps; return mean loss (kept local)."""
    named = [(n, p) for n, p in rt.model.named_parameters() if p.requires_grad]
    g_params = [p for n, p in named if is_param_of(n, GLOBAL)]
    p_params = [p for n, p in named if is_param_of(n, PERSONAL)]
    optim = torch.optim.AdamW(
        [
            {"params": g_params, "lr": spec.lr_global},
            {"params": p_params, "lr": spec.lr_personal},
        ],
        weight_decay=0.0,
    )
    all_params = g_params + p_params

    rt.model.train()
    losses: list[float] = []
    for batch in _batches(loader, spec.local_steps):
        loss = flow_matching_loss(rt, batch, spec.cfg_ratio)
        if not torch.isfinite(loss):
            optim.zero_grad(set_to_none=True)
            continue
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, spec.grad_clip)
        optim.step()
        losses.append(loss.item())
    rt.model.eval()
    return sum(losses) / len(losses) if losses else math.nan
