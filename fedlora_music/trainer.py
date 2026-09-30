"""Local training on the client's own prepared inputs, for any model backend.

The loss comes from the backend (``Runtime.backend.loss``); this module owns the
optimiser loop and the held-out evaluation protocol. Both adapters train jointly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch

from fedlora_music.adapters import GLOBAL, PERSONAL, is_param_of
from fedlora_music.backends import Batch
from fedlora_music.config import TrainSpec
from fedlora_music.model import Runtime


@torch.no_grad()
def heldout_loss(rt: Runtime, loader: Iterable[Batch], seed: int = 0) -> float:
    """Mean loss on held-out inputs with fixed randomness (for diffusion: noise and t).

    The RNG is reseeded per batch and condition dropout is off, so every adapter
    variant sees exactly the same randomness: differences come from the weights
    alone. Pass an unshuffled loader. Reseeds the global RNG; meant for eval
    processes. Values are comparable within one backend only.
    """
    was_training = rt.model.training
    rt.model.eval()
    losses: list[float] = []
    for i, batch in enumerate(loader):
        torch.manual_seed(seed + i)
        loss = rt.backend.loss(rt, batch, train=False, cfg_ratio=0.0)
        if torch.isfinite(loss):
            losses.append(loss.item())
    rt.model.train(was_training)
    return sum(losses) / len(losses) if losses else math.nan


def _batches(loader: Iterable[Batch], steps: int) -> Iterator[Batch]:
    """Yield exactly ``steps`` batches, reshuffling on every pass over the data."""
    produced = 0
    while produced < steps:
        for batch in loader:
            yield batch
            produced += 1
            if produced == steps:
                return


def train_local(rt: Runtime, loader: Iterable[Batch], spec: TrainSpec) -> float:
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
        loss = rt.backend.loss(rt, batch, train=True, cfg_ratio=spec.cfg_ratio)
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
