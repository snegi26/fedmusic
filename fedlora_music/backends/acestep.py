"""ACE-Step 1.5 backend (open weights, MIT).

``spec.root`` is the ACE-Step checkout holding ``checkpoints/``; ``spec.variant`` a
DiT folder inside it (default ``acestep-v15-turbo``). ACE-Step pins its own torch
build, so this package is installed into ACE-Step's environment (see README).

The training step mirrors ACE-Step's corrected trainer
(``acestep.training_v2.fixed_lora_module``): logit-normal timesteps from the model
config, CFG dropout onto the null condition embedding, and MSE against the flow
``x1 - x0``. ``acestep`` is imported lazily, inside methods only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from fedlora_music.backends.base import (
    Batch,
    Generator,
    ModelBackend,
    PrepareResult,
    fingerprint_files,
)

if TYPE_CHECKING:
    from fedlora_music.model import Runtime

logger = logging.getLogger(__name__)

_TENSOR_KEYS = (
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "context_latents",
)


class AceStepBackend(ModelBackend):
    name: ClassVar[str] = "acestep"
    default_variant: ClassVar[str] = "acestep-v15-turbo"
    adapter_root: ClassVar[str] = "decoder"
    weights_license: ClassVar[str] = "MIT"

    _timesteps: dict[str, float] | None = None

    @property
    def root(self) -> Path:
        if self.spec.root is None:
            raise ValueError("the acestep backend needs `model-root` (the ACE-Step checkout)")
        return self.spec.root

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    def _model_dir(self) -> Path:
        path = self.checkpoint_dir / self.variant
        if not path.is_dir():
            raise FileNotFoundError(
                f"{path} not found; download it with `uv run acestep-download` in {self.root}"
            )
        return path

    # -- model -------------------------------------------------------------------
    def load_model(self, device: torch.device, precision: str) -> nn.Module:
        from acestep.training_v2.model_loader import load_decoder_for_training

        model = load_decoder_for_training(
            checkpoint_dir=str(self.checkpoint_dir),
            variant=self.variant,
            device=str(device),
            precision=precision,
        )
        logger.info("Loaded ACE-Step %s on %s (%s)", self.variant, device, precision)
        return model

    def _timestep_config(self) -> dict[str, float]:
        if self._timesteps is None:
            from acestep.training_v2.model_loader import read_model_config

            mcfg = read_model_config(str(self.checkpoint_dir), self.variant)
            self._timesteps = {
                "timestep_mu": float(mcfg.get("timestep_mu", -0.4)),
                "timestep_sigma": float(mcfg.get("timestep_sigma", 1.0)),
                "data_proportion": float(mcfg.get("data_proportion", 0.0)),
            }
        return self._timesteps

    def fingerprint(self) -> str:
        model_dir = self._model_dir()
        files = [p for p in model_dir.rglob("*") if p.is_file()]
        return fingerprint_files(model_dir, files, extra=f"{self.name}:{self.variant}")

    # -- data --------------------------------------------------------------------
    def build_loader(self, tensor_dir: Path, batch_size: int, shuffle: bool = True) -> DataLoader:
        from acestep.training.data_module import (
            PreprocessedTensorDataset,
            collate_preprocessed_batch,
        )
        from acestep.training.path_safety import set_safe_root

        if not tensor_dir.is_dir():
            raise FileNotFoundError(
                f"No local data at {tensor_dir}. Run `fedlora-prepare` on this client first."
            )
        # ACE-Step only reads tensors under its "safe root", which defaults to the
        # process's start-up directory. The client's own data folder (often elsewhere,
        # e.g. under ~/Library/Application Support) is the trusted location here.
        set_safe_root(str(tensor_dir))
        dataset = PreprocessedTensorDataset(str(tensor_dir))
        if len(dataset) == 0:
            raise ValueError(f"{tensor_dir} contains no preprocessed tensors")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_preprocessed_batch,
            num_workers=0,
            drop_last=False,
        )

    def prepare(
        self,
        *,
        audio_dir: Path | None,
        dataset_json: Path | None,
        out_dir: Path,
        max_duration: float,
    ) -> PrepareResult:
        from acestep.training_v2.preprocess import preprocess_audio_files

        result = preprocess_audio_files(
            audio_dir=str(audio_dir) if audio_dir else None,
            output_dir=str(out_dir),
            checkpoint_dir=str(self.checkpoint_dir),
            variant=self.variant,
            max_duration=max_duration,
            dataset_json=str(dataset_json) if dataset_json else None,
        )
        return PrepareResult(processed=int(result["processed"]), total=int(result["total"]))

    # -- objective ---------------------------------------------------------------
    def loss(self, rt: Runtime, batch: Batch, *, train: bool, cfg_ratio: float) -> torch.Tensor:
        from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps

        ts = self._timestep_config()
        model = rt.model
        use_autocast = rt.device.type in ("cuda", "xpu", "mps") and rt.dtype != torch.float32
        ctx = (
            torch.autocast(device_type=rt.device.type, dtype=rt.dtype)
            if use_autocast
            else nullcontext()
        )
        with ctx:
            t_in = {
                k: batch[k].to(rt.device, dtype=rt.dtype, non_blocking=True) for k in _TENSOR_KEYS
            }
            x0 = t_in["target_latents"]
            ehs = t_in["encoder_hidden_states"]
            null_emb = getattr(model, "null_condition_emb", None)
            if train and null_emb is not None and cfg_ratio > 0.0:
                ehs = apply_cfg_dropout(ehs, null_emb, cfg_ratio=cfg_ratio)

            x1 = torch.randn_like(x0)
            t, _ = sample_timesteps(
                batch_size=x0.shape[0],
                device=rt.device,
                dtype=rt.dtype,
                data_proportion=ts["data_proportion"],
                timestep_mu=ts["timestep_mu"],
                timestep_sigma=ts["timestep_sigma"],
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

    # -- generation --------------------------------------------------------------
    def open_generator(self) -> Generator:
        return _AceGenerator(self.root, self.variant)


class _AceGenerator(Generator):
    def __init__(self, root: Path, variant: str) -> None:
        from acestep.handler import AceStepHandler

        self.dit: Any = AceStepHandler()
        status, ok = self.dit.initialize_service(
            project_root=str(root), config_path=variant, device="auto"
        )
        if not ok:
            raise RuntimeError(f"ACE-Step init failed: {status}")
        self._adapter: Path | None = None

    def _use(self, adapter_dir: Path | None, scale: float) -> None:
        if adapter_dir != self._adapter:
            if self._adapter is not None:
                self.dit.unload_lora()
                self._adapter = None
            if adapter_dir is not None:
                msg = self.dit.load_lora(str(adapter_dir))
                if not msg.startswith("✅"):
                    raise RuntimeError(f"adapter load failed: {msg}")
                self._adapter = adapter_dir
        if adapter_dir is not None:
            self.dit.set_lora_scale(scale)

    def generate(
        self,
        adapter_dir: Path | None,
        caption: str,
        seeds: Sequence[int],
        duration: float,
        out_dir: Path,
        lyrics: str = "[Instrumental]",
        adapter_scale: float = 1.0,
    ) -> list[Path]:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        self._use(adapter_dir, adapter_scale)
        random = any(s < 0 for s in seeds)
        result = generate_music(
            self.dit,
            None,  # no LM planner: DiT-only, caption drives the prompt
            GenerationParams(caption=caption, lyrics=lyrics, duration=duration, thinking=False),
            GenerationConfig(
                batch_size=len(seeds),
                use_random_seed=random,
                seeds=None if random else list(seeds),
                audio_format="flac",
            ),
            save_dir=str(out_dir),
        )
        if not result.success:
            raise RuntimeError(f"generation failed: {result.error}")
        return [Path(a["path"]) for a in result.audios]
