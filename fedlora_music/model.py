"""Load the ACE-Step model once per process and keep it warm across rounds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn

from fedlora_music.adapters import inject_dual_lora
from fedlora_music.config import ModelSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Runtime:
    model: nn.Module
    device: torch.device
    dtype: torch.dtype
    timestep_mu: float
    timestep_sigma: float
    data_proportion: float


def select_device() -> tuple[torch.device, str]:
    """Pick device and ACE-Step precision string (bf16 on capable CUDA, else fp32)."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "fp32"
    return torch.device("cpu"), "fp32"


def load_base_with_adapters(spec: ModelSpec, device: torch.device, precision: str) -> nn.Module:
    """Load ``AceStepConditionGenerationModel`` via ACE-Step's own loader and add adapters."""
    from acestep.training_v2.model_loader import load_decoder_for_training

    model = load_decoder_for_training(
        checkpoint_dir=str(spec.checkpoint_dir),
        variant=spec.model_variant,
        device=str(device),
        precision=precision,
    )
    model = inject_dual_lora(model, spec.adapters)
    return model.to(device)


@lru_cache(maxsize=1)
def get_runtime(spec: ModelSpec) -> Runtime:
    """Process-wide singleton. Flower simulation reuses worker processes across rounds."""
    from acestep.training_v2.model_loader import read_model_config

    device, precision = select_device()
    model = load_base_with_adapters(spec, device, precision)
    mcfg = read_model_config(str(spec.checkpoint_dir), spec.model_variant)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)
    logger.info("Loaded %s on %s (%s)", spec.model_variant, device, dtype)
    return Runtime(
        model=model,
        device=device,
        dtype=dtype,
        timestep_mu=float(mcfg.get("timestep_mu", -0.4)),
        timestep_sigma=float(mcfg.get("timestep_sigma", 1.0)),
        data_proportion=float(mcfg.get("data_proportion", 0.0)),
    )
