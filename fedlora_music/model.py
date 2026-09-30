"""Load the base model once per process and keep it warm across rounds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn

from fedlora_music.adapters import inject_dual_lora
from fedlora_music.backends import ModelBackend, get_backend
from fedlora_music.config import ModelSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Runtime:
    model: nn.Module
    device: torch.device
    dtype: torch.dtype
    backend: ModelBackend


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_base_with_adapters(spec: ModelSpec, device: torch.device, precision: str) -> nn.Module:
    """The backend's base model with the ``global`` and ``personal`` adapters added."""
    if spec.adapters is None:
        raise ValueError("an AdapterSpec is required to add adapters")
    backend = get_backend(spec)
    model = backend.load_model(device, precision)
    model = inject_dual_lora(model, spec.adapters, root=backend.adapter_root)
    return model.to(device)


@lru_cache(maxsize=1)
def get_runtime(spec: ModelSpec) -> Runtime:
    """Process-wide singleton. Flower simulation reuses worker processes across rounds."""
    backend = get_backend(spec)
    device = select_device()
    precision = backend.precision(device)
    model = load_base_with_adapters(spec, device, precision)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)
    logger.info("Runtime ready: %s on %s (%s)", backend.key, device, dtype)
    return Runtime(model=model, device=device, dtype=dtype, backend=backend)
