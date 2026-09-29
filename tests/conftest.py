"""Shared toy model: a tiny stand-in for ACE-Step's DiT, so no checkpoints are needed."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DIM = 16


class Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.relu(self.q_proj(x)))


class Toy(nn.Module):
    """Has a ``.decoder`` like ``AceStepConditionGenerationModel``."""

    def __init__(self, d: int = DIM) -> None:
        super().__init__()
        self.decoder = nn.Sequential(Block(d), Block(d))


def toy_loss(rt: Any, batch: dict[str, torch.Tensor], cfg_ratio: float) -> torch.Tensor:
    """Replaces the flow-matching loss: regress ``y`` from ``x`` through the decoder."""
    del cfg_ratio
    return nn.functional.mse_loss(rt.model.decoder.base_model(batch["x"]), batch["y"])


@pytest.fixture
def toy_batches() -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    return [
        {"x": torch.randn(4, DIM, generator=g), "y": torch.randn(4, DIM, generator=g)}
        for _ in range(2)
    ]


@pytest.fixture
def run_config(tmp_path: Path) -> dict[str, Any]:
    """The app's real run config, shrunk to the toy model and pointed at ``tmp_path``."""
    rc = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    return {
        **rc,
        "target-modules": "q_proj,o_proj",
        "global-rank": 4,
        "global-alpha": 8,
        "personal-rank": 2,
        "personal-alpha": 2,
        "local-steps": 3,
        "lr-global": 1e-2,
        "lr-personal": 1e-2,
        "clients-root": str(tmp_path / "clients"),
        "server-output-dir": str(tmp_path / "server_out"),
        "ace-project-root": str(tmp_path / "ace"),
    }
