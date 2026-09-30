"""Shared fixtures. Federated tests run on the built-in ``toy`` backend: no checkpoints."""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
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
    """Minimal model with a ``.decoder``, for adapter-level tests."""

    def __init__(self, d: int = DIM) -> None:
        super().__init__()
        self.decoder = nn.Sequential(Block(d), Block(d))


@pytest.fixture(autouse=True)
def _fresh_caches() -> Iterator[None]:
    """Model and backend caches are process-wide; keep tests independent."""
    from fedlora_music.backends import get_backend
    from fedlora_music.model import get_runtime

    get_runtime.cache_clear()
    get_backend.cache_clear()
    yield
    get_runtime.cache_clear()
    get_backend.cache_clear()


@pytest.fixture
def run_config(tmp_path: Path) -> dict[str, Any]:
    """The app's real run config on the toy backend, pointed at ``tmp_path``."""
    rc = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    return {
        **rc,
        "model-backend": "toy",
        "model-root": "",
        "model-variant": "",
        "global-rank": 4,
        "global-alpha": 8,
        "personal-rank": 2,
        "personal-alpha": 2,
        "local-steps": 3,
        "lr-global": 1e-2,
        "lr-personal": 1e-2,
        "clients-root": str(tmp_path / "clients"),
        "server-output-dir": str(tmp_path / "server_out"),
    }


def make_songs(folder: Path, n: int = 3) -> Path:
    """Stand-in audio files; the toy backend hashes bytes and never decodes them."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"song{i}.wav").write_bytes(f"not really audio {i}".encode())
    return folder
