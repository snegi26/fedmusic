"""A tiny CPU stand-in model for tests, CI and the dev VMs. No downloads, no GPU.

It exercises the whole harness for real (adapter injection, federated rounds, DP,
fusion, export, adapter loading for generation) while the "model" is a 2-block MLP
and the "audio" is a sine tone. Useful to check plumbing, never for quality.

* ``prepare`` turns each audio file into a fixed random ``(x, y)`` pair seeded by a
  hash of the file bytes (nothing is decoded).
* ``loss`` regresses ``y`` from ``x`` through the adapted blocks.
* ``generate`` maps the caption and seed to a tone frequency through the adapted
  model and writes a short 16 kHz WAV, so an adapter audibly changes the output.
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fedlora_music.backends.base import Batch, Generator, ModelBackend, PrepareResult
from fedlora_music.embeddings import list_audio

if TYPE_CHECKING:
    from fedlora_music.model import Runtime

DIM = 16
SAMPLE_RATE = 16_000


class _Block(nn.Module):
    """Has the same projection names as ACE-Step's attention, so the default
    ``target-modules`` work unchanged."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.v_proj, self.o_proj = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.o_proj(
            torch.relu(self.q_proj(x)) * torch.tanh(self.k_proj(x) + self.v_proj(x))
        )


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(_Block(DIM), _Block(DIM))


def _seeded(seed_material: bytes) -> torch.Generator:
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") >> 1
    return torch.Generator().manual_seed(seed)


class _Pairs(Dataset):
    def __init__(self, files: list[Path]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return torch.load(self.files[idx], map_location="cpu", weights_only=True)


class ToyBackend(ModelBackend):
    name: ClassVar[str] = "toy"
    default_variant: ClassVar[str] = "tiny"
    adapter_root: ClassVar[str] = "decoder"
    weights_license: ClassVar[str] = "n/a (randomly initialized, seed 0)"

    def load_model(self, device: torch.device, precision: str) -> nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = ToyModel()
        model.requires_grad_(False)
        return model.to(device)

    def fingerprint(self) -> str:
        return hashlib.sha256(f"toy:{self.variant}:dim={DIM}:blocks=2:v1".encode()).hexdigest()

    def build_loader(self, tensor_dir: Path, batch_size: int, shuffle: bool = True) -> DataLoader:
        files = sorted(tensor_dir.glob("*.pt")) if tensor_dir.is_dir() else []
        if not files:
            raise FileNotFoundError(
                f"No local data at {tensor_dir}. Run `fedlora-prepare` on this client first."
            )
        return DataLoader(_Pairs(files), batch_size=batch_size, shuffle=shuffle)

    def prepare(
        self,
        *,
        audio_dir: Path | None,
        dataset_json: Path | None,
        out_dir: Path,
        max_duration: float,
    ) -> PrepareResult:
        source = dataset_json or audio_dir
        if source is None:
            raise ValueError("toy prepare needs an audio folder or a dataset JSON")
        files = list_audio(source)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(files):
            g = _seeded(path.read_bytes())
            pair = {"x": torch.randn(DIM, generator=g), "y": torch.randn(DIM, generator=g)}
            torch.save(pair, out_dir / f"{i:05d}.pt")
        return PrepareResult(processed=len(files), total=len(files))

    def loss(self, rt: Runtime, batch: Batch, *, train: bool, cfg_ratio: float) -> torch.Tensor:
        x, y = batch["x"].to(rt.device), batch["y"].to(rt.device)
        if not train:  # a stochastic term, so a fixed seed matters as for real models
            x = x + 0.01 * torch.randn_like(x)
        return nn.functional.mse_loss(rt.model.decoder.base_model(x), y).float()

    def open_generator(self) -> Generator:
        return _ToyGenerator(self)


class _ToyGenerator(Generator):
    def __init__(self, backend: ToyBackend) -> None:
        self.backend = backend

    def _decoder(self, adapter_dir: Path | None, scale: float) -> nn.Module:
        decoder = self.backend.load_model(torch.device("cpu"), "fp32").decoder
        if adapter_dir is None:
            return decoder
        from peft import PeftModel

        peft = PeftModel.from_pretrained(decoder, str(adapter_dir))
        if scale != 1.0:
            for name, param in peft.named_parameters():
                if ".lora_B." in name:
                    param.data.mul_(scale)
        return peft.base_model

    @torch.no_grad()
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
        decoder = self._decoder(adapter_dir, adapter_scale)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for seed in seeds:
            seed = (
                seed if seed >= 0 else int.from_bytes(hashlib.sha256(caption.encode()).digest()[:4])
            )
            x = torch.randn(DIM, generator=_seeded(f"{caption}\0{seed}".encode()))
            level = torch.sigmoid(decoder(x).mean()).item()
            freq = 110.0 * 2 ** (3 * level)  # 110-880 Hz
            n = int(max(duration, 0.1) * SAMPLE_RATE)
            frames = b"".join(
                struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)))
                for i in range(n)
            )
            path = out_dir / f"toy-{hashlib.sha256(caption.encode()).hexdigest()[:8]}-{seed}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(frames)
            paths.append(path)
        return paths
