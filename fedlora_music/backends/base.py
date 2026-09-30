"""The contract between the federated harness and a base model.

Everything model-independent (DP clipping and noise, the privacy ledger, client
policy, FedAvg, the dual adapters and their fusion, evaluation metrics) lives in the
rest of the package. A backend supplies only what depends on the model:

* loading the base model and naming the submodule that gets the adapters
* turning audio into training inputs, and loading those inputs
* the training loss (and a deterministic variant for held-out evaluation)
* generating audio with an exported adapter
* a fingerprint of the exact base weights, so clients on a different model refuse
  to join instead of silently corrupting the average

Backend modules must not import their model library at import time: the registry
imports a backend class to read its metadata, and the client checks its allowlist
before that.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

if TYPE_CHECKING:
    from fedlora_music.config import ModelSpec
    from fedlora_music.model import Runtime

Batch = dict[str, Any]


@dataclass(frozen=True)
class PrepareResult:
    processed: int
    total: int


class Generator(ABC):
    """A loaded generation pipeline. Switching adapters must not reload the base model."""

    @abstractmethod
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
        """One clip per seed, written under ``out_dir``. ``adapter_dir=None``: base model.
        A negative seed means a random one."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Free model memory."""


class ModelBackend(ABC):
    """One base model family. Instances are per :class:`ModelSpec` and cached."""

    name: ClassVar[str]
    default_variant: ClassVar[str]
    #: Attribute of the loaded model that the LoRA adapters wrap.
    adapter_root: ClassVar[str] = "decoder"
    #: Human-readable license of the default weights, shown in docs and ``--help``.
    weights_license: ClassVar[str] = "unknown"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def variant(self) -> str:
        return self.spec.variant or self.default_variant

    @property
    def key(self) -> str:
        """Storage key: ``<backend>/<variant>`` (see ``ClientStore.model``)."""
        return f"{self.name}/{self.variant}"

    def precision(self, device: torch.device) -> str:
        """Precision string for ``device``: bf16 on capable CUDA, else fp32."""
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"

    @abstractmethod
    def load_model(self, device: torch.device, precision: str) -> nn.Module:
        """The base model, all weights frozen, with ``adapter_root`` as an attribute."""

    @abstractmethod
    def fingerprint(self) -> str:
        """Stable hash of the base weights. Same weights on every machine, same hash."""

    @abstractmethod
    def build_loader(
        self, tensor_dir: Path, batch_size: int, shuffle: bool = True
    ) -> Iterable[Batch]:
        """Batches of prepared inputs from ``tensor_dir``."""

    @abstractmethod
    def loss(self, rt: Runtime, batch: Batch, *, train: bool, cfg_ratio: float) -> torch.Tensor:
        """Scalar fp32 loss. ``train=False``: no condition dropout, and all randomness
        from the global RNG, so a fixed seed gives a fixed value."""

    @abstractmethod
    def prepare(
        self,
        *,
        audio_dir: Path | None,
        dataset_json: Path | None,
        out_dir: Path,
        max_duration: float,
    ) -> PrepareResult:
        """Turn the client's audio into training inputs under ``out_dir``, locally."""

    @abstractmethod
    def open_generator(self) -> Generator:
        """Load the generation pipeline once, for any number of clips and adapters."""


_FINGERPRINT_CACHE = (
    Path(os.environ.get("FEDLORA_CACHE_DIR", Path.home() / ".cache" / "fedlora_music"))
    / "fingerprints.json"
)


def fingerprint_files(root: Path, files: Iterable[Path], extra: str = "") -> str:
    """SHA-256 over the relative paths and contents of ``files`` under ``root``.

    Multi-GB checkpoints take a while to hash, so per-file digests are cached,
    keyed by path, size and modification time.
    """
    try:
        cache = json.loads(_FINGERPRINT_CACHE.read_text())
    except (OSError, ValueError):
        cache = {}
    h = hashlib.sha256(extra.encode())
    changed = False
    for path in sorted(files):
        st = path.stat()
        key = f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
        digest = cache.get(key)
        if digest is None:
            fh = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    fh.update(chunk)
            digest = cache[key] = fh.hexdigest()
            changed = True
        h.update(f"{path.relative_to(root).as_posix()}\0{digest}\n".encode())
    if changed:
        try:
            _FINGERPRINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _FINGERPRINT_CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(_FINGERPRINT_CACHE)
        except OSError:
            pass  # a read-only home only costs re-hashing next time
    return h.hexdigest()
