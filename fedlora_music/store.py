"""Client-local storage. Everything under a client's directory stays on that client."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ClientStore:
    root: Path

    @classmethod
    def for_partition(cls, clients_root: Path, partition_id: int) -> ClientStore:
        return cls(clients_root / f"client-{partition_id}")

    @classmethod
    def resolve(
        cls, data_dir: Path | None, clients_root: Path, node_config: Mapping[str, Any]
    ) -> ClientStore:
        """Deployment: the node's own ``data-dir``. Simulation: ``client-<partition-id>``."""
        if data_dir is not None:
            return cls(data_dir)
        if "partition-id" in node_config:
            return cls.for_partition(clients_root, int(node_config["partition-id"]))
        raise ValueError(
            "node config must set `data-dir` (deployment) or `partition-id` (simulation)"
        )

    @property
    def tensor_dir(self) -> Path:
        """ACE-Step preprocessed ``.pt`` tensors produced locally by ``fedlora-prepare``."""
        return self.root / "tensors"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def personal_adapter_path(self) -> Path:
        return self.state_dir / "personal_adapter.safetensors"

    @property
    def local_global_path(self) -> Path:
        return self.state_dir / "global_adapter_local.safetensors"

    @property
    def ledger_path(self) -> Path:
        return self.state_dir / "privacy_ledger.json"

    @property
    def fused_adapter_dir(self) -> Path:
        """Ready-to-load PEFT adapter = latest global + personal (the client's model)."""
        return self.root / "export" / "fused_adapter"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "train_log.jsonl"

    def load_tensors(self, path: Path) -> dict[str, torch.Tensor] | None:
        from safetensors.torch import load_file

        return load_file(str(path)) if path.is_file() else None

    def save_tensors(self, path: Path, state: dict[str, torch.Tensor]) -> None:
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        save_file({k: v.contiguous() for k, v in state.items()}, str(tmp))
        tmp.replace(path)

    def append_log(self, record: dict[str, object]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **record}) + "\n")
