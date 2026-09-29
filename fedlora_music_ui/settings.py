"""Persisted UI settings and the commands they produce. No Toga imports: unit-testable."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Settings:
    ace_project_root: str = ""
    python: str = ""  # runtime interpreter; blank -> <ace>/.venv/bin/python
    songs_dir: str = ""
    data_dir: str = ""
    superlink: str = "127.0.0.1:9092"
    ca_cert: str = ""
    insecure: bool = False  # local testing only: no TLS, no node auth
    epsilon_budget: float = 10.0
    min_noise_multiplier: float = 1.0
    max_dp_delta: float = 1e-5
    model_variant: str = "acestep-v15-turbo"
    runtime_port: int = 9094

    # -- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> Settings:
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    # -- derived paths -------------------------------------------------------
    @property
    def ace_root(self) -> Path:
        return Path(self.ace_project_root).expanduser()

    @property
    def data(self) -> Path:
        return Path(self.data_dir).expanduser()

    @property
    def runtime_python(self) -> Path:
        if self.python:
            return Path(self.python).expanduser()
        return self.ace_root / ".venv" / "bin" / "python"

    @property
    def supernode_exe(self) -> Path:
        return self.runtime_python.parent / "flower-supernode"

    @property
    def key_dir(self) -> Path:
        return self.data / "keys"

    @property
    def private_key(self) -> Path:
        return self.key_dir / "supernode_key"

    @property
    def node_config_path(self) -> Path:
        return self.data / "node_config.toml"

    @property
    def fused_adapter_dir(self) -> Path:
        return self.data / "export" / "fused_adapter"

    # -- validation ----------------------------------------------------------
    def problems(self, action: str) -> list[str]:
        """Human-readable blockers for ``action`` in {prepare, identity, join, generate}."""
        out: list[str] = []
        if not self.data_dir:
            out.append("Choose a data folder.")
        if action != "identity" and not (self.ace_root / "checkpoints").is_dir():
            out.append("ACE-Step folder must contain 'checkpoints' (run acestep-download).")
        if not self.runtime_python.is_file():
            out.append(f"Python runtime not found at {self.runtime_python}.")
        if action == "prepare" and not Path(self.songs_dir).expanduser().is_dir():
            out.append("Choose the folder with your songs.")
        if action == "join":
            if not self.supernode_exe.is_file():
                out.append(f"flower-supernode not found next to {self.runtime_python}.")
            if not (self.data / "tensors").is_dir():
                out.append("Prepare your songs first.")
            if not self.insecure:
                if not Path(self.ca_cert).expanduser().is_file():
                    out.append("Choose the server's CA certificate (or enable local testing).")
                if not self.private_key.is_file():
                    out.append("Create your identity key and send it to the operator.")
            if self.min_noise_multiplier <= 0:
                out.append("Minimum noise must be > 0.")
        if action == "generate" and not (self.fused_adapter_dir / "adapter_config.json").is_file():
            out.append("No personalised model yet: join and complete at least one round.")
        return out


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops -> CPU
    return env


def _toml_str(value: str) -> str:
    return json.dumps(value)  # JSON string escapes are valid TOML basic strings


def write_node_config(s: Settings) -> Path:
    """Client-owned policy handed to the ClientApp via ``--node-config <file>.toml``."""
    lines = [
        f"data-dir = {_toml_str(str(s.data.resolve()))}",
        f"ace-project-root = {_toml_str(str(s.ace_root.resolve()))}",
        f"epsilon-budget = {float(s.epsilon_budget)!r}",
        f"min-noise-multiplier = {float(s.min_noise_multiplier)!r}",
        f"max-dp-delta = {float(s.max_dp_delta)!r}",
    ]
    s.data.mkdir(parents=True, exist_ok=True)
    s.node_config_path.write_text("\n".join(lines) + "\n")
    return s.node_config_path


def prepare_cmd(s: Settings) -> list[str]:
    return [
        str(s.runtime_python), "-m", "fedlora_music.prepare",
        "--audio-dir", str(Path(s.songs_dir).expanduser()),
        "--client-dir", str(s.data),
        "--ace-project-root", str(s.ace_root),
        "--model-variant", s.model_variant,
    ]  # fmt: skip


def identity_cmd(s: Settings) -> list[str]:
    return [str(s.runtime_python), "-m", "fedlora_music.identity", "--out-dir", str(s.key_dir)]


def supernode_cmd(s: Settings) -> list[str]:
    cmd = [str(s.supernode_exe), "--superlink", s.superlink]
    if s.insecure:
        cmd.append("--insecure")
    else:
        cmd += ["--root-certificates", str(Path(s.ca_cert).expanduser())]
        cmd += ["--auth-supernode-private-key", str(s.private_key)]
    cmd += [
        "--node-config", str(s.node_config_path),
        "--host", "127.0.0.1",  # local runtime API: never expose on the network
        "--port", str(s.runtime_port),
    ]  # fmt: skip
    return cmd


def generate_cmd(s: Settings, caption: str, duration: float) -> list[str]:
    return [
        str(s.runtime_python), "-m", "fedlora_music.generate",
        "--client-dir", str(s.data),
        "--ace-project-root", str(s.ace_root),
        "--model-variant", s.model_variant,
        "--caption", caption,
        "--duration", str(duration),
    ]  # fmt: skip


@dataclass(frozen=True)
class Status:
    rounds: int
    epsilon: float
    last_loss: float | None


def read_status(s: Settings) -> Status:
    """Read local progress files only (no torch import)."""
    ledger = s.data / "state" / "privacy_ledger.json"
    log = s.data / "state" / "train_log.jsonl"
    rounds = len(json.loads(ledger.read_text())["noise_multipliers"]) if ledger.is_file() else 0
    epsilon, loss = 0.0, None
    if log.is_file():
        lines = log.read_text().strip().splitlines()
        if lines:
            last = json.loads(lines[-1])
            epsilon, loss = float(last.get("epsilon", 0.0)), last.get("loss")
    return Status(rounds=rounds, epsilon=epsilon, last_loss=loss)


def open_file_cmd(path: str) -> list[str]:
    return ["open", path] if sys.platform == "darwin" else ["xdg-open", path]
