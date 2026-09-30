"""Typed configuration.

Two sources, with different trust:

* **Run config** (``[tool.flwr.app.config]``) - set by whoever submits the run,
  i.e. the federation operator. Shared by all clients.
* **Node config** (``flower-supernode --node-config``) - set by the client on its
  own machine. Holds local paths, the client's privacy floor and the model backends
  it is willing to run. The run config can make privacy stricter than the node's
  floor, never looser, and can only pick a backend the node allows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterSpec:
    """Shape of the two LoRA adapters. Hashable so it can key the model cache."""

    target_modules: tuple[str, ...]
    global_rank: int
    global_alpha: int
    personal_rank: int
    personal_alpha: int
    lora_dropout: float
    freeze_global_a: bool

    def __post_init__(self) -> None:
        if not self.target_modules:
            raise ValueError("target-modules must name at least one module")
        for name, value in (
            ("global-rank", self.global_rank),
            ("personal-rank", self.personal_rank),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")


# Backends shipped with this package. Only these run unless a client's node config
# lists others in ``allowed-backends`` (see ``LocalPolicy``).
BUILTIN_BACKENDS: frozenset[str] = frozenset({"acestep", "toy"})


def _legacy_root(cfg: Mapping[str, Any]) -> Any:
    """``model-root``, falling back to the pre-backend key ``ace-project-root``."""
    return cfg.get("model-root") or cfg.get("ace-project-root") or None


@dataclass(frozen=True)
class ModelSpec:
    """Which base model to load. Hashable so it can key the model cache.

    ``backend`` names a model backend (see ``fedlora_music.backends``). ``root`` is
    the backend's install folder (ACE-Step: the checkout that holds ``checkpoints/``)
    and ``variant`` a model inside it; an empty variant means the backend's default.
    """

    backend: str
    root: Path | None
    variant: str
    adapters: AdapterSpec | None = None  # not needed to prepare data or generate


@dataclass(frozen=True)
class PrivacySpec:
    """Client-level differential privacy parameters (applied on the client)."""

    clip_norm: float
    noise_multiplier: float
    delta: float
    epsilon_budget: float

    def __post_init__(self) -> None:
        if self.clip_norm <= 0:
            raise ValueError("dp-clip-norm must be > 0")
        if self.noise_multiplier <= 0:
            raise ValueError("dp-noise-multiplier must be > 0 (0 would disable DP)")
        if not 0 < self.delta < 1:
            raise ValueError("dp-delta must be in (0, 1)")


@dataclass(frozen=True)
class TrainSpec:
    local_steps: int
    batch_size: int
    lr_global: float
    lr_personal: float
    grad_clip: float
    cfg_ratio: float


@dataclass(frozen=True)
class FedLoRAConfig:
    model: ModelSpec
    privacy: PrivacySpec
    train: TrainSpec
    clients_root: Path
    server_output_dir: Path
    num_server_rounds: int
    fraction_train: float
    min_train_nodes: int
    min_available_nodes: int
    seed: int

    @classmethod
    def from_run_config(cls, rc: Mapping[str, Any]) -> FedLoRAConfig:
        adapters = AdapterSpec(
            target_modules=tuple(
                m.strip() for m in str(rc["target-modules"]).split(",") if m.strip()
            ),
            global_rank=int(rc["global-rank"]),
            global_alpha=int(rc["global-alpha"]),
            personal_rank=int(rc["personal-rank"]),
            personal_alpha=int(rc["personal-alpha"]),
            lora_dropout=float(rc["lora-dropout"]),
            freeze_global_a=bool(rc["freeze-global-a"]),
        )
        root = _legacy_root(rc)
        return cls(
            model=ModelSpec(
                backend=str(rc.get("model-backend", "acestep")),
                root=Path(str(root)).expanduser().resolve() if root else None,
                variant=str(rc.get("model-variant", "")),
                adapters=adapters,
            ),
            privacy=PrivacySpec(
                clip_norm=float(rc["dp-clip-norm"]),
                noise_multiplier=float(rc["dp-noise-multiplier"]),
                delta=float(rc["dp-delta"]),
                epsilon_budget=float(rc["dp-epsilon-budget"]),
            ),
            train=TrainSpec(
                local_steps=int(rc["local-steps"]),
                batch_size=int(rc["batch-size"]),
                lr_global=float(rc["lr-global"]),
                lr_personal=float(rc["lr-personal"]),
                grad_clip=float(rc["grad-clip"]),
                cfg_ratio=float(rc["cfg-ratio"]),
            ),
            clients_root=Path(str(rc["clients-root"])).expanduser().resolve(),
            server_output_dir=Path(str(rc["server-output-dir"])).expanduser().resolve(),
            num_server_rounds=int(rc["num-server-rounds"]),
            fraction_train=float(rc["fraction-train"]),
            min_train_nodes=int(rc["min-train-nodes"]),
            min_available_nodes=int(rc["min-available-nodes"]),
            seed=int(rc["seed"]),
        )


@dataclass(frozen=True)
class LocalPolicy:
    """Client-owned settings read from ``context.node_config``.

    Keys (all optional; simulation passes only ``partition-id``):
      ``data-dir``              this client's data folder (deployment mode)
      ``model-root``            local model install (``ace-project-root`` still works)
      ``allowed-backends``      comma-separated backends this client will run;
                                default: the built-in ones
      ``epsilon-budget``        max total epsilon this client will ever spend
      ``min-noise-multiplier``  lowest sigma this client accepts
      ``max-dp-delta``          largest delta this client accepts
    """

    data_dir: Path | None = None
    model_root: Path | None = None
    allowed_backends: frozenset[str] = BUILTIN_BACKENDS
    epsilon_budget: float | None = None
    min_noise_multiplier: float = 0.0
    max_delta: float = 1.0

    @classmethod
    def from_node_config(cls, nc: Mapping[str, Any]) -> LocalPolicy:
        def path(value: Any) -> Path | None:
            return Path(str(value)).expanduser().resolve() if value else None

        allowed = BUILTIN_BACKENDS
        if "allowed-backends" in nc:
            allowed = frozenset(
                b.strip() for b in str(nc["allowed-backends"]).split(",") if b.strip()
            )
        return cls(
            data_dir=path(nc.get("data-dir")),
            model_root=path(_legacy_root(nc)),
            allowed_backends=allowed,
            epsilon_budget=float(nc["epsilon-budget"]) if "epsilon-budget" in nc else None,
            min_noise_multiplier=float(nc.get("min-noise-multiplier", 0.0)),
            max_delta=float(nc.get("max-dp-delta", 1.0)),
        )

    def effective_privacy(self, run: PrivacySpec) -> PrivacySpec:
        """Take the stricter of operator and client settings on every axis."""
        budget = run.epsilon_budget
        if self.epsilon_budget is not None:
            budget = min(budget, self.epsilon_budget)
        return PrivacySpec(
            clip_norm=run.clip_norm,  # does not affect epsilon, only utility
            noise_multiplier=max(run.noise_multiplier, self.min_noise_multiplier),
            delta=min(run.delta, self.max_delta),
            epsilon_budget=budget,
        )

    def effective_model(self, run: ModelSpec) -> ModelSpec:
        """The operator's model choice, if this client allows it, with local paths.

        Checked before any backend code is imported: the run config comes from the
        operator, and naming a backend means choosing code that runs next to this
        client's data.
        """
        if run.backend not in self.allowed_backends:
            raise BackendNotAllowedError(
                f"model backend {run.backend!r} is not in this client's allowed-backends "
                f"({', '.join(sorted(self.allowed_backends)) or 'none'})"
            )
        if self.model_root is None:
            return run
        return replace(run, root=self.model_root)


class BackendNotAllowedError(PermissionError):
    """The operator asked for a model backend the client has not allowed."""
