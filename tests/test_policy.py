"""Client policy must only ever tighten privacy relative to the operator's run config."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedlora_music.config import (
    AdapterSpec,
    BackendNotAllowedError,
    LocalPolicy,
    ModelSpec,
    PrivacySpec,
)
from fedlora_music.store import ClientStore

RUN = PrivacySpec(clip_norm=0.05, noise_multiplier=4.0, delta=1e-5, epsilon_budget=20.0)


def test_empty_policy_keeps_run_settings() -> None:
    assert LocalPolicy.from_node_config({"partition-id": 0}).effective_privacy(RUN) == RUN


def test_operator_cannot_loosen_client_floor() -> None:
    policy = LocalPolicy.from_node_config(
        {"epsilon-budget": 5.0, "min-noise-multiplier": 6.0, "max-dp-delta": 1e-6}
    )
    eff = policy.effective_privacy(RUN)
    assert eff.noise_multiplier == 6.0
    assert eff.epsilon_budget == 5.0
    assert eff.delta == 1e-6
    assert eff.clip_norm == RUN.clip_norm


def test_operator_can_tighten() -> None:
    policy = LocalPolicy.from_node_config({"epsilon-budget": 50.0, "min-noise-multiplier": 1.0})
    eff = policy.effective_privacy(RUN)
    assert eff.noise_multiplier == 4.0 and eff.epsilon_budget == 20.0


def test_store_resolution(tmp_path: Path) -> None:
    root = tmp_path / "clients"
    assert ClientStore.resolve(None, root, {"partition-id": 3}).root == root / "client-3"
    policy = LocalPolicy.from_node_config({"data-dir": str(tmp_path / "mine")})
    assert ClientStore.resolve(policy.data_dir, root, {}).root == (tmp_path / "mine").resolve()
    with pytest.raises(ValueError):
        ClientStore.resolve(None, root, {})


def _spec(backend: str = "acestep") -> ModelSpec:
    return ModelSpec(
        backend=backend,
        root=Path("/operator/path"),
        variant="",
        adapters=AdapterSpec(("q_proj",), 16, 16, 8, 8, 0.0, True),
    )


def test_local_model_root_overrides_run(tmp_path: Path) -> None:
    for key in ("model-root", "ace-project-root"):  # the old key still works
        policy = LocalPolicy.from_node_config({key: str(tmp_path)})
        assert policy.effective_model(_spec()).root == tmp_path.resolve()
    assert LocalPolicy().effective_model(_spec()) == _spec()


def test_builtin_backends_allowed_by_default_plugins_are_not() -> None:
    assert LocalPolicy().effective_model(_spec("toy")).backend == "toy"
    with pytest.raises(BackendNotAllowedError):
        LocalPolicy().effective_model(_spec("somebody-elses-plugin"))


def test_client_allowlist_narrows_or_opts_in() -> None:
    only_ace = LocalPolicy.from_node_config({"allowed-backends": "acestep"})
    with pytest.raises(BackendNotAllowedError, match="allowed-backends"):
        only_ace.effective_model(_spec("toy"))
    plugin = LocalPolicy.from_node_config({"allowed-backends": "acestep, my-plugin"})
    assert plugin.effective_model(_spec("my-plugin")).backend == "my-plugin"


def test_backend_names_are_never_import_paths() -> None:
    from fedlora_music.backends import backend_class

    with pytest.raises(KeyError):
        backend_class("os:system")
    assert backend_class("toy").name == "toy"
