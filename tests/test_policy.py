"""Client policy must only ever tighten privacy relative to the operator's run config."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedlora_music.config import AdapterSpec, LocalPolicy, ModelSpec, PrivacySpec
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


def test_local_ace_root_overrides_run(tmp_path: Path) -> None:
    spec = ModelSpec(
        ace_project_root=Path("/operator/path"),
        model_variant="acestep-v15-turbo",
        adapters=AdapterSpec(("q_proj",), 16, 16, 8, 8, 0.0, True),
    )
    policy = LocalPolicy.from_node_config({"ace-project-root": str(tmp_path)})
    assert policy.effective_model(spec).ace_project_root == tmp_path.resolve()
    assert LocalPolicy().effective_model(spec) is spec
