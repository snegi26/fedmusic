"""UI command building and node-config output (no Toga or torch needed)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from fedlora_music_ui.settings import (
    Settings,
    read_status,
    supernode_cmd,
    write_node_config,
)


def _settings(tmp_path: Path, **kw: object) -> Settings:
    return Settings(
        ace_project_root=str(tmp_path / "ace"),
        data_dir=str(tmp_path / "My Data"),  # space on purpose
        ca_cert=str(tmp_path / "ca.crt"),
        **kw,
    )


def test_node_config_is_valid_toml_with_client_policy(tmp_path: Path) -> None:
    s = _settings(tmp_path, epsilon_budget=7.5, min_noise_multiplier=3.0)
    cfg = tomllib.loads(write_node_config(s).read_text())
    assert cfg["data-dir"] == str((tmp_path / "My Data").resolve())
    assert cfg["epsilon-budget"] == 7.5
    assert cfg["min-noise-multiplier"] == 3.0
    assert cfg["max-dp-delta"] == 1e-5
    assert cfg["model-root"] == str((tmp_path / "ace").resolve())
    assert cfg["allowed-backends"] == "acestep"  # only the model this client set up


def test_supernode_cmd_secure_by_default(tmp_path: Path) -> None:
    cmd = supernode_cmd(_settings(tmp_path))
    assert "--insecure" not in cmd
    assert cmd[cmd.index("--root-certificates") + 1].endswith("ca.crt")
    assert cmd[cmd.index("--auth-supernode-private-key") + 1].endswith("supernode_key")
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--node-config") + 1].endswith("node_config.toml")


def test_supernode_cmd_insecure_drops_tls_and_auth(tmp_path: Path) -> None:
    cmd = supernode_cmd(_settings(tmp_path, insecure=True))
    assert "--insecure" in cmd
    assert "--root-certificates" not in cmd and "--auth-supernode-private-key" not in cmd


def test_problems_flag_missing_pieces(tmp_path: Path) -> None:
    probs = " ".join(_settings(tmp_path).problems("join"))
    assert "checkpoints" in probs and "CA certificate" in probs and "identity" in probs


def test_settings_roundtrip_ignores_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    _settings(tmp_path, superlink="fl.example.org:9092").save(path)
    raw = json.loads(path.read_text())
    raw["future_field"] = 1
    path.write_text(json.dumps(raw))
    assert Settings.load(path).superlink == "fl.example.org:9092"


def test_read_status_spans_models(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    (s.data / "state").mkdir(parents=True)
    (s.data / "state" / "privacy_ledger.json").write_text(
        json.dumps({"noise_multipliers": [4.0, 4.0, 4.0]})
    )
    other = s.data / "models" / "toy" / "tiny" / "state"
    other.mkdir(parents=True)
    (other / "train_log.jsonl").write_text(json.dumps({"ts": 2, "epsilon": 4.1, "loss": 9}) + "\n")
    (s.model_dir / "state").mkdir(parents=True)
    (s.model_dir / "state" / "train_log.jsonl").write_text(
        json.dumps({"ts": 1, "epsilon": 3.2, "loss": 0.5}) + "\n"
    )
    st = read_status(s)
    # Epsilon is the latest total across models; the loss is the chosen model's.
    assert (st.rounds, st.epsilon, st.last_loss) == (3, 4.1, 0.5)
    assert s.model_dir == s.data / "models" / "acestep" / "acestep-v15-turbo"
