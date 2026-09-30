"""Federated rounds on the toy backend: what leaves the client, and when it refuses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import make_songs
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.supercore.task_identity import TaskIdentity

from fedlora_music import client_app, prepare, server_app
from fedlora_music.adapters import GLOBAL, PERSONAL, is_param_of
from fedlora_music.backends import get_backend
from fedlora_music.backends.toy import ToyBackend
from fedlora_music.client_app import (
    DP_WEIGHT_KEY,
    ERR_BACKEND_NOT_ALLOWED,
    ERR_BUDGET_EXHAUSTED,
    ERR_MODEL_MISMATCH,
    FINGERPRINT_KEY,
)
from fedlora_music.config import FedLoRAConfig
from fedlora_music.privacy import l2_norm
from fedlora_music.store import ClientStore


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, int]:
    """Prepare client 0's songs, and count every model load and data read."""
    # Normally set by Flower's runtime before a ClientApp runs; Message() reads it.
    for attr in ("_task_id", "_run_id", "_node_id"):
        monkeypatch.setattr(TaskIdentity, attr, 1)
    songs = make_songs(tmp_path / "songs")
    client = tmp_path / "clients" / "client-0"
    assert prepare.main(["--audio-dir", str(songs), "--client-dir", str(client),
                         "--model-backend", "toy"]) == 0  # fmt: skip

    counts = {"runtime": 0, "loader": 0}
    real_runtime, real_loader = client_app.get_runtime, ToyBackend.build_loader

    def get_runtime(spec: Any) -> Any:
        counts["runtime"] += 1
        return real_runtime(spec)

    def build_loader(self: ToyBackend, *args: Any, **kwargs: Any) -> Any:
        counts["loader"] += 1
        return real_loader(self, *args, **kwargs)

    monkeypatch.setattr(client_app, "get_runtime", get_runtime)
    monkeypatch.setattr(ToyBackend, "build_loader", build_loader)
    return counts


def _round(
    run_config: dict[str, Any],
    node_config: dict[str, Any],
    arrays: ArrayRecord,
    rnd: int,
    fingerprint: str | None = None,
) -> Message:
    cfg = FedLoRAConfig.from_run_config(run_config)
    config = {
        "server-round": rnd,
        FINGERPRINT_KEY: fingerprint or ToyBackend(cfg.model).fingerprint(),
    }
    msg = Message(
        content=RecordDict({"arrays": arrays, "config": ConfigRecord(config)}),
        dst_node_id=1,
        message_type="train",
    )
    ctx = Context(
        run_id=1, node_id=1, node_config=node_config, state=RecordDict(), run_config=run_config
    )
    return client_app.train(msg, ctx)


def _initial(run_config: dict[str, Any]) -> ArrayRecord:
    return server_app.initial_global_adapter(FedLoRAConfig.from_run_config(run_config))


def test_initial_adapter_is_global_only_and_seeded(run_config: dict[str, Any]) -> None:
    a = _initial(run_config).to_torch_state_dict()
    b = _initial(run_config).to_torch_state_dict()
    assert a and all(is_param_of(k, GLOBAL) for k in a)
    assert all(torch.equal(a[k], b[k]) for k in a)  # every client gets the same shared A
    assert all(not v.any() for k, v in a.items() if ".lora_B." in k)  # B starts at zero


def test_round_releases_only_clipped_global_update(
    calls: dict[str, int], run_config: dict[str, Any]
) -> None:
    clip = 1e-3
    # Near-zero noise isolates clipping; its epsilon is huge, so lift the budget.
    run_config |= {"dp-clip-norm": clip, "dp-noise-multiplier": 1e-9, "dp-epsilon-budget": math.inf}
    initial = _initial(run_config)
    received = initial.to_torch_state_dict()

    reply = _round(run_config, {"partition-id": 0}, initial, rnd=1)

    assert not reply.has_error()
    assert set(reply.content.keys()) == {"arrays", "metrics"}
    released = reply.content["arrays"].to_torch_state_dict()
    assert released.keys() == received.keys()
    assert not any(is_param_of(k, PERSONAL) for k in released)
    assert set(reply.content["metrics"].keys()) == {DP_WEIGHT_KEY, "epsilon"}
    assert reply.content["metrics"][DP_WEIGHT_KEY] == 1  # no sample counts

    # FFA-LoRA: the shared A is public and frozen, so it comes back unchanged.
    for k in released:
        if ".lora_A." in k:
            assert torch.equal(released[k], received[k])
    delta = [released[k] - received[k] for k in released if ".lora_B." in k]
    assert math.isclose(l2_norm(delta), clip, rel_tol=1e-3)


def test_model_state_is_per_model_and_ledger_is_shared(
    calls: dict[str, int], run_config: dict[str, Any], tmp_path: Path
) -> None:
    arrays = _initial(run_config)
    for rnd in (1, 2):
        arrays = _round(run_config, {"partition-id": 0}, arrays, rnd).content["arrays"]

    client = ClientStore.for_partition(tmp_path / "clients", 0)
    local = client.model("toy/tiny")
    assert local.personal_adapter_path.is_file()
    assert (local.fused_adapter_dir / "adapter_config.json").is_file()
    # The ledger sits outside models/: switching models must not reset epsilon.
    assert client.ledger_path == client.root / "state" / "privacy_ledger.json"
    assert len(json.loads(client.ledger_path.read_text())["noise_multipliers"]) == 2
    log = [json.loads(line) for line in local.log_path.read_text().splitlines()]
    assert [r["round"] for r in log] == [1, 2]
    assert all(math.isfinite(r["loss"]) for r in log)
    assert log[1]["epsilon"] > log[0]["epsilon"]


def test_client_noise_floor_beats_operator(
    calls: dict[str, int], run_config: dict[str, Any], tmp_path: Path
) -> None:
    run_config |= {"dp-noise-multiplier": 1.0, "dp-epsilon-budget": 1e9}
    _round(run_config, {"partition-id": 0, "min-noise-multiplier": 7.0}, _initial(run_config), 1)
    ledger = json.loads(ClientStore.for_partition(tmp_path / "clients", 0).ledger_path.read_text())
    assert ledger["noise_multipliers"] == [7.0]


def test_exhausted_budget_refuses_before_touching_data(
    calls: dict[str, int], run_config: dict[str, Any]
) -> None:
    run_config |= {"dp-noise-multiplier": 1.0, "dp-epsilon-budget": 0.5}
    reply = _round(run_config, {"partition-id": 0}, _initial(run_config), 1)
    assert reply.has_error()
    assert reply.error.code == ERR_BUDGET_EXHAUSTED
    assert calls == {"runtime": 0, "loader": 0}


def test_backend_not_in_client_allowlist_is_refused(
    calls: dict[str, int], run_config: dict[str, Any], tmp_path: Path
) -> None:
    node = {"partition-id": 0, "allowed-backends": "acestep"}
    reply = _round(run_config, node, _initial(run_config), 1)
    assert reply.has_error()
    assert reply.error.code == ERR_BACKEND_NOT_ALLOWED
    assert calls == {"runtime": 0, "loader": 0}
    assert not ClientStore.for_partition(tmp_path / "clients", 0).ledger_path.exists()


def test_different_base_weights_are_refused(
    calls: dict[str, int], run_config: dict[str, Any]
) -> None:
    reply = _round(run_config, {"partition-id": 0}, _initial(run_config), 1, fingerprint="other")
    assert reply.has_error()
    assert reply.error.code == ERR_MODEL_MISMATCH
    assert calls == {"runtime": 0, "loader": 0}


def test_server_sends_the_backend_fingerprint(run_config: dict[str, Any]) -> None:
    spec = FedLoRAConfig.from_run_config(run_config).model
    assert get_backend(spec).fingerprint() == ToyBackend(spec).fingerprint()
    assert len(get_backend(spec).fingerprint()) == 64
