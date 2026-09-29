"""End-to-end federated round on the toy model: what leaves the client, and when it refuses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import Toy, toy_loss
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.supercore.task_identity import TaskIdentity

from fedlora_music import client_app, server_app, trainer
from fedlora_music.adapters import GLOBAL, PERSONAL, is_param_of
from fedlora_music.client_app import DP_WEIGHT_KEY, ERR_BUDGET_EXHAUSTED
from fedlora_music.config import FedLoRAConfig
from fedlora_music.model import Runtime
from fedlora_music.privacy import l2_norm
from fedlora_music.store import ClientStore


@pytest.fixture
def fake_ace(
    monkeypatch: pytest.MonkeyPatch, toy_batches: list[dict[str, torch.Tensor]]
) -> dict[str, int]:
    """Swap ACE-Step out for the toy model and count how often data is touched."""
    # Normally set by Flower's runtime before a ClientApp runs; Message() reads it.
    for attr in ("_task_id", "_run_id", "_node_id"):
        monkeypatch.setattr(TaskIdentity, attr, 1)
    calls = {"runtime": 0, "loader": 0}
    runtimes: dict[Any, Runtime] = {}

    def load_toy(spec: Any, device: torch.device, precision: str) -> torch.nn.Module:
        from fedlora_music.adapters import inject_dual_lora

        return inject_dual_lora(Toy(), spec.adapters)

    def get_runtime(spec: Any) -> Runtime:
        calls["runtime"] += 1
        if spec not in runtimes:  # process-wide cache, like the real lru_cache
            torch.manual_seed(1)
            model = load_toy(spec, torch.device("cpu"), "fp32")
            runtimes[spec] = Runtime(model, torch.device("cpu"), torch.float32, -0.4, 1.0, 0.0)
        return runtimes[spec]

    def build_loader(tensor_dir: Path, batch_size: int) -> list[dict[str, torch.Tensor]]:
        calls["loader"] += 1
        return toy_batches

    monkeypatch.setattr(server_app, "load_base_with_adapters", load_toy)
    monkeypatch.setattr(client_app, "get_runtime", get_runtime)
    monkeypatch.setattr(client_app, "build_loader", build_loader)
    monkeypatch.setattr(trainer, "flow_matching_loss", toy_loss)
    return calls


def _round(
    run_config: dict[str, Any], node_config: dict[str, Any], arrays: ArrayRecord, rnd: int
) -> Message:
    msg = Message(
        content=RecordDict({"arrays": arrays, "config": ConfigRecord({"server-round": rnd})}),
        dst_node_id=1,
        message_type="train",
    )
    ctx = Context(
        run_id=1, node_id=1, node_config=node_config, state=RecordDict(), run_config=run_config
    )
    return client_app.train(msg, ctx)


def _initial(run_config: dict[str, Any]) -> ArrayRecord:
    return server_app.initial_global_adapter(FedLoRAConfig.from_run_config(run_config))


def test_initial_adapter_is_global_only_and_seeded(
    fake_ace: dict[str, int], run_config: dict[str, Any]
) -> None:
    a = _initial(run_config).to_torch_state_dict()
    b = _initial(run_config).to_torch_state_dict()
    assert a and all(is_param_of(k, GLOBAL) for k in a)
    assert all(torch.equal(a[k], b[k]) for k in a)  # every client gets the same shared A
    assert all(not v.any() for k, v in a.items() if ".lora_B." in k)  # B starts at zero


def test_round_releases_only_clipped_global_update(
    fake_ace: dict[str, int], run_config: dict[str, Any]
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


def test_local_state_persists_and_ledger_counts_rounds(
    fake_ace: dict[str, int], run_config: dict[str, Any], tmp_path: Path
) -> None:
    arrays = _initial(run_config)
    for rnd in (1, 2):
        arrays = _round(run_config, {"partition-id": 0}, arrays, rnd).content["arrays"]

    store = ClientStore.for_partition(tmp_path / "clients", 0)
    assert store.personal_adapter_path.is_file()
    assert (store.fused_adapter_dir / "adapter_config.json").is_file()
    ledger = json.loads(store.ledger_path.read_text())
    assert len(ledger["noise_multipliers"]) == 2
    log = [json.loads(line) for line in store.log_path.read_text().splitlines()]
    assert [r["round"] for r in log] == [1, 2]
    assert all(math.isfinite(r["loss"]) for r in log)
    assert log[1]["epsilon"] > log[0]["epsilon"]


def test_client_noise_floor_beats_operator(
    fake_ace: dict[str, int], run_config: dict[str, Any], tmp_path: Path
) -> None:
    run_config |= {"dp-noise-multiplier": 1.0, "dp-epsilon-budget": 1e9}
    _round(run_config, {"partition-id": 0, "min-noise-multiplier": 7.0}, _initial(run_config), 1)
    ledger = json.loads(ClientStore.for_partition(tmp_path / "clients", 0).ledger_path.read_text())
    assert ledger["noise_multipliers"] == [7.0]


def test_exhausted_budget_refuses_before_touching_data(
    fake_ace: dict[str, int], run_config: dict[str, Any]
) -> None:
    run_config |= {"dp-noise-multiplier": 1.0, "dp-epsilon-budget": 0.5}
    reply = _round(run_config, {"partition-id": 0}, _initial(run_config), 1)
    assert reply.has_error()
    assert reply.error.code == ERR_BUDGET_EXHAUSTED
    assert fake_ace == {"runtime": 0, "loader": 0}
