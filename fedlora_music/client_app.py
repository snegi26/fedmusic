"""Flower ClientApp: trains locally, releases only a DP-noised global-adapter update.

Data flow per round (nothing else leaves the client):

    server --(global adapter)--> client
    client: train global + personal on local tensors
    client: delta = clip(global_new - global_received) + N(0, (sigma*C)^2)
    client --(global_received + noised delta, dp metadata)--> server

Raw audio, latents, captions, sample counts, losses and the personal adapter all
stay in the client's data folder (node config ``data-dir`` in deployment,
``<clients-root>/client-<partition-id>`` in simulation).

Privacy parameters are the stricter of the operator's run config and the client's
own node config (see ``LocalPolicy``), and the budget check happens before any
data is read.
"""

from __future__ import annotations

import logging
import secrets
from collections import OrderedDict

from flwr.app import ArrayRecord, Context, Error, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedlora_music.adapters import (
    GLOBAL,
    PERSONAL,
    adapter_state,
    export_fused_adapter,
    load_adapter_state,
    reset_adapter,
    trainable_names,
)
from fedlora_music.config import FedLoRAConfig, LocalPolicy
from fedlora_music.model import get_runtime
from fedlora_music.privacy import PrivacyLedger, privatize_update
from fedlora_music.store import ClientStore
from fedlora_music.trainer import build_loader, train_local

logger = logging.getLogger(__name__)

ERR_BUDGET_EXHAUSTED = 4001
DP_WEIGHT_KEY = "dp-weight"

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = FedLoRAConfig.from_run_config(context.run_config)
    policy = LocalPolicy.from_node_config(context.node_config)
    privacy = policy.effective_privacy(cfg.privacy)
    store = ClientStore.resolve(policy.data_dir, cfg.clients_root, context.node_config)

    # 1. Refuse before touching data if this round would exceed the local budget.
    ledger = PrivacyLedger.load(store.ledger_path)
    eps_next = ledger.epsilon(privacy.delta, extra=privacy.noise_multiplier)
    if eps_next > privacy.epsilon_budget:
        logger.warning("%s: privacy budget exhausted (eps would be %.2f)", store.root, eps_next)
        return Message(Error(ERR_BUDGET_EXHAUSTED, "client privacy budget exhausted"), reply_to=msg)

    rt = get_runtime(policy.effective_model(cfg.model))
    model = rt.model

    # 2. Install the received global adapter and this client's own personal adapter.
    received = msg.content["arrays"].to_torch_state_dict()
    load_adapter_state(model, GLOBAL, received)
    personal = store.load_tensors(store.personal_adapter_path)
    if personal is None:
        reset_adapter(model, PERSONAL, seed=secrets.randbits(63))
    else:
        load_adapter_state(model, PERSONAL, personal)

    # 3. Local training on local data only.
    loader = build_loader(store.tensor_dir, cfg.train.batch_size)
    loss = train_local(rt, loader, cfg.train)

    # 4. Privatise the global update. Frozen coords (global A under FFA) are public
    #    and unchanged, so they are neither counted in the norm nor noised.
    trained_global = adapter_state(model, GLOBAL)
    released_keys = trainable_names(model, GLOBAL)
    delta = {k: trained_global[k] - received[k] for k in released_keys}
    noised, pre_clip_norm = privatize_update(delta, privacy.clip_norm, privacy.noise_multiplier)
    released = OrderedDict(
        (k, received[k] + noised[k] if k in noised else received[k].clone()) for k in received
    )

    # 5. Persist local state (never transmitted).
    trained_personal = adapter_state(model, PERSONAL)
    store.save_tensors(store.personal_adapter_path, trained_personal)
    store.save_tensors(store.local_global_path, trained_global)
    export_fused_adapter(
        trained_global, trained_personal, cfg.model.adapters, store.fused_adapter_dir
    )
    ledger.record(privacy.noise_multiplier)
    ledger.save(store.ledger_path)
    store.append_log(
        {
            "round": int(msg.content["config"].get("server-round", -1)),
            "loss": loss,
            "pre_clip_norm": pre_clip_norm,
            "epsilon": eps_next,
            "noise_multiplier": privacy.noise_multiplier,
        }
    )

    # 6. Reply. Uniform weight: sample counts would leak data size and break the
    #    per-client sensitivity bound. Epsilon derives only from public parameters.
    metrics = MetricRecord({DP_WEIGHT_KEY: 1, "epsilon": eps_next})
    content = RecordDict({"arrays": ArrayRecord(torch_state_dict=released), "metrics": metrics})
    return Message(content=content, reply_to=msg)
