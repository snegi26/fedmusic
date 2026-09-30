"""Flower ServerApp: FedAvg over the DP-noised global adapter only.

The server never receives data, sample counts, losses, or personal adapters, and it
cannot remove client-side noise. It holds only the shared ``global`` adapter, and
sends the base model's fingerprint so clients with other weights refuse to train.
"""

from __future__ import annotations

import logging

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from safetensors.torch import save_file

from fedlora_music.adapters import GLOBAL, adapter_state
from fedlora_music.backends import get_backend
from fedlora_music.client_app import DP_WEIGHT_KEY, FINGERPRINT_KEY
from fedlora_music.config import FedLoRAConfig
from fedlora_music.model import load_base_with_adapters

logger = logging.getLogger(__name__)

app = ServerApp()


def initial_global_adapter(cfg: FedLoRAConfig) -> ArrayRecord:
    """Build the shared initial adapter (seeded A, zero B) from the public base model.

    Loads on CPU once to discover module shapes; this also fixes the shared random
    ``A`` matrices used by FFA-LoRA. Clients receive ``A`` in this record, so they
    never need to reproduce the random draw themselves.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model = load_base_with_adapters(cfg.model, torch.device("cpu"), "fp32")
    state = adapter_state(model, GLOBAL)
    del model
    return ArrayRecord(torch_state_dict=state)


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = FedLoRAConfig.from_run_config(context.run_config)
    backend = get_backend(cfg.model)
    logger.info("Model: %s (weights license: %s)", backend.key, backend.weights_license)

    strategy = FedAvg(
        fraction_train=cfg.fraction_train,
        fraction_evaluate=0.0,  # evaluation happens locally, by ear, on each client
        min_train_nodes=cfg.min_train_nodes,
        min_available_nodes=cfg.min_available_nodes,
        weighted_by_key=DP_WEIGHT_KEY,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_global_adapter(cfg),
        num_rounds=cfg.num_server_rounds,
        train_config=ConfigRecord({FINGERPRINT_KEY: backend.fingerprint()}),
    )

    cfg.server_output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.server_output_dir / "global_adapter.safetensors"
    state = {k: v.contiguous() for k, v in result.arrays.to_torch_state_dict().items()}
    save_file(state, str(out))
    logger.info("Saved final global adapter to %s", out)
