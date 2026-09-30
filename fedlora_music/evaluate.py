"""Evaluate a client's adapters locally. Nothing here is sent to the server.

Compares four variants of the model on the client's own held-out data:

* ``base``      the base model with no adapter
* ``global``    the federated adapter only
* ``personal``  the client's own adapter only
* ``fused``     both (what ``fedlora-generate`` uses)

Metrics per variant (see ``fedlora_music.metrics``):

* held-out training loss (the backend's objective, fixed randomness); comparable
  within one model only
* style distance (KAD; FAD when there are enough clips) to the held-out songs
* prompt adherence (CLAP audio-text similarity)
* diversity (mean pairwise CLAP distance between generations)
* copy screen (nearest training song by CLAP similarity)
* optionally Audiobox Aesthetics scores (``--aesthetics``)

The CLAP metrics are independent of the model under test, so they compare models.
Held-out songs are prepared with ``fedlora-prepare --split eval``. Results go to
``<client-dir>/models/<backend>/<variant>/eval/runs/<timestamp>/`` (``report.json``,
``embeddings.safetensors``, generated audio).

    fedlora-eval --client-dir ./clients/client-0 --model-root ../ACE-Step-1.5
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from fedlora_music.adapters import GLOBAL, PERSONAL, StateDict, export_fused_adapter
from fedlora_music.backends import ModelBackend, get_backend
from fedlora_music.cli import add_model_args, model_spec
from fedlora_music.config import FedLoRAConfig, ModelSpec
from fedlora_music.metrics import (
    centroid_similarity,
    copy_check,
    frechet_distance,
    kernel_distance,
    pairwise_diversity,
    prompt_adherence,
)
from fedlora_music.store import ClientStore, ModelStore

logger = logging.getLogger(__name__)

VARIANTS: dict[str, tuple[str, ...]] = {
    "base": (),
    "global": (GLOBAL,),
    "personal": (PERSONAL,),
    "fused": (GLOBAL, PERSONAL),
}
DEFAULT_PROMPTS = (
    "warm lo-fi hip hop, dusty drums, rhodes piano",
    "upbeat indie rock, jangly guitars, driving drums",
    "ambient electronic, evolving pads, slow tempo",
    "acoustic folk, fingerpicked guitar, intimate",
)


def load_app_config(app_dir: Path) -> FedLoRAConfig:
    """Read the run config from the Flower app's ``pyproject.toml``."""
    data = tomllib.loads((app_dir / "pyproject.toml").read_text(encoding="utf-8"))
    return FedLoRAConfig.from_run_config(data["tool"]["flwr"]["app"]["config"])


def _zero_b(state: StateDict) -> StateDict:
    return StateDict((k, torch.zeros_like(v) if ".lora_B." in k else v) for k, v in state.items())


def variant_states(
    global_state: StateDict, personal_state: StateDict, variant: str
) -> tuple[StateDict, StateDict]:
    """Adapter pair whose fusion equals running only the variant's adapters.

    Zeroing ``B`` removes an adapter's contribution exactly, so every non-base
    variant can be exported as a standard fused PEFT adapter for generation.
    """
    active = VARIANTS[variant]
    if not active:
        raise ValueError("the base variant has no adapter")
    g = global_state if GLOBAL in active else _zero_b(global_state)
    p = personal_state if PERSONAL in active else _zero_b(personal_state)
    return g, p


def read_prompts(prompts_file: Path | None, eval_source: Path | None) -> list[str]:
    """``--prompts`` file > captions in the held-out dataset JSON > built-in prompts."""
    if prompts_file is not None:
        lines = prompts_file.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    if eval_source is not None and eval_source.suffix == ".json" and eval_source.is_file():
        raw = json.loads(eval_source.read_text(encoding="utf-8"))
        samples = raw if isinstance(raw, list) else raw.get("samples", [])
        captions = [s["caption"] for s in samples if s.get("caption")]
        if captions:
            return captions
    return list(DEFAULT_PROMPTS)


def score_variant(
    generated: torch.Tensor,
    prompts: torch.Tensor,
    reference: torch.Tensor | None,
    training: torch.Tensor | None,
    copy_threshold: float,
) -> dict[str, Any]:
    """All embedding-based metrics for one variant. ``prompts`` is row-aligned with clips."""
    out: dict[str, Any] = {
        "clips": len(generated),
        "prompt_adherence": prompt_adherence(generated, prompts),
        "diversity": pairwise_diversity(generated),
    }
    if reference is not None and len(reference) >= 2 and len(generated) >= 2:
        out["kad_to_reference"] = kernel_distance(generated, reference)
        out["centroid_similarity"] = centroid_similarity(generated, reference)
        dim = generated.shape[1]
        # A covariance estimate needs more clips than dimensions to mean anything.
        enough = min(len(generated), len(reference)) > dim
        out["fad_to_reference"] = frechet_distance(generated, reference) if enough else None
    if training is not None and len(training) > 0:
        out["copy"] = asdict(copy_check(generated, training, copy_threshold))
    return out


def _load_adapters(store: ModelStore) -> tuple[StateDict, StateDict]:
    g = store.load_tensors(store.local_global_path)
    p = store.load_tensors(store.personal_adapter_path)
    if g is None or p is None:
        raise FileNotFoundError(
            f"No trained adapters in {store.state_dir}; run at least one federated round first"
        )
    return StateDict(g), StateDict(p)


def _release_model() -> None:
    from fedlora_music.model import get_runtime

    get_runtime.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def heldout_losses(
    spec: ModelSpec,
    store: ModelStore,
    states: tuple[StateDict, StateDict],
    variants: Sequence[str],
    seed: int,
) -> dict[str, float]:
    from fedlora_music.adapters import load_adapter_state, only_adapters
    from fedlora_music.model import get_runtime
    from fedlora_music.trainer import heldout_loss

    rt = get_runtime(spec)
    load_adapter_state(rt.model, GLOBAL, states[0])
    load_adapter_state(rt.model, PERSONAL, states[1])
    loader = rt.backend.build_loader(store.eval_tensor_dir, batch_size=1, shuffle=False)
    losses = {}
    for variant in variants:
        with only_adapters(rt.model, VARIANTS[variant], root=rt.backend.adapter_root):
            losses[variant] = heldout_loss(rt, loader, seed=seed)
        logger.info("held-out loss %-8s %.5f", variant, losses[variant])
    del rt
    _release_model()
    return losses


def generate_variants(
    backend: ModelBackend,
    states: tuple[StateDict, StateDict],
    variants: Sequence[str],
    prompts: Sequence[str],
    samples_per_prompt: int,
    duration: float,
    seed: int,
    out_dir: Path,
) -> dict[str, list[tuple[Path, str]]]:
    """Generate ``samples_per_prompt`` clips per prompt per variant, with shared seeds."""
    adapters = backend.spec.adapters
    assert adapters is not None
    generator = backend.open_generator()
    clips: dict[str, list[tuple[Path, str]]] = {}
    try:
        for variant in variants:
            adapter_dir = None
            if VARIANTS[variant]:
                adapter_dir = export_fused_adapter(
                    *variant_states(*states, variant),
                    adapters,
                    out_dir / "adapters" / variant,
                    root=backend.adapter_root,
                )
            clips[variant] = []
            for i, prompt in enumerate(prompts):
                seeds = [seed + i * samples_per_prompt + j for j in range(samples_per_prompt)]
                paths = generator.generate(
                    adapter_dir, prompt, seeds, duration, out_dir / "audio" / variant
                )
                clips[variant].extend((p, prompt) for p in paths)
            logger.info("generated %d clips for %s", len(clips[variant]), variant)
    finally:
        generator.close()
    return clips


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--client-dir", type=Path, required=True)
    add_model_args(ap)
    ap.add_argument("--app-dir", type=Path, default=Path("."), help="Folder with pyproject.toml")
    ap.add_argument(
        "--reference-audio",
        type=Path,
        default=None,
        help="Held-out songs (folder or dataset JSON). Defaults to the `--split eval` source",
    )
    ap.add_argument(
        "--training-audio",
        type=Path,
        default=None,
        help="Training songs for the copy screen. Defaults to the `--split train` source",
    )
    ap.add_argument("--prompts", type=Path, default=None, help="One caption per line")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--samples-per-prompt", type=int, default=2)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-loss", action="store_true")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--aesthetics", action="store_true", help="Add Audiobox Aesthetics scores")
    ap.add_argument("--clap-checkpoint", default=None)
    ap.add_argument("--copy-threshold", type=float, default=0.95)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - VARIANTS.keys()
    if unknown:
        ap.error(f"unknown variants {sorted(unknown)}; choose from {list(VARIANTS)}")

    client = ClientStore(args.client_dir.expanduser().resolve())
    cfg = load_app_config(args.app_dir.expanduser().resolve())
    spec = model_spec(args, run=cfg.model)
    backend = get_backend(spec)
    store = client.model(backend.key)
    states = _load_adapters(store)
    sources = client.load_sources()

    def source(explicit: Path | None, split: str) -> Path | None:
        if explicit is not None:
            return explicit.expanduser().resolve()
        return Path(sources[split]) if split in sources else None

    reference_src, training_src = (
        source(args.reference_audio, "eval"),
        source(args.training_audio, "train"),
    )
    out_dir = (args.out_dir or store.eval_dir / "runs" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "client_dir": str(client.root),
        "created": time.time(),
        "model": backend.key,
        "variants": {v: {} for v in variants},
    }

    if not args.skip_loss:
        if store.eval_tensor_dir.is_dir():
            for v, loss in heldout_losses(spec, store, states, variants, args.seed).items():
                report["variants"][v]["heldout_loss"] = loss
        else:
            logger.warning(
                "No held-out tensors at %s; run `fedlora-prepare --split eval` to get a "
                "held-out loss",
                store.eval_tensor_dir,
            )

    if not args.skip_generation:
        from fedlora_music.embeddings import DEFAULT_CLAP, ClapEmbedder, list_audio

        prompts = read_prompts(args.prompts, reference_src)
        clips = generate_variants(
            backend,
            states,
            variants,
            prompts,
            args.samples_per_prompt,
            args.duration,
            args.seed,
            out_dir,
        )
        clap = ClapEmbedder(
            args.clap_checkpoint or DEFAULT_CLAP,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        embeddings: dict[str, torch.Tensor] = {}
        if reference_src is not None:
            embeddings["reference"] = clap.embed_audio(list_audio(reference_src))
        else:
            logger.warning("No held-out songs given; style metrics are skipped")
        if training_src is not None:
            embeddings["training"] = clap.embed_audio(list_audio(training_src))
        report["embedder"] = args.clap_checkpoint or DEFAULT_CLAP
        report["counts"] = {k: len(v) for k, v in embeddings.items()} | {"prompts": len(prompts)}

        for v in variants:
            paths = [p for p, _ in clips[v]]
            gen = embeddings[f"generated.{v}"] = clap.embed_audio(paths)
            text = embeddings[f"text.{v}"] = clap.embed_text([t for _, t in clips[v]])
            report["variants"][v].update(
                score_variant(
                    gen,
                    text,
                    embeddings.get("reference"),
                    embeddings.get("training"),
                    args.copy_threshold,
                )
            )
            if args.aesthetics:
                from fedlora_music.embeddings import aesthetics_scores

                rows = aesthetics_scores(paths)
                report["variants"][v]["aesthetics"] = {
                    k: sum(r[k] for r in rows) / len(rows) for k in rows[0]
                }

        from safetensors.torch import save_file

        save_file(
            {k: v.contiguous() for k, v in embeddings.items()},
            str(out_dir / "embeddings.safetensors"),
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "report.json")
    print(out_dir / "report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
