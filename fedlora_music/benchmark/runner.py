"""Privacy-vs-quality benchmark over simulated clients (see ``fedlora_music.benchmark.fma``).

Stages, each resumable (finished work is skipped):

1. ``prepare``  ``fedlora-prepare`` each client's train and eval splits once.
2. ``train``    one ``flwr run`` simulation per noise multiplier sigma, each with fresh
                client state that links to the shared prepared tensors.
3. ``eval``     ``fedlora-eval`` for every client of every run.
4. ``report``   ``results.csv`` and ``results.md`` in the bench folder: per sigma and
                variant, the mean held-out loss, style distance, personalization
                gap, prompt adherence, diversity and copy rate, next to the ε each
                client spent.

    fedlora-benchmark --bench-dir ../fedlora-bench --model-root ../ACE-Step-1.5 \\
        --sigmas 1,2,4,8 --rounds 10

The personalization gap needs every client's held-out songs in one place, so it
exists only in simulation. In a real deployment each client runs ``fedlora-eval``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from fedlora_music.backends import backend_class
from fedlora_music.cli import add_model_args, model_spec
from fedlora_music.config import ModelSpec
from fedlora_music.evaluate import VARIANTS, load_app_config
from fedlora_music.metrics import kernel_distance, personalization
from fedlora_music.privacy import epsilon_after
from fedlora_music.store import ClientStore

logger = logging.getLogger(__name__)

STAGES = ("prepare", "train", "eval", "report")
_SUMMARY_KEYS = ("heldout_loss", "kad_to_reference", "prompt_adherence", "diversity")


def sigma_label(sigma: float) -> str:
    return f"sigma-{sigma:g}"


def client_names(bench_dir: Path) -> list[str]:
    summary = json.loads((bench_dir / "partition.json").read_text(encoding="utf-8"))
    return [c["name"] for c in summary["clients"]]


def to_toml(values: Mapping[str, str | int | float | bool]) -> str:
    """Flat key/value TOML. JSON string escaping is valid TOML basic-string escaping."""
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = repr(value)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def run_config(
    *,
    model: ModelSpec,
    clients_root: Path,
    server_output_dir: Path,
    num_clients: int,
    rounds: int,
    sigma: float,
    delta: float,
) -> dict[str, str | int | float | bool]:
    """Run-config overrides for one sweep point. Paths are absolute: the simulation
    runs from its own working directory."""
    cfg: dict[str, str | int | float | bool] = {
        "model-backend": model.backend,
        "model-variant": model.variant,
        "clients-root": str(clients_root),
        "server-output-dir": str(server_output_dir),
        "num-server-rounds": rounds,
        "fraction-train": 1.0,
        "min-train-nodes": num_clients,
        "min-available-nodes": num_clients,
        "dp-noise-multiplier": sigma,
        # The sweep reports ε; it must not stop early because of the default budget.
        "dp-epsilon-budget": math.ceil(epsilon_after(rounds, sigma, delta)) + 1.0,
    }
    if model.root is not None:
        cfg["model-root"] = str(model.root)
    return cfg


def federation_config(num_clients: int, num_cpus: float, num_gpus: float) -> str:
    return (
        f"num-supernodes={num_clients} "
        f"client-resources-num-cpus={num_cpus:g} client-resources-num-gpus={num_gpus:g}"
    )


def _run(cmd: Sequence[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(list(cmd), check=True)


def _has_tensors(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.pt"))


def stage_prepare(args: argparse.Namespace, clients: Sequence[str]) -> None:
    for name in clients:
        client = ClientStore(args.bench_dir / "prepared" / name)
        data, store = args.bench_dir / name, client.model(args.model_key)
        for split, out in (("train", store.tensor_dir), ("eval", store.eval_tensor_dir)):
            if _has_tensors(out):
                logger.info("%s/%s already prepared", name, split)
                continue
            cmd = [
                sys.executable,
                "-m",
                "fedlora_music.prepare",
                "--dataset-json",
                str(data / f"{split}.json"),
                "--client-dir",
                str(client.root),
                *args.model_flags,
                "--split",
                split,
            ]
            _run(cmd)


def _link_client(prepared: ClientStore, fresh: ClientStore, key: str) -> None:
    """Fresh state (ledger, personal adapter) per run; tensors are shared read-only."""
    src, dst = prepared.model(key), fresh.model(key)
    dst.eval_dir.mkdir(parents=True, exist_ok=True)
    for a, b in ((src.tensor_dir, dst.tensor_dir), (src.eval_tensor_dir, dst.eval_tensor_dir)):
        if not b.exists():
            b.symlink_to(a.resolve(), target_is_directory=True)
    if prepared.sources_path.is_file():
        shutil.copy2(prepared.sources_path, fresh.sources_path)


def stage_train(args: argparse.Namespace, clients: Sequence[str], delta: float) -> None:
    for sigma in args.sigmas:
        run_dir = args.bench_dir / "runs" / sigma_label(sigma)
        server_out = run_dir / "server_out"
        if (server_out / "global_adapter.safetensors").is_file():
            logger.info("%s already trained", run_dir.name)
            continue
        for name in clients:
            _link_client(
                ClientStore(args.bench_dir / "prepared" / name),
                ClientStore(run_dir / "clients" / name),
                args.model_key,
            )
        cfg_path = run_dir / "run_config.toml"
        cfg_path.write_text(
            to_toml(
                run_config(
                    model=args.spec,
                    clients_root=run_dir / "clients",
                    server_output_dir=server_out,
                    num_clients=len(clients),
                    rounds=args.rounds,
                    sigma=sigma,
                    delta=delta,
                )
            ),
            encoding="utf-8",
        )
        _run(
            [
                "flwr",
                "run",
                str(args.app_dir),
                args.connection,
                "--run-config",
                str(cfg_path),
                "--federation-config",
                federation_config(len(clients), args.num_cpus, args.num_gpus),
                "--stream",
            ]
        )
        if not (server_out / "global_adapter.safetensors").is_file():
            raise RuntimeError(f"{run_dir.name}: no global adapter after `flwr run`; check logs")


def stage_eval(args: argparse.Namespace, clients: Sequence[str]) -> None:
    for k, sigma in enumerate(args.sigmas):
        run_dir = args.bench_dir / "runs" / sigma_label(sigma)
        # The base model does not depend on sigma: score it once, in the first run.
        variants = [v for v in args.variants if v != "base" or k == 0]
        for name in clients:
            out_dir = run_dir / "eval" / name
            if (out_dir / "report.json").is_file():
                logger.info("%s/%s already evaluated", run_dir.name, name)
                continue
            data = args.bench_dir / name
            _run(
                [
                    sys.executable,
                    "-m",
                    "fedlora_music.evaluate",
                    "--client-dir",
                    str(run_dir / "clients" / name),
                    *args.model_flags,
                    "--app-dir",
                    str(args.app_dir),
                    "--reference-audio",
                    str(data / "eval.json"),
                    "--training-audio",
                    str(data / "train.json"),
                    "--variants",
                    ",".join(variants),
                    "--samples-per-prompt",
                    str(args.samples_per_prompt),
                    "--duration",
                    str(args.duration),
                    "--out-dir",
                    str(out_dir),
                ]
            )


def summarize(
    reports: Mapping[str, Mapping[str, Any]],
    distances: Sequence[Sequence[float]] | None,
    variant: str,
) -> dict[str, float | None]:
    """Mean of each metric over clients for one variant, plus the personalization gap."""
    row: dict[str, float | None] = {}
    for key in _SUMMARY_KEYS:
        vals = [
            r["variants"][variant][key]
            for r in reports.values()
            if r["variants"].get(variant, {}).get(key) is not None
        ]
        row[key] = fmean(vals) if vals else None
    copy_rates = [
        r["variants"][variant]["copy"]["rate_above_threshold"]
        for r in reports.values()
        if "copy" in r["variants"].get(variant, {})
    ]
    row["copy_rate"] = fmean(copy_rates) if copy_rates else None
    if distances is not None:
        p = personalization(distances)
        row["personalization_gap"], row["personalization_top1"] = p.mean_gap, p.top1
    else:
        row["personalization_gap"] = row["personalization_top1"] = None
    return row


def _load_eval(run_dir: Path, clients: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    reports, embeddings = {}, {}
    for name in clients:
        out = run_dir / "eval" / name
        if (out / "report.json").is_file():
            reports[name] = json.loads((out / "report.json").read_text(encoding="utf-8"))
        if (out / "embeddings.safetensors").is_file():
            embeddings[name] = load_file(str(out / "embeddings.safetensors"))
    return reports, embeddings


def _distance_matrix(
    embeddings: Mapping[str, Mapping[str, Any]], clients: Sequence[str], variant: str
) -> list[list[float]] | None:
    """``D[i][j]`` = KAD between client i's generations and client j's held-out songs."""
    key = f"generated.{variant}"
    if not all(key in embeddings.get(c, {}) and "reference" in embeddings[c] for c in clients):
        return None
    return [
        [kernel_distance(embeddings[i][key], embeddings[j]["reference"]) for j in clients]
        for i in clients
    ]


def stage_report(args: argparse.Namespace, clients: Sequence[str], delta: float) -> Path:
    rows: list[dict[str, Any]] = []
    first = args.bench_dir / "runs" / sigma_label(args.sigmas[0])
    base_reports, base_emb = _load_eval(first, clients)
    if base_reports and any("base" in r["variants"] for r in base_reports.values()):
        rows.append(
            {"sigma": None, "epsilon": 0.0, "variant": "base"}
            | summarize(base_reports, _distance_matrix(base_emb, clients, "base"), "base")
        )
    for sigma in args.sigmas:
        reports, emb = _load_eval(args.bench_dir / "runs" / sigma_label(sigma), clients)
        if not reports:
            continue
        eps = epsilon_after(args.rounds, sigma, delta)
        for variant in (v for v in args.variants if v != "base"):
            rows.append(
                {"sigma": sigma, "epsilon": eps, "variant": variant}
                | summarize(reports, _distance_matrix(emb, clients, variant), variant)
            )
    if not rows:
        raise RuntimeError("no evaluation reports found; run the eval stage first")

    csv_path = args.bench_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = render_markdown(rows, len(clients), args.rounds, delta)
    (args.bench_dir / "results.md").write_text(md, encoding="utf-8")
    print(md)
    return csv_path


def render_markdown(
    rows: Sequence[Mapping[str, Any]], clients: int, rounds: int, delta: float
) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    cols = (
        "sigma",
        "epsilon",
        "variant",
        *_SUMMARY_KEYS,
        "personalization_gap",
        "personalization_top1",
        "copy_rate",
    )
    head = (
        f"{clients} clients, {rounds} rounds, δ={delta:g}. Lower is better for loss and KAD; "
        "higher is better for gap, top-1 and prompt adherence.\n\n"
    )
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    table += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return head + "\n".join(table) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bench-dir", type=Path, required=True, help="fedlora-fma-partition output")
    add_model_args(ap)
    ap.add_argument("--app-dir", type=Path, default=Path("."), help="Folder with pyproject.toml")
    ap.add_argument("--connection", default="fedlora-sim", help="SuperLink connection name")
    ap.add_argument("--sigmas", default="1,2,4,8", help="Noise multipliers to sweep")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--samples-per-prompt", type=int, default=1)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--num-cpus", type=float, default=4)
    ap.add_argument("--num-gpus", type=float, default=1.0)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.bench_dir = args.bench_dir.expanduser().resolve()
    args.app_dir = args.app_dir.expanduser().resolve()
    args.sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if bad := (set(stages) - set(STAGES)) | (set(args.variants) - VARIANTS.keys()):
        ap.error(f"unknown stage/variant: {sorted(bad)}")
    if args.bench_dir.is_relative_to(args.app_dir):
        ap.error("--bench-dir must be outside --app-dir, or `flwr run` would bundle the audio")

    clients = client_names(args.bench_dir)
    cfg = load_app_config(args.app_dir)
    delta = cfg.privacy.delta
    args.spec = model_spec(args, run=cfg.model)
    args.model_key = backend_class(args.spec.backend)(args.spec).key
    args.model_flags = [
        "--model-backend", args.spec.backend,
        "--model-variant", args.spec.variant,
        *(["--model-root", str(args.spec.root)] if args.spec.root else []),
    ]  # fmt: skip
    if "prepare" in stages:
        stage_prepare(args, clients)
    if "train" in stages:
        stage_train(args, clients, delta)
    if "eval" in stages:
        stage_eval(args, clients)
    if "report" in stages:
        stage_report(args, clients, delta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
