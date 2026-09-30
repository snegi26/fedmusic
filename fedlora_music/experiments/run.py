"""fedlora-experiment: data experiments with older recordings, genres and recording eras.

    fedlora-experiment synth  --out ../synth-music                 # tone-based test set
    fedlora-experiment plan   experiments/historic-vs-modern.toml --out ../fedlora-exp
    fedlora-experiment run    experiments/historic-vs-modern.toml --out ../fedlora-exp \\
        --model-root ../ACE-Step-1.5 --sigmas 4 --rounds 10       # + any fedlora-benchmark option
    fedlora-experiment report --out ../fedlora-exp/historic-vs-modern

`plan` assigns tracks to clients and writes their data (simulating recording media
where asked) for every sweep variant. `run` plans, then runs the full benchmark
(prepare, federated training per noise level, evaluation) for each variant. `report`
adds era descriptors of the generated audio to the benchmark tables. Keep `--out`
outside this repository so `flwr run` does not bundle the audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from fedlora_music.embeddings import list_audio
from fedlora_music.experiments import design, features, layout, sources

logger = logging.getLogger(__name__)


def plan(exp_path: Path, out: Path, data_root: Path, force: bool = False) -> dict[str, Path]:
    exp = design.Experiment.load(exp_path)
    tracks = sources.load(exp.source, data_root)
    logger.info("%s: %d candidate tracks from %s", exp.name, len(tracks), exp.source.get("kind"))
    dirs = {}
    for label, variant in exp.variants().items():
        vdir = out / exp.name / label
        dirs[label] = vdir
        if (vdir / "partition.json").is_file() and not force:
            logger.info("%s/%s already planned", exp.name, label)
            continue
        if vdir.exists():
            shutil.rmtree(vdir)
        clients = design.assign(tracks, variant)
        layout.write_clients(clients, variant, vdir, label)
        print(summarize_plan(vdir))
    return dirs


def summarize_plan(vdir: Path) -> str:
    part = json.loads((vdir / "partition.json").read_text(encoding="utf-8"))
    lines = [
        f"\n{part['experiment']} / {part['variant']} (mix={part['mix']:g})",
        "| client | label | medium | train genres | eval genres | years |",
        "|---|---|---|---|---|---|",
    ]
    for c in part["clients"]:
        train = Counter(t["genre"] for t in c["train"])
        years = [t["year"] for t in c["train"] + c["eval"] if t["year"] is not None]
        lines.append(
            f"| {c['name']} | {c['label']} | {c['vintage']} | "
            + ", ".join(f"{g} {n}" for g, n in train.most_common())
            + f" | {', '.join(sorted({t['genre'] for t in c['eval']}))} | "
            + (f"{min(years)}-{max(years)}" if years else "-")
            + " |"
        )
    return "\n".join(lines)


# -- era report ----------------------------------------------------------------------


def _describe_files(paths: list[Path]) -> dict[str, float]:
    rows = []
    for p in paths:
        audio, sr = sources.read_audio(p)
        rows.append(features.describe(audio, sr))
    return features.mean_features(rows)


def era_rows(vdir: Path) -> list[dict[str, Any]]:
    """Per run, model variant and client: era distance of generated audio to the
    client's own held-out songs vs to the other clients' (gap > 0 = its own era)."""
    part = json.loads((vdir / "partition.json").read_text(encoding="utf-8"))
    clients = part["clients"]
    refs = {c["name"]: _describe_files(list_audio(vdir / c["name"] / "eval.json")) for c in clients}
    rows = []
    for run_dir in sorted((vdir / "runs").glob("sigma-*")):
        for c in clients:
            audio_root = run_dir / "eval" / c["name"] / "audio"
            for mdir in sorted(p for p in audio_root.glob("*") if p.is_dir()):
                gen = _describe_files(sorted(q for q in mdir.rglob("*") if q.is_file()))
                if not gen:
                    continue
                own = features.era_distance(gen, refs[c["name"]])
                others = [
                    features.era_distance(gen, refs[o["name"]]) for o in clients if o is not c
                ]
                rows.append(
                    {
                        "setting": part["variant"],
                        "sigma": run_dir.name.removeprefix("sigma-"),
                        "model": mdir.name,
                        "client": c["name"],
                        "label": c["label"],
                        "medium": c["vintage"],
                        "era_distance_own": own,
                        "era_distance_others": fmean(others),
                        "era_gap": fmean(others) - own,
                        **{f"gen_{k}": v for k, v in gen.items()},
                        **{f"ref_{k}": v for k, v in refs[c["name"]].items()},
                    }
                )
    return rows


def _cell(value: str | None) -> str:
    if not value:
        return "-"
    try:
        return f"{float(value):.4g}"
    except ValueError:
        return value


def report(exp_dir: Path) -> Path:
    variants = sorted(p for p in exp_dir.iterdir() if (p / "partition.json").is_file())
    if not variants:
        raise FileNotFoundError(f"no planned variants under {exp_dir}")
    bench_rows, era = [], []
    for vdir in variants:
        if (vdir / "results.csv").is_file():
            with (vdir / "results.csv").open(newline="", encoding="utf-8") as fh:
                bench_rows += [
                    {"setting": vdir.name, "model": r.pop("variant"), **r}
                    for r in csv.DictReader(fh)
                ]
        era += era_rows(vdir)

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    md = [f"# {exp_dir.name}\n"]
    if bench_rows:
        write_csv(exp_dir / "results.csv", bench_rows)
        cols = [
            "setting",
            "model",
            "sigma",
            "epsilon",
            "personalization_gap",
            "personalization_top1",
            "kad_to_reference",
            "heldout_loss",
        ]
        cols = [c for c in cols if c in bench_rows[0]]
        md += [
            "## Benchmark (style, CLAP)\n",
            "| " + " | ".join(cols) + " |",
            "|" + "---|" * len(cols),
        ]
        md += ["| " + " | ".join(_cell(r.get(c)) for c in cols) + " |" for r in bench_rows]
    if era:
        write_csv(exp_dir / "era_features.csv", era)
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in era:
            groups.setdefault((r["setting"], r["sigma"], r["model"]), []).append(r)
        md += [
            "\n## Era signature of generated audio\n",
            "Distance of each client's generations to its own held-out songs' sound "
            "(bandwidth, noise, crackle, stereo) vs other clients'. Gap > 0: the model "
            "reproduces the client's own era/medium.\n",
            "| setting | sigma | model | era distance (own) | era gap | clients with gap > 0 |",
            "|---|---|---|---|---|---|",
        ]
        for (setting, sigma, model), rows in sorted(groups.items()):
            wins = sum(r["era_gap"] > 0 for r in rows)
            own = fmean(r["era_distance_own"] for r in rows)
            gap = fmean(r["era_gap"] for r in rows)
            md.append(
                f"| {setting} | {sigma} | {model} | {own:.3f} | {gap:.3f} | {wins}/{len(rows)} |"
            )
    if len(md) == 1:
        md.append("No results yet: run the experiment first.")
    path = exp_dir / "report.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    return path


# -- CLI -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="write a small tone-based dataset (folders source)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--genres", type=int, default=4)
    s.add_argument("--per-genre", type=int, default=12)
    s.add_argument("--seconds", type=float, default=4.0)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("experiment", type=Path)
        p.add_argument("--out", type=Path, required=True, help="outside this repository")
        p.add_argument(
            "--data-root", type=Path, default=Path("."), help="base for relative source paths"
        )
        p.add_argument("--force", action="store_true", help="re-plan existing variants")
    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, required=True, help="the experiment's folder")
    args, rest = ap.parse_known_args(argv)
    if rest and args.cmd != "run":
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "synth":
        print(
            sources.synthesize(
                args.out.expanduser().resolve(), args.genres, args.per_genre, args.seconds
            )
        )
        return 0
    if args.cmd == "report":
        report(args.out.expanduser().resolve())
        return 0

    out = args.out.expanduser().resolve()
    dirs = plan(args.experiment, out, args.data_root.expanduser().resolve(), args.force)
    if args.cmd == "run":
        from fedlora_music.benchmark import runner

        for label, vdir in dirs.items():
            logger.info("==> %s", label)
            runner.main(["--bench-dir", str(vdir), *rest])
        report(out / design.Experiment.load(args.experiment).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
