"""FMA partitioning and benchmark bookkeeping, on a synthetic FMA-shaped metadata set."""

from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import pytest

from fedlora_music.benchmark.fma import (
    audio_path,
    partition,
    read_tracks,
    write_partition,
)
from fedlora_music.benchmark.runner import (
    federation_config,
    render_markdown,
    run_config,
    summarize,
    to_toml,
)
from fedlora_music.embeddings import list_audio
from fedlora_music.privacy import epsilon_after

GENRES = {1: "Rock", 2: "Punk", 3: "Electronic", 4: "Folk"}


def _fake_fma(root: Path, per_genre: int = 6) -> tuple[Path, Path]:
    """``tracks.csv`` with FMA's three header rows, ``genres.csv``, and empty mp3s."""
    meta, audio = root / "fma_metadata", root / "fma_small"
    meta.mkdir()
    with (meta / "genres.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["genre_id", "#tracks", "parent", "title", "top_level"])
        w.writerows([gid, 0, 0, title, gid] for gid, title in GENRES.items())
    top = ["", "artist", "artist", "set", "set", "track", "track", "track"]
    sub = ["", "id", "name", "split", "subset", "genre_top", "genres", "title"]
    rows, tid = [], 1
    for gid, genre in [(1, "Rock"), (3, "Electronic"), (4, "Folk")]:
        for k in range(per_genre):
            genres = f"[{gid}, 2]" if genre == "Rock" else f"[{gid}]"
            subset = "medium" if k == 0 else "small"  # one track per genre outside small
            rows.append([tid, 100 + gid, f"artist{gid}", "training", subset, genre, genres, "t"])
            path = audio_path(audio, tid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            tid += 1
    with (meta / "tracks.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerows([top, sub, ["track_id", *[""] * (len(top) - 1)], *rows])
    return meta, audio


def test_read_tracks_filters_subset_and_names_genres(tmp_path: Path) -> None:
    meta, _ = _fake_fma(tmp_path)
    small = read_tracks(meta, "small")
    assert len(small) == 15 and {t.subset for t in small} == {"small"}
    assert len(read_tracks(meta, "medium")) == 18
    rock = next(t for t in small if t.genre_top == "Rock")
    assert rock.caption == "Rock, Punk"


def test_partition_is_deterministic_disjoint_and_balanced(tmp_path: Path) -> None:
    tracks = read_tracks(_fake_fma(tmp_path)[0], "small")
    a = partition(tracks, 2, "genre", tracks_per_client=5, eval_fraction=0.2, seed=3)
    assert a == partition(tracks, 2, "genre", tracks_per_client=5, eval_fraction=0.2, seed=3)
    assert len({s.group for s in a}) == 2
    for s in a:
        assert (len(s.train), len(s.eval)) == (3, 2)  # at least 2 held out
        assert not {t.track_id for t in s.train} & {t.track_id for t in s.eval}
        assert {t.genre_top for t in s.train + s.eval} == {s.group}
    with pytest.raises(ValueError, match="groups"):
        partition(tracks, 4, "genre", tracks_per_client=5, eval_fraction=0.2, seed=0)


def test_write_partition_emits_ace_step_dataset_json(tmp_path: Path) -> None:
    meta, audio = _fake_fma(tmp_path)
    splits = partition(read_tracks(meta, "small"), 2, "artist", 5, 0.2, seed=0)
    write_partition(splits, audio, tmp_path / "bench")
    client = tmp_path / "bench" / "client-0"
    data = json.loads((client / "eval.json").read_text())
    assert len(data["samples"]) == 2
    assert all(s["caption"] and s["lyrics"] == "[Instrumental]" for s in data["samples"])
    # Relative audio paths resolve through the symlinks, as ACE-Step and fedlora-eval read them.
    assert len(list_audio(client / "eval.json")) == 2
    assert len(list_audio(client / "audio" / "train")) == 3
    summary = json.loads((tmp_path / "bench" / "partition.json").read_text())
    assert [c["name"] for c in summary["clients"]] == ["client-0", "client-1"]


def test_run_config_round_trips_through_toml(tmp_path: Path) -> None:
    cfg = run_config(
        ace_project_root=tmp_path / "ace",
        clients_root=tmp_path / 'we"ird',
        server_output_dir=tmp_path / "out",
        num_clients=3,
        rounds=10,
        sigma=4.0,
        delta=1e-5,
        model_variant=None,
    )
    parsed = tomllib.loads(to_toml(cfg))
    assert parsed == cfg
    assert parsed["dp-epsilon-budget"] > epsilon_after(10, 4.0, 1e-5)
    assert "model-variant" not in parsed
    assert federation_config(3, 4, 1.0) == (
        "num-supernodes=3 client-resources-num-cpus=4 client-resources-num-gpus=1"
    )


def test_summarize_and_render() -> None:
    def report(loss: float, rate: float) -> dict:
        return {
            "variants": {
                "fused": {
                    "heldout_loss": loss,
                    "kad_to_reference": 0.1,
                    "prompt_adherence": 0.3,
                    "diversity": 0.2,
                    "copy": {"rate_above_threshold": rate},
                }
            }
        }

    row = summarize(
        {"client-0": report(1.0, 0.0), "client-1": report(3.0, 0.5)},
        [[0.1, 0.5], [0.6, 0.2]],
        "fused",
    )
    assert row["heldout_loss"] == 2.0 and row["copy_rate"] == 0.25
    assert row["personalization_top1"] == 1.0
    assert row["personalization_gap"] == pytest.approx(0.4)
    md = render_markdown([{"sigma": 4.0, "epsilon": 8.1, "variant": "fused", **row}], 2, 10, 1e-5)
    assert "| 4 | 8.1 | fused | 2 |" in md
    assert summarize({"client-0": report(1.0, 0.0)}, None, "global")["heldout_loss"] is None


def test_report_stage_aggregates_eval_outputs(tmp_path: Path) -> None:
    """Fake two runs of `fedlora-eval` output, then build results.csv / results.md."""
    import torch
    from safetensors.torch import save_file

    from fedlora_music.benchmark import runner

    bench, app = tmp_path / "bench", tmp_path / "app"
    app.mkdir()
    (app / "pyproject.toml").write_text(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    bench.mkdir()
    clients = ["client-0", "client-1"]
    (bench / "partition.json").write_text(json.dumps({"clients": [{"name": c} for c in clients]}))
    g = torch.Generator().manual_seed(0)
    styles = {c: torch.randn(1, 8, generator=g) * 5 for c in clients}
    for k, sigma in enumerate((1.0, 4.0)):
        variants = ["base", "fused"] if k == 0 else ["fused"]
        for c in clients:
            out = bench / "runs" / runner.sigma_label(sigma) / "eval" / c
            out.mkdir(parents=True)
            report = {"variants": {v: {"heldout_loss": 1.0 + k} for v in variants}}
            (out / "report.json").write_text(json.dumps(report))
            near = styles[c] + 0.1 * torch.randn(6, 8, generator=g)
            emb = {"reference": styles[c] + 0.1 * torch.randn(6, 8, generator=g)}
            emb |= {f"generated.{v}": near.clone() for v in variants}
            save_file(emb, str(out / "embeddings.safetensors"))

    assert runner.main([
        "--bench-dir", str(bench), "--ace-project-root", str(tmp_path / "ace"),
        "--app-dir", str(app), "--sigmas", "1,4", "--variants", "base,fused",
        "--stages", "report",
    ]) == 0  # fmt: skip
    rows = list(csv.DictReader((bench / "results.csv").open()))
    assert [(r["sigma"], r["variant"]) for r in rows] == [
        ("", "base"),
        ("1.0", "fused"),
        ("4.0", "fused"),
    ]
    assert all(float(r["personalization_top1"]) == 1.0 for r in rows)
    assert float(rows[1]["epsilon"]) > float(rows[2]["epsilon"]) > 0
    assert (bench / "results.md").is_file()


def test_bench_dir_inside_app_dir_is_rejected(tmp_path: Path) -> None:
    from fedlora_music.benchmark import runner

    with pytest.raises(SystemExit):
        runner.main([
            "--bench-dir", str(tmp_path / "app" / "bench"), "--ace-project-root", str(tmp_path),
            "--app-dir", str(tmp_path / "app"),
        ])  # fmt: skip
