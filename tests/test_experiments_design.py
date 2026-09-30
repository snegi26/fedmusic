"""Experiment definitions, client assignment, data layout and the era report."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from fedlora_music.embeddings import list_audio
from fedlora_music.experiments import design, layout, run, sources

ROOT = Path(__file__).resolve().parents[1]


def _exp(**overrides: object) -> design.Experiment:
    raw = {
        "name": "t",
        "source": {"kind": "folders", "root": "synth"},
        "tracks_per_client": 6,
        "eval_fraction": 0.34,
        "clients": [
            {"name": "a", "genres": ["drone"]},
            {"name": "b", "genres": ["pulse"], "vintage": "shellac-1920s"},
            {"name": "c", "genres": ["percussion"]},
        ],
    }
    return design.Experiment.from_dict(raw | overrides)


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return sources.synthesize(tmp_path_factory.mktemp("synth"), genres=3, per_genre=8, seconds=0.5)


def test_repo_experiments_are_valid() -> None:
    files = sorted((ROOT / "experiments").glob("*.toml"))
    assert len(files) >= 5
    for f in files:
        exp = design.Experiment.load(f)
        assert exp.variants()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mixx": 1}, "unknown experiment keys"),
        ({"clients": [{"name": "a"}]}, "at least 2 clients"),
        ({"clients": [{"name": "a", "vintage": "wax"}, {"name": "b"}]}, "unknown vintage"),
        ({"clients": [{"name": "a", "years": [2000, 1990]}, {"name": "b"}]}, "years"),
        ({"sweep": {"seed": [1, 2]}}, "can only sweep"),
        ({"sweep": {"mix": [0.0, 2.0]}}, "mix must be"),
    ],
)
def test_invalid_definitions_are_rejected(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _exp(**override)


def test_sweep_expands_to_labelled_variants() -> None:
    variants = _exp(sweep={"mix": [0.0, 0.5], "tracks_per_client": [4, 6]}).variants()
    assert set(variants) == {
        "mix-0_tracks-per-client-4",
        "mix-0_tracks-per-client-6",
        "mix-0.5_tracks-per-client-4",
        "mix-0.5_tracks-per-client-6",
    }
    assert variants["mix-0.5_tracks-per-client-4"].mix == 0.5


def test_synthetic_source_has_genres_and_years(synth: Path) -> None:
    tracks = sources.from_folders(synth)
    assert Counter(t.genre for t in tracks) == {"drone": 8, "pulse": 8, "percussion": 8}
    years = sorted(t.year for t in tracks if t.genre == "drone")
    assert years[0] == 1920 and years[-1] == 2020
    assert all(t.caption.endswith("0s") for t in tracks)
    audio, sr = sources.read_audio(tracks[0].path)
    assert sr == 22_050 and audio.shape[1] == 1


def test_assignment_is_disjoint_and_mix_controls_genre_skew(synth: Path) -> None:
    tracks = sources.from_folders(synth)
    pure = design.assign(tracks, _exp())
    keys = [t.key for c in pure for t in c.train + c.eval]
    assert len(keys) == len(set(keys)) == 18  # disjoint across clients
    for c in pure:
        assert len(c.eval) == 2 and len(c.train) == 4
        assert {t.genre for t in c.train + c.eval} == set(c.spec.genres)
    mixed = design.assign(tracks, _exp(mix=1.0))
    assert any(len({t.genre for t in c.train}) > 1 for c in mixed)
    for c in mixed:  # held-out songs stay the client's own style
        assert {t.genre for t in c.eval} == set(c.spec.genres)
    assert design.assign(tracks, _exp(mix=1.0)) == mixed  # seeded


def test_filters_and_shortage(synth: Path) -> None:
    tracks = sources.from_folders(synth)
    old = design.ClientSpec("old", years=(1900, 1960))
    assert all(t.year <= 1960 for t in tracks if old.matches(t))
    with pytest.raises(ValueError, match="needs 20"):
        design.assign(tracks, _exp(tracks_per_client=20))


def test_layout_links_clean_tracks_and_degrades_vintage(synth: Path, tmp_path: Path) -> None:
    exp = _exp()
    part = layout.write_clients(design.assign(sources.from_folders(synth), exp), exp, tmp_path)
    summary = json.loads(part.read_text())
    assert [c["name"] for c in summary["clients"]] == ["client-0", "client-1", "client-2"]
    assert [c["vintage"] for c in summary["clients"]][1] == "shellac-1920s"
    clean = list_audio(tmp_path / "client-0" / "train.json")
    old = list_audio(tmp_path / "client-1" / "train.json")
    assert len(clean) == 4 and all((tmp_path / "client-0" / "audio" / "train").iterdir())
    assert all((tmp_path / "client-0" / "audio" / "train" / p.name).is_symlink() for p in clean)
    assert not any((tmp_path / "client-1" / "audio" / "train" / p.name).is_symlink() for p in old)
    assert "shellac" in run.summarize_plan(tmp_path)


def test_era_report_from_generated_audio(synth: Path, tmp_path: Path) -> None:
    """Fake a run: each client 'generates' its own held-out audio, so the gap is > 0."""
    exp = _exp()
    vdir = tmp_path / "t" / "base"
    layout.write_clients(design.assign(sources.from_folders(synth), exp), exp, vdir)
    for i in range(3):
        gen = vdir / "runs" / "sigma-4" / "eval" / f"client-{i}" / "audio" / "fused"
        gen.mkdir(parents=True)
        for p in list_audio(vdir / f"client-{i}" / "eval.json"):
            audio, sr = sources.read_audio(p)
            sources.write_wav(gen / p.name, audio, sr)
    run.report(tmp_path / "t")
    md = (tmp_path / "t" / "report.md").read_text()
    assert "Era signature" in md and "| base | 4 | fused |" in md
    with (tmp_path / "t" / "era_features.csv").open() as fh:
        rows = {r["client"]: r for r in csv.DictReader(fh)}
    assert set(rows) == {"client-0", "client-1", "client-2"}
    # Each client "generated" its own held-out audio, so it is closest to its own era.
    assert all(float(r["era_gap"]) > 0 for r in rows.values())
    assert rows["client-1"]["medium"] == "shellac-1920s"
