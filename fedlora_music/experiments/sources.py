"""Where experiment tracks come from: FMA, your own folders, or a synthetic set.

* ``fma``      FMA metadata + audio. ``fma_medium`` and ``fma_large`` include the
               "Old-Time / Historic" genre (digitized old recordings) next to modern
               genres; ``fma_small`` has 8 modern genres only.
* ``folders``  your own collection: ``<root>/<genre>/**/<audio>``. An optional
               ``<root>/metadata.csv`` (columns ``path`` relative to root, and any of
               ``genre``, ``year``, ``artist``, ``caption``) adds years and captions,
               which the era experiments need.
* synthetic    ``synthesize()`` writes a small tone-based set with genres and years,
               to try the pipeline end to end without downloading anything.
"""

from __future__ import annotations

import csv
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fedlora_music.embeddings import AUDIO_EXTENSIONS


@dataclass(frozen=True)
class Track:
    key: str  # unique and filesystem-safe
    path: Path
    genre: str
    genres: tuple[str, ...] = ()
    year: int | None = None
    artist: str = ""
    caption: str = ""

    def all_genres(self) -> set[str]:
        return {g.lower() for g in (self.genre, *self.genres) if g}


def _caption(genre: str, genres: tuple[str, ...], year: int | None) -> str:
    parts = list(dict.fromkeys(g for g in (genre, *genres) if g))
    if year is not None:
        parts.append(f"{year // 10 * 10}s")
    return ", ".join(parts)


def from_fma(metadata_dir: Path, audio_dir: Path, subset: str) -> list[Track]:
    from fedlora_music.benchmark.fma import audio_path, read_tracks

    tracks = []
    for t in read_tracks(metadata_dir, subset):
        path = audio_path(audio_dir, t.track_id)
        if t.genre_top and path.is_file():
            tracks.append(
                Track(
                    key=f"fma{t.track_id:06d}",
                    path=path,
                    genre=t.genre_top,
                    genres=t.genres,
                    year=t.year,
                    artist=f"{t.artist_id}:{t.artist}",
                    # No decade in FMA captions: its years are often digitization dates.
                    caption=t.caption,
                )
            )
    return tracks


def from_folders(root: Path) -> list[Track]:
    meta: dict[str, dict[str, str]] = {}
    if (root / "metadata.csv").is_file():
        with (root / "metadata.csv").open(newline="", encoding="utf-8") as fh:
            meta = {row["path"]: row for row in csv.DictReader(fh)}
    tracks = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            continue  # files directly under root have no genre folder
        row = meta.get(rel.as_posix(), {})
        genre = row.get("genre") or rel.parts[0]
        year = int(row["year"]) if (row.get("year") or "").strip().isdigit() else None
        key = "-".join(rel.with_suffix("").parts).replace(" ", "_")
        tracks.append(
            Track(
                key=key,
                path=path,
                genre=genre,
                year=year,
                artist=row.get("artist", ""),
                caption=row.get("caption") or _caption(genre, (), year),
            )
        )
    return tracks


def load(source: Mapping[str, Any], base: Path) -> list[Track]:
    """Tracks for an experiment's ``[source]`` table; relative paths resolve from ``base``."""

    def path(key: str) -> Path:
        value = Path(str(source[key])).expanduser()
        return value if value.is_absolute() else (base / value).resolve()

    kind = source.get("kind")
    if kind == "fma":
        return from_fma(
            path("metadata_dir"), path("audio_dir"), str(source.get("subset", "medium"))
        )
    if kind == "folders":
        return from_folders(path("root"))
    raise ValueError(f"source kind must be 'fma' or 'folders', got {kind!r}")


# -- audio IO ------------------------------------------------------------------------


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """(frames, channels) float32 and sample rate. soundfile for any format (MP3
    needs it); plain 16-bit WAV works without it."""
    try:
        import soundfile as sf
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                f"reading {path.suffix} needs soundfile: pip install soundfile"
            ) from None
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise RuntimeError(f"{path}: only 16-bit WAV without soundfile") from None
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            data = raw.reshape(-1, w.getnchannels()).astype(np.float32) / 32768
            return data, w.getframerate()
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """16-bit PCM WAV with the standard library (readable everywhere)."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# -- synthetic set -------------------------------------------------------------------

_RECIPES = ("drone", "pulse", "percussion", "arpeggio", "chorale", "bass")


def _render(recipe: str, seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    root = 110.0 * 2 ** (rng.integers(0, 12) / 12)
    if recipe == "drone":
        x = sum(np.sin(2 * np.pi * root * r * t) for r in (1, 1.5, 2, 3)) / 4
    elif recipe == "pulse":
        x = np.sign(np.sin(2 * np.pi * root * 2 * t)) * (np.sin(2 * np.pi * 2 * t) > 0) * 0.5
    elif recipe == "percussion":
        env = np.exp(-((t * 4) % 1) * 25)
        x = rng.standard_normal(t.size) * env * 0.6
    elif recipe == "arpeggio":
        step = (t * 8).astype(int) % 4
        freq = root * 2 * np.array([1, 1.25, 1.5, 2])[step]
        x = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-((t * 8) % 1) * 6)
    elif recipe == "chorale":
        x = sum(np.sin(2 * np.pi * root * 2 * r * t + i) for i, r in enumerate((1, 1.26, 1.5))) / 3
        x *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t)
    else:  # bass
        x = np.tanh(3 * np.sin(2 * np.pi * root / 2 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * t))
    return (0.5 * x / (np.max(np.abs(x)) or 1.0)).astype(np.float32)


def synthesize(
    out_dir: Path,
    genres: int = 4,
    per_genre: int = 12,
    seconds: float = 4.0,
    sr: int = 22_050,
    years: tuple[int, int] = (1920, 2020),
    seed: int = 0,
) -> Path:
    """A folders-style source of tone-based "genres" with spread-out years."""
    if not 1 <= genres <= len(_RECIPES):
        raise ValueError(f"genres must be 1..{len(_RECIPES)}")
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(genres):
        recipe = _RECIPES[g]
        for i in range(per_genre):
            year = int(years[0] + (years[1] - years[0]) * i / max(1, per_genre - 1))
            rel = Path(recipe) / f"{recipe}-{i:03d}.wav"
            write_wav(out_dir / rel, _render(recipe, seconds, sr, rng), sr)
            rows.append(
                {
                    "path": rel.as_posix(),
                    "genre": recipe,
                    "year": year,
                    "artist": f"synth-{recipe}",
                    "caption": _caption(recipe, (), year),
                }
            )
    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out_dir


__all__ = [
    "Track",
    "from_fma",
    "from_folders",
    "load",
    "read_audio",
    "synthesize",
    "write_wav",
]
