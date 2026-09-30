"""Split the Free Music Archive (FMA) into simulated clients with distinct styles.

FMA (https://github.com/mdeff/fma) is Creative Commons-licensed and ships genre and
artist metadata, so each simulated client can get its own style: one genre or one
artist per client. Download ``fma_metadata.zip`` and an audio subset (``fma_small``
is 8,000 30-second clips over 8 genres), unzip both, then:

    fedlora-fma-partition --metadata-dir fma_metadata --audio-dir fma_small \\
        --out-dir ../fedlora-bench --num-clients 4 --group-by genre

For each client this writes ``client-<i>/{train,eval}.json`` (ACE-Step dataset JSON
with genre captions) and ``client-<i>/audio/{train,eval}/`` (symlinks, or copies with
``--copy``). Keep ``--out-dir`` outside the app folder so ``flwr run`` does not
bundle the audio.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FmaTrack:
    track_id: int
    artist_id: int
    artist: str
    genre_top: str
    genres: tuple[str, ...]
    subset: str
    # Recording year, else album release year. For digitized archive material this is
    # often the digitization date, not the original recording's: prefer the genre
    # ("Old-Time / Historic") to find old recordings in FMA.
    year: int | None = None

    @property
    def caption(self) -> str:
        return ", ".join(dict.fromkeys(g for g in (self.genre_top, *self.genres) if g))


@dataclass(frozen=True)
class ClientSplit:
    name: str
    group: str
    train: tuple[FmaTrack, ...]
    eval: tuple[FmaTrack, ...]


# fma_small is inside fma_medium, which is inside fma_large; ``set.subset`` holds the
# smallest subset a track belongs to.
_SUBSETS = ("small", "medium", "large")


def audio_path(audio_dir: Path, track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return audio_dir / tid[:3] / f"{tid}.mp3"


def read_genres(metadata_dir: Path) -> dict[int, str]:
    with (metadata_dir / "genres.csv").open(newline="", encoding="utf-8") as fh:
        return {int(row["genre_id"]): row["title"] for row in csv.DictReader(fh)}


def _year(row: list[str], col: dict[str, int], *keys: str) -> int | None:
    for key in keys:
        value = row[col[key]] if key in col else ""
        if len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


def read_tracks(metadata_dir: Path, subset: str) -> list[FmaTrack]:
    """Parse ``tracks.csv`` (three header rows) for tracks in ``subset``."""
    allowed = set(_SUBSETS[: _SUBSETS.index(subset) + 1])
    genre_names = read_genres(metadata_dir)
    tracks = []
    with (metadata_dir / "tracks.csv").open(newline="", encoding="utf-8") as fh:
        rows = csv.reader(fh)
        top, sub, _ = next(rows), next(rows), next(rows)
        col = {f"{t}.{s}": i for i, (t, s) in enumerate(zip(top, sub, strict=True)) if t}
        for row in rows:
            if row[col["set.subset"]] not in allowed:
                continue
            genre_ids = ast.literal_eval(row[col["track.genres"]] or "[]")
            tracks.append(
                FmaTrack(
                    track_id=int(row[0]),
                    artist_id=int(row[col["artist.id"]]),
                    artist=row[col["artist.name"]],
                    genre_top=row[col["track.genre_top"]],
                    genres=tuple(genre_names[g] for g in genre_ids if g in genre_names),
                    subset=row[col["set.subset"]],
                    year=_year(row, col, "track.date_recorded", "album.date_released"),
                )
            )
    return tracks


def partition(
    tracks: Iterable[FmaTrack],
    num_clients: int,
    group_by: str,
    tracks_per_client: int,
    eval_fraction: float,
    seed: int,
) -> list[ClientSplit]:
    """One group (genre or artist) per client, each split into train and held-out eval.

    Only groups with at least ``tracks_per_client`` tracks qualify, so every client
    gets the same amount of data. At least 2 eval tracks are kept so distribution
    distances are defined.
    """
    if group_by not in ("genre", "artist"):
        raise ValueError("group_by must be 'genre' or 'artist'")
    if tracks_per_client < 3:
        raise ValueError("tracks_per_client must be >= 3 (2 held out + 1 to train on)")
    groups: dict[str, list[FmaTrack]] = defaultdict(list)
    for t in tracks:
        key = t.genre_top if group_by == "genre" else f"{t.artist_id}:{t.artist}"
        if key:
            groups[key].append(t)
    eligible = sorted(k for k, v in groups.items() if len(v) >= tracks_per_client)
    if len(eligible) < num_clients:
        raise ValueError(
            f"only {len(eligible)} {group_by} groups have >= {tracks_per_client} tracks; "
            f"lower --tracks-per-client or --num-clients"
        )
    rng = random.Random(seed)
    chosen = rng.sample(eligible, num_clients)
    n_eval = max(2, round(tracks_per_client * eval_fraction))
    splits = []
    for i, key in enumerate(chosen):
        members = sorted(groups[key], key=lambda t: t.track_id)
        picked = rng.sample(members, tracks_per_client)
        splits.append(
            ClientSplit(
                name=f"client-{i}",
                group=key,
                train=tuple(picked[n_eval:]),
                eval=tuple(picked[:n_eval]),
            )
        )
    return splits


def dataset_json(tracks: Sequence[FmaTrack], split: str) -> dict[str, object]:
    """ACE-Step dataset JSON; ``audio_path`` is relative to the JSON file."""
    return {
        "metadata": {"tag_position": "prepend", "genre_ratio": 0, "custom_tag": ""},
        "samples": [
            {
                "filename": f"{t.track_id:06d}.mp3",
                "audio_path": f"audio/{split}/{t.track_id:06d}.mp3",
                "caption": t.caption,
                "lyrics": "[Instrumental]",
                "genre": t.genre_top,
                "bpm": None,
                "keyscale": "",
                "timesignature": "",
                "duration": 0,
            }
            for t in tracks
        ],
    }


def write_partition(
    splits: Sequence[ClientSplit], audio_dir: Path, out_dir: Path, copy: bool = False
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        client_dir = out_dir / split.name
        for name, tracks in (("train", split.train), ("eval", split.eval)):
            dst_dir = client_dir / "audio" / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for t in tracks:
                src, dst = audio_path(audio_dir, t.track_id), dst_dir / f"{t.track_id:06d}.mp3"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if copy:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())
            (client_dir / f"{name}.json").write_text(
                json.dumps(dataset_json(tracks, name), indent=2), encoding="utf-8"
            )
    summary = out_dir / "partition.json"
    summary.write_text(
        json.dumps(
            {
                "clients": [
                    {
                        "name": s.name,
                        "group": s.group,
                        "train": [asdict(t) for t in s.train],
                        "eval": [asdict(t) for t in s.eval],
                    }
                    for s in splits
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--metadata-dir", type=Path, required=True, help="Unzipped fma_metadata")
    ap.add_argument("--audio-dir", type=Path, required=True, help="Unzipped fma_small/medium")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subset", choices=_SUBSETS, default="small")
    ap.add_argument("--num-clients", type=int, default=4)
    ap.add_argument("--group-by", choices=("genre", "artist"), default="genre")
    ap.add_argument("--tracks-per-client", type=int, default=40)
    ap.add_argument("--eval-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="Copy audio instead of symlinking")
    args = ap.parse_args(argv)

    audio_dir = args.audio_dir.expanduser().resolve()
    tracks = [
        t
        for t in read_tracks(args.metadata_dir.expanduser().resolve(), args.subset)
        if audio_path(audio_dir, t.track_id).is_file()
    ]
    splits = partition(
        tracks,
        args.num_clients,
        args.group_by,
        args.tracks_per_client,
        args.eval_fraction,
        args.seed,
    )
    summary = write_partition(splits, audio_dir, args.out_dir.expanduser().resolve(), args.copy)
    for s in splits:
        print(f"{s.name}: {s.group} ({len(s.train)} train, {len(s.eval)} eval)")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
