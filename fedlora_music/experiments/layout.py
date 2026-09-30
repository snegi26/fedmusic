"""Write assigned clients in the layout ``fedlora-benchmark`` reads.

Per client: ``<client>/audio/{train,eval}/<key>.<ext>`` and ACE-Step dataset JSONs
``<client>/{train,eval}.json``; plus ``partition.json`` describing the design. Clean
tracks are symlinked; tracks for a client with a vintage profile are degraded and
written as WAV. The medium applies to everything a client holds, including tracks
mixed in from other clients.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fedlora_music.experiments.design import ClientData, Experiment
from fedlora_music.experiments.sources import Track, read_audio, write_wav
from fedlora_music.experiments.vintage import degrade


def _seed(base: int, key: str) -> int:
    return base ^ int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def _materialize(track: Track, dst_dir: Path, vintage: str, seed: int) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if vintage == "clean":
        name = f"{track.key}{track.path.suffix.lower()}"
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(track.path.resolve())
        return name
    name = f"{track.key}.wav"
    audio, sr = read_audio(track.path)
    write_wav(dst_dir / name, degrade(audio, sr, vintage, seed=_seed(seed, track.key)), sr)
    return name


def _sample(track: Track, split: str, filename: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "audio_path": f"audio/{split}/{filename}",
        "caption": track.caption or track.genre,
        "lyrics": "[Instrumental]",
        "genre": track.genre,
        "bpm": None,
        "keyscale": "",
        "timesignature": "",
        "duration": 0,
    }


def _describe(t: Track) -> dict[str, Any]:
    return {
        "key": t.key,
        "genre": t.genre,
        "year": t.year,
        "artist": t.artist,
        "source": str(t.path),
    }


def write_clients(
    clients: Sequence[ClientData], exp: Experiment, out_dir: Path, variant: str = "base"
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, client in enumerate(clients):
        cdir = out_dir / f"client-{i}"
        for split, tracks in (("train", client.train), ("eval", client.eval)):
            samples = [
                _sample(
                    t, split, _materialize(t, cdir / "audio" / split, client.spec.vintage, exp.seed)
                )
                for t in tracks
            ]
            (cdir / f"{split}.json").write_text(
                json.dumps(
                    {
                        "metadata": {"tag_position": "prepend", "genre_ratio": 0, "custom_tag": ""},
                        "samples": samples,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        summary.append(
            {
                "name": f"client-{i}",
                "label": client.name,
                "genres": list(client.spec.genres),
                "years": list(client.spec.years) if client.spec.years else None,
                "vintage": client.spec.vintage,
                "train": [_describe(t) for t in client.train],
                "eval": [_describe(t) for t in client.eval],
            }
        )
    path = out_dir / "partition.json"
    path.write_text(
        json.dumps(
            {
                "experiment": exp.name,
                "variant": variant,
                "mix": exp.mix,
                "tracks_per_client": exp.tracks_per_client,
                "seed": exp.seed,
                "clients": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
