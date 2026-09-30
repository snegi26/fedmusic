"""Experiment definitions (TOML) and how tracks are assigned to simulated clients.

An experiment names a data source, a list of clients (each a filter on genre and
year, plus an optional simulated recording medium), and optionally a sweep over
settings. Example::

    name = "genre-mixing"
    tracks_per_client = 30
    eval_fraction = 0.2
    seed = 0
    mix = 0.0                      # 0 = each client only its own style, 1 = IID

    [source]
    kind = "fma"
    metadata_dir = "fma_metadata"
    audio_dir = "fma_medium"
    subset = "medium"

    [[clients]]
    name = "historic"
    genres = ["Old-Time / Historic"]

    [[clients]]
    name = "jazz-on-shellac"
    genres = ["Jazz"]
    vintage = "shellac-1920s"      # see fedlora_music.experiments.vintage

    [sweep]
    mix = [0.0, 0.5, 1.0]          # one full benchmark per value

Held-out tracks always come from the client's own filter, so style distances are
measured against the client's style even when training data is mixed.
"""

from __future__ import annotations

import itertools
import random
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fedlora_music.experiments.sources import Track
from fedlora_music.experiments.vintage import PROFILES

SWEEPABLE = ("mix", "tracks_per_client", "eval_fraction")
_TOP_KEYS = {"name", "description", "source", "clients", "sweep", "seed", *SWEEPABLE}
_CLIENT_KEYS = {"name", "genres", "years", "vintage", "artists"}


@dataclass(frozen=True)
class ClientSpec:
    name: str
    genres: tuple[str, ...] = ()  # any of these (case-insensitive); empty = any genre
    years: tuple[int, int] | None = None  # inclusive; tracks without a year never match
    artists: tuple[str, ...] = ()  # substring match on the artist field; empty = any
    vintage: str = "clean"

    def matches(self, t: Track) -> bool:
        if self.genres and not ({g.lower() for g in self.genres} & t.all_genres()):
            return False
        if self.years is not None and (
            t.year is None or not self.years[0] <= t.year <= self.years[1]
        ):
            return False
        return not self.artists or any(a.lower() in t.artist.lower() for a in self.artists)


@dataclass(frozen=True)
class Experiment:
    name: str
    source: Mapping[str, Any]
    clients: tuple[ClientSpec, ...]
    tracks_per_client: int = 30
    eval_fraction: float = 0.2
    mix: float = 0.0
    seed: int = 0
    description: str = ""
    sweep: Mapping[str, Sequence[Any]] | None = None

    @classmethod
    def load(cls, path: Path) -> Experiment:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, default_name=path.stem)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], default_name: str = "experiment") -> Experiment:
        if unknown := set(raw) - _TOP_KEYS:
            raise ValueError(f"unknown experiment keys: {sorted(unknown)}")
        clients = []
        for c in raw.get("clients", []):
            if unknown := set(c) - _CLIENT_KEYS:
                raise ValueError(f"unknown keys in client {c.get('name')!r}: {sorted(unknown)}")
            vintage = c.get("vintage", "clean")
            if vintage not in PROFILES:
                raise ValueError(f"unknown vintage {vintage!r}; choose from {sorted(PROFILES)}")
            years = tuple(c["years"]) if "years" in c else None
            if years is not None and (len(years) != 2 or years[0] > years[1]):
                raise ValueError(f"client {c['name']!r}: years must be [first, last]")
            clients.append(
                ClientSpec(
                    name=str(c["name"]),
                    genres=tuple(c.get("genres", ())),
                    years=years,
                    artists=tuple(c.get("artists", ())),
                    vintage=vintage,
                )
            )
        if len(clients) < 2:
            raise ValueError("an experiment needs at least 2 clients")
        if len({c.name for c in clients}) != len(clients):
            raise ValueError("client names must be unique")
        sweep = raw.get("sweep")
        if sweep and (unknown := set(sweep) - set(SWEEPABLE)):
            raise ValueError(f"can only sweep {SWEEPABLE}, not {sorted(unknown)}")
        exp = cls(
            name=str(raw.get("name", default_name)),
            description=str(raw.get("description", "")),
            source=dict(raw.get("source", {})),
            clients=tuple(clients),
            tracks_per_client=int(raw.get("tracks_per_client", 30)),
            eval_fraction=float(raw.get("eval_fraction", 0.2)),
            mix=float(raw.get("mix", 0.0)),
            seed=int(raw.get("seed", 0)),
            sweep=dict(sweep) if sweep else None,
        )
        for variant in exp.variants().values():
            variant._check()
        return exp

    def _check(self) -> None:
        if not 0.0 <= self.mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        if self.tracks_per_client < 3:
            raise ValueError("tracks_per_client must be >= 3 (2 held out + 1 to train on)")

    def variants(self) -> dict[str, Experiment]:
        """One experiment per point of the sweep, keyed by a folder-safe label."""
        if not self.sweep:
            return {"base": self}
        keys = sorted(self.sweep)
        out = {}
        for values in itertools.product(*(self.sweep[k] for k in keys)):
            label = "_".join(
                f"{k.replace('_', '-')}-{v:g}" for k, v in zip(keys, values, strict=True)
            )
            out[label] = replace(self, sweep=None, **dict(zip(keys, values, strict=True)))
        return out


@dataclass(frozen=True)
class ClientData:
    spec: ClientSpec
    train: tuple[Track, ...]
    eval: tuple[Track, ...]

    @property
    def name(self) -> str:
        return self.spec.name


def assign(tracks: Sequence[Track], exp: Experiment) -> list[ClientData]:
    """Disjoint tracks per client, then `mix` of each client's training tracks swapped
    for tracks from the other clients (evenly, seeded). Held-out tracks stay the
    client's own."""
    rng = random.Random(exp.seed)
    n = exp.tracks_per_client
    n_eval = max(2, round(n * exp.eval_fraction))
    candidates = {c.name: [t for t in tracks if c.matches(t)] for c in exp.clients}
    # Scarcest clients pick first, so overlapping filters fail less often.
    order = sorted(exp.clients, key=lambda c: len(candidates[c.name]))
    used: set[str] = set()
    picked: dict[str, list[Track]] = {}
    for c in order:
        free = sorted((t for t in candidates[c.name] if t.key not in used), key=lambda t: t.key)
        if len(free) < n:
            raise ValueError(
                f"client {c.name!r} matches {len(free)} unused tracks, needs {n}; "
                "lower tracks_per_client or widen its filter"
            )
        picked[c.name] = rng.sample(free, n)
        used.update(t.key for t in picked[c.name])

    held = {c.name: picked[c.name][:n_eval] for c in exp.clients}
    own = {c.name: picked[c.name][n_eval:] for c in exp.clients}
    n_train = n - n_eval
    keep = round((1.0 - exp.mix) * n_train)
    pool = [t for c in exp.clients for t in own[c.name][keep:]]
    rng.shuffle(pool)
    share = n_train - keep
    return [
        ClientData(
            spec=c,
            train=tuple(own[c.name][:keep] + pool[i * share : (i + 1) * share]),
            eval=tuple(held[c.name]),
        )
        for i, c in enumerate(exp.clients)
    ]
