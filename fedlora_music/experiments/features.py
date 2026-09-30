"""Model-independent descriptors of a recording's sound: its "era signature".

CLAP embeddings (``fedlora_music.metrics``) capture musical style; these capture the
medium: how wide the frequency band is, how much noise and crackle there is, how
wide the stereo image is. They are cheap, deterministic signal measurements, so they
can check whether a generated clip picked up a client's vintage sound.

| feature             | meaning                                                       |
|---------------------|---------------------------------------------------------------|
| centroid_hz         | spectral centre of mass (brightness)                          |
| rolloff_hz          | frequency below which 85% of the energy lies                  |
| band_low_hz/high_hz | edges of the band within 50 dB of the loudest frequency       |
| hf_ratio_db         | energy above 5 kHz relative to the total                      |
| dynamic_range_db    | loud frames (90th pct) over quiet frames (10th pct)           |
| crest_db            | peak over RMS                                                 |
| click_rate          | sudden sample jumps per second (crackle)                      |
| stereo_width        | side energy over mid energy (0 = mono)                        |
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

FRAME, HOP = 2048, 1024
ERA_KEYS = (
    "centroid_hz",
    "rolloff_hz",
    "band_low_hz",
    "band_high_hz",
    "hf_ratio_db",
    "dynamic_range_db",
    "crest_db",
    "click_rate",
    "stereo_width",
)
_HZ_KEYS = {"centroid_hz", "rolloff_hz", "band_low_hz", "band_high_hz"}


def _frames(x: np.ndarray) -> np.ndarray:
    if x.size < FRAME:
        x = np.pad(x, (0, FRAME - x.size))
    n = 1 + (x.size - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def _count_clicks(x: np.ndarray, sr: int) -> int:
    """Crackle: 1 ms blocks whose peak second difference is over 3x the median peak of
    the surrounding 31 ms. The second difference emphasises transients over slowly
    varying music; adjacent flagged blocks count as one click. On synthetic tests it
    recovers the simulated crackle rates with no false hits on clean music or noise."""
    block = max(1, sr // 1000)
    hf = np.diff(x, n=2)
    n = hf.size // block
    if n < 31:
        return 0
    peak = np.abs(hf[: n * block].reshape(n, block)).max(axis=1) + 1e-20
    window = np.lib.stride_tricks.sliding_window_view(np.pad(peak, 15, mode="edge"), 31)
    spikes = peak > 3 * np.median(window, axis=1)
    return int(np.count_nonzero(spikes & ~np.concatenate(([False], spikes[:-1]))))


def describe(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Era descriptors of one clip. ``audio`` is (frames,) or (frames, channels)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    mid = x.mean(axis=1)
    side = (x[:, 0] - x[:, 1]) / 2 if x.shape[1] > 1 else np.zeros_like(mid)
    eps = 1e-12

    frames = _frames(mid)
    power = np.abs(np.fft.rfft(frames * np.hanning(FRAME), axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sr)
    spectrum = power.mean(axis=0) + eps
    total = spectrum.sum()

    centroid = float((spectrum * freqs).sum() / total)
    rolloff = float(freqs[np.searchsorted(np.cumsum(spectrum), 0.85 * total)])
    audible = np.nonzero(10 * np.log10(spectrum / spectrum.max()) > -50)[0]
    band_low, band_high = float(freqs[audible[0]]), float(freqs[audible[-1]])
    hf = spectrum[freqs >= 5000].sum()

    frame_db = 10 * np.log10((frames**2).mean(axis=1) + eps)
    rms = math.sqrt(float((mid**2).mean()) + eps)
    clicks = _count_clicks(mid, sr)

    return {
        "centroid_hz": centroid,
        "rolloff_hz": rolloff,
        "band_low_hz": band_low,
        "band_high_hz": band_high,
        "hf_ratio_db": float(10 * np.log10(hf / total + eps)),
        "dynamic_range_db": float(np.percentile(frame_db, 90) - np.percentile(frame_db, 10)),
        "crest_db": float(20 * np.log10(np.max(np.abs(mid)) / rms + eps)),
        "click_rate": clicks / (mid.size / sr),
        "stereo_width": float((side**2).sum() / ((mid**2).sum() + eps)),
    }


def mean_features(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def era_distance(a: Mapping[str, float], b: Mapping[str, float], keys: Sequence[str] = ERA_KEYS):
    """Mean per-feature distance on comparable scales: octaves for frequencies, tens of
    dB for levels, log for click rates, absolute for stereo width. 0 = same sound."""
    parts = []
    for k in keys:
        x, y = a[k], b[k]
        if k in _HZ_KEYS:
            parts.append(abs(math.log2((x + 1.0) / (y + 1.0))))
        elif k.endswith("_db"):
            parts.append(abs(x - y) / 10)
        elif k == "click_rate":
            parts.append(abs(math.log1p(x) - math.log1p(y)))
        else:
            parts.append(abs(x - y))
    return float(np.mean(parts))
