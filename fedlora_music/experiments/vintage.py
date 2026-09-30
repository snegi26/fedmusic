"""Simulate older recording media on modern audio.

Real historic recordings mix two things: the music of the time and the sound of the
medium (narrow bandwidth, hiss, crackle, speed wobble). Degrading modern tracks with
a medium's characteristics separates the second from the first, so an experiment can
ask whether a client's adapter learns the *sound of an era* independently of genre.

The profiles are coarse approximations for experiments, not restorations or
emulations of any specific device:

| profile          | band (Hz)   | channels | hiss (dB) | crackle/s | wow/flutter        |
|------------------|-------------|----------|-----------|-----------|--------------------|
| shellac-1920s    | 200-4000    | mono     | -30       | 8         | 0.40% @ 1.3 Hz     |
| tape-1950s       | 60-10000    | mono     | -45       | 0         | 0.15% @ 0.5 Hz + flutter |
| vinyl-1970s      | 30-15000    | stereo   | -55       | 1.5       | 0.08% @ 0.55 Hz    |
| cassette-1980s   | 40-12000    | stereo   | -48       | 0         | 0.12% @ 1 Hz + flutter |

Everything is seeded: the same input, profile and seed give the same output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VintageProfile:
    name: str
    low_hz: float  # high-pass corner
    high_hz: float  # low-pass corner
    mono: bool
    hiss_db: float | None  # noise level relative to the signal's RMS; None = no hiss
    crackle_per_s: float  # mean clicks per second (Poisson)
    wow: tuple[tuple[float, float], ...]  # (depth as a fraction of speed, rate in Hz)
    drive: float  # tanh saturation drive; 1.0 = none


PROFILES: dict[str, VintageProfile] = {
    p.name: p
    for p in (
        VintageProfile("clean", 0.0, float("inf"), False, None, 0.0, (), 1.0),
        VintageProfile("shellac-1920s", 200.0, 4000.0, True, -30.0, 8.0, ((0.004, 1.3),), 1.6),
        VintageProfile(
            "tape-1950s", 60.0, 10000.0, True, -45.0, 0.0, ((0.0015, 0.5), (0.0005, 8.0)), 1.25
        ),
        VintageProfile("vinyl-1970s", 30.0, 15000.0, False, -55.0, 1.5, ((0.0008, 0.55),), 1.05),
        VintageProfile(
            "cassette-1980s", 40.0, 12000.0, False, -48.0, 0.0, ((0.0012, 1.0), (0.0008, 10.0)), 1.3
        ),
    )
}


def _band_mask(n: int, sr: int, low: float, high: float) -> np.ndarray:
    """rfft-domain band-pass with third-octave raised-cosine skirts (no ringing edges)."""
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = np.ones_like(freqs)
    skirt = 2 ** (1 / 3)
    if low > 0:
        lo0, lo1 = low / skirt, low
        mask[freqs <= lo0] = 0.0
        ramp = (freqs > lo0) & (freqs < lo1)
        mask[ramp] *= 0.5 - 0.5 * np.cos(np.pi * (freqs[ramp] - lo0) / (lo1 - lo0))
    if np.isfinite(high) and high < sr / 2:
        hi0, hi1 = high, min(high * skirt, sr / 2)
        mask[freqs >= hi1] = 0.0
        ramp = (freqs > hi0) & (freqs < hi1)
        mask[ramp] *= 0.5 + 0.5 * np.cos(np.pi * (freqs[ramp] - hi0) / (hi1 - hi0))
    return mask


def _band_limit(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    mask = _band_mask(x.shape[0], sr, low, high)[:, None]
    return np.fft.irfft(np.fft.rfft(x, axis=0) * mask, n=x.shape[0], axis=0)


def _wow(x: np.ndarray, sr: int, wow: tuple[tuple[float, float], ...], rng: np.random.Generator):
    """Periodic speed variation: read the signal at slowly wobbling positions."""
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64)
    pos = t.copy()
    for depth, rate in wow:
        phase = rng.uniform(0, 2 * np.pi)
        omega = 2 * np.pi * rate / sr
        pos += depth / omega * np.sin(omega * t + phase)  # d(pos)/dt = 1 + depth*cos(...)
    pos = np.clip(pos, 0, n - 1)
    return np.stack([np.interp(pos, t, x[:, c]) for c in range(x.shape[1])], axis=1)


def _crackle(n: int, sr: int, rate: float, level: float, rng: np.random.Generator):
    out = np.zeros(n)
    count = rng.poisson(rate * n / sr)
    decay = np.exp(-np.arange(int(0.001 * sr) + 1) / (0.0002 * sr))  # ~1 ms click
    for start in rng.integers(0, n, size=count):
        amp = rng.uniform(0.2, 1.0) * level * rng.choice((-1.0, 1.0))
        end = min(n, start + decay.size)
        out[start:end] += amp * decay[: end - start]
    return out


def degrade(audio: np.ndarray, sr: int, profile: str | VintageProfile, seed: int = 0) -> np.ndarray:
    """Apply a recording-medium profile. ``audio`` is (frames,) or (frames, channels),
    float in [-1, 1]; the result is (frames, channels), float32, peak <= 0.99."""
    p = PROFILES[profile] if isinstance(profile, str) else profile
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if p.name == "clean":
        return x.astype(np.float32)
    rng = np.random.default_rng(seed)
    if p.mono:
        x = x.mean(axis=1, keepdims=True)
    if p.wow:
        x = _wow(x, sr, p.wow, rng)
    if p.drive > 1.0:
        x = np.tanh(p.drive * x) / np.tanh(p.drive)
    rms = float(np.sqrt(np.mean(x**2))) or 1e-4
    if p.hiss_db is not None:
        x = x + rng.standard_normal(x.shape) * rms * 10 ** (p.hiss_db / 20)
    # The recording's bandwidth limits the music and the hiss; surface crackle comes
    # from the disc itself at playback, after that limit.
    x = _band_limit(x, sr, p.low_hz, p.high_hz)
    if p.crackle_per_s > 0:
        x = x + _crackle(x.shape[0], sr, p.crackle_per_s, 4 * rms, rng)[:, None]
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32)
