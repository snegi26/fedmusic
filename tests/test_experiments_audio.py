"""Vintage simulation and era descriptors on synthetic signals with known properties."""

from __future__ import annotations

import numpy as np
import pytest

from fedlora_music.experiments.features import ERA_KEYS, describe, era_distance, mean_features
from fedlora_music.experiments.vintage import PROFILES, degrade

SR = 22_050


def _music(seconds: float = 2.0, stereo: bool = True) -> np.ndarray:
    """Broadband stand-in for music: harmonics up to ~9 kHz plus a little noise."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([110, 440, 1760, 5000, 9000]))
    x = 0.2 * x + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    return np.stack([x, np.roll(x, 50)], axis=1) if stereo else x


def test_clean_is_identity_and_profiles_are_deterministic() -> None:
    x = _music()
    assert np.allclose(degrade(x, SR, "clean"), x, atol=1e-6)
    for name in PROFILES:
        a, b = degrade(x, SR, name, seed=3), degrade(x, SR, name, seed=3)
        assert np.array_equal(a, b)
        assert a.shape[0] == x.shape[0] and np.isfinite(a).all() and np.abs(a).max() <= 0.99
    assert not np.array_equal(
        degrade(x, SR, "shellac-1920s", 1), degrade(x, SR, "shellac-1920s", 2)
    )


def test_shellac_sounds_older_than_the_original() -> None:
    x = _music()
    clean, old = describe(x, SR), describe(degrade(x, SR, "shellac-1920s"), SR)
    assert old["band_high_hz"] < 5000 < clean["band_high_hz"]  # narrow band
    assert old["hf_ratio_db"] < clean["hf_ratio_db"] - 10  # little energy above 5 kHz
    assert old["stereo_width"] == pytest.approx(0.0, abs=1e-9)  # mono
    assert old["click_rate"] > clean["click_rate"]  # crackle
    # Media get further from the original the older they are.
    d = {n: era_distance(clean, describe(degrade(x, SR, n), SR)) for n in PROFILES}
    assert d["clean"] == pytest.approx(0.0, abs=1e-6)
    assert d["shellac-1920s"] > d["tape-1950s"] > d["vinyl-1970s"]


def test_describe_known_signals() -> None:
    t = np.arange(SR) / SR
    tone = describe(0.5 * np.sin(2 * np.pi * 1000 * t), SR)
    assert tone["centroid_hz"] == pytest.approx(1000, rel=0.05)
    assert tone["stereo_width"] == 0.0 and set(tone) == set(ERA_KEYS)
    noise = describe(0.1 * np.random.default_rng(1).standard_normal(SR), SR)
    assert noise["rolloff_hz"] > 8000 and noise["band_high_hz"] > 10000
    assert mean_features([tone, tone]) == tone and mean_features([]) == {}
