"""Tests for client-side DP. Run: pytest -q"""

from __future__ import annotations

import math

import torch

from fedlora_music.privacy import (
    PrivacyLedger,
    epsilon_after,
    epsilon_for,
    l2_norm,
    privatize_update,
)


def test_clip_bounds_norm_without_noise_contribution() -> None:
    delta = {"a": torch.full((10,), 3.0), "b": torch.full((5,), 4.0)}
    gen = torch.Generator().manual_seed(0)
    # tiny sigma isolates the clipping behaviour
    out, pre = privatize_update(delta, clip_norm=1.0, noise_multiplier=1e-9, generator=gen)
    assert math.isclose(pre, l2_norm(delta.values()), rel_tol=1e-6)
    assert math.isclose(l2_norm(out.values()), 1.0, rel_tol=1e-4)


def test_small_update_is_not_scaled_up() -> None:
    delta = {"a": torch.tensor([0.1, 0.0])}
    out, _ = privatize_update(
        delta, clip_norm=1.0, noise_multiplier=1e-9, generator=torch.Generator().manual_seed(0)
    )
    assert torch.allclose(out["a"], delta["a"], atol=1e-6)


def test_noise_std_matches_sigma_times_clip() -> None:
    delta = {"a": torch.zeros(200_000)}
    out, _ = privatize_update(
        delta, clip_norm=0.5, noise_multiplier=2.0, generator=torch.Generator().manual_seed(0)
    )
    assert math.isclose(out["a"].std().item(), 1.0, rel_tol=0.02)


def test_epsilon_monotone_in_rounds_and_sigma() -> None:
    e1, e10 = epsilon_after(1, 1.0, 1e-5), epsilon_after(10, 1.0, 1e-5)
    assert 0 < e1 < e10
    assert epsilon_after(10, 4.0, 1e-5) < e10
    assert epsilon_after(0, 1.0, 1e-5) == 0.0


def test_epsilon_single_release_against_closed_form_bound() -> None:
    # Gaussian mechanism, sensitivity 2C, noise sigma*C: classical bound
    # eps <= (2/sigma) * sqrt(2 ln(1.25/delta)) for eps < 1 regime (sigma large).
    sigma, delta = 20.0, 1e-5
    classical = (2.0 / sigma) * math.sqrt(2 * math.log(1.25 / delta))
    assert epsilon_after(1, sigma, delta) <= classical


def test_ledger_roundtrip_and_mixed_sigma_accounting(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    ledger = PrivacyLedger.load(path)
    assert ledger.releases == 0 and ledger.epsilon(1e-5) == 0.0
    ledger.record(4.0)
    ledger.record(2.0)
    ledger.save(path)
    loaded = PrivacyLedger.load(path)
    assert loaded.noise_multipliers == [4.0, 2.0]
    assert math.isclose(loaded.epsilon(1e-5), epsilon_for([4.0, 2.0], 1e-5))
    # a noisier round costs less than a quieter one
    assert loaded.epsilon(1e-5, extra=8.0) < loaded.epsilon(1e-5, extra=1.0)
