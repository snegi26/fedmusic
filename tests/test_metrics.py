"""Embedding metrics on synthetic data with known answers."""

from __future__ import annotations

import math

import pytest
import torch

from fedlora_music.metrics import (
    centroid_similarity,
    copy_check,
    frechet_distance,
    kernel_distance,
    pairwise_diversity,
    personalization,
    prompt_adherence,
)


def _cloud(n: int = 200, d: int = 8, shift: float = 0.0, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed)) + shift


def test_frechet_zero_for_same_set_and_mean_shift_for_translation() -> None:
    a = _cloud()
    assert frechet_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    # Same covariance, mean moved by 0.5 in each of 8 dims: distance = 8 * 0.25.
    assert frechet_distance(a, a + 0.5) == pytest.approx(2.0, rel=1e-6)


def test_frechet_is_symmetric() -> None:
    a, b = _cloud(seed=0), _cloud(seed=1, shift=0.3)
    assert frechet_distance(a, b) == pytest.approx(frechet_distance(b, a), rel=1e-6)


def test_kernel_distance_grows_with_separation() -> None:
    a = _cloud(n=50, seed=0)
    near, far = _cloud(n=50, seed=1), _cloud(n=50, seed=2, shift=3.0)
    assert abs(kernel_distance(a, near)) < kernel_distance(a, far)
    with pytest.raises(ValueError):
        kernel_distance(a[:1], near)


def test_diversity_extremes() -> None:
    assert pairwise_diversity(torch.ones(4, 3)) == pytest.approx(0.0, abs=1e-12)
    assert pairwise_diversity(torch.eye(3)) == pytest.approx(1.0)
    assert math.isnan(pairwise_diversity(torch.ones(1, 3)))


def test_prompt_adherence_is_row_aligned() -> None:
    audio = torch.eye(3)
    assert prompt_adherence(audio, torch.eye(3)) == pytest.approx(1.0)
    assert prompt_adherence(audio, torch.eye(3).flip(0)) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        prompt_adherence(audio, torch.eye(2, 3))


def test_centroid_similarity() -> None:
    a = _cloud(shift=5.0)
    assert centroid_similarity(a, a) == pytest.approx(1.0)
    assert centroid_similarity(a, -a) == pytest.approx(-1.0)


def test_copy_check_flags_exact_copy() -> None:
    training = _cloud(n=10)
    generated = torch.cat([training[:1], _cloud(n=3, seed=5)])
    report = copy_check(generated, training, threshold=0.999)
    assert report.max_nn_similarity == pytest.approx(1.0)
    assert report.rate_above_threshold == pytest.approx(0.25)


def test_personalization_gap_and_top1() -> None:
    own_nearest = [[0.1, 1.0, 1.0], [1.0, 0.2, 1.0], [1.0, 1.0, 0.3]]
    p = personalization(own_nearest)
    assert p.top1 == 1.0
    assert p.gap == pytest.approx([0.9, 0.8, 0.7])
    assert personalization([[1.0, 0.1], [0.1, 1.0]]).top1 == 0.0
    with pytest.raises(ValueError):
        personalization([[0.0]])
