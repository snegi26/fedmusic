"""Quality metrics over audio/text embeddings. Pure torch, no model code.

All functions take embedding matrices of shape ``(n, d)``, one row per clip, and
compute in float64. They are used by ``fedlora-eval`` (per client, locally) and by
``fedlora-benchmark`` (across simulated clients).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


def _as64(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected an (n, d) embedding matrix, got shape {tuple(x.shape)}")
    return x.detach().to("cpu", torch.float64)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity, shape ``(len(a), len(b))``."""
    return _normalize(_as64(a)) @ _normalize(_as64(b)).T


def frechet_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fréchet distance between Gaussians fitted to ``a`` and ``b`` (FAD on audio embeddings).

    ``||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a^1/2 S_b S_a^1/2)^1/2)``. The covariance
    estimate is only meaningful with many more clips than embedding dimensions; use
    :func:`kernel_distance` for small sets.
    """
    a, b = _as64(a), _as64(b)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("frechet_distance needs at least 2 clips per set")
    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = torch.cov(a.T), torch.cov(b.T)
    s_a, s_b = torch.atleast_2d(s_a), torch.atleast_2d(s_b)

    def sqrtm_psd(m: torch.Tensor) -> torch.Tensor:
        vals, vecs = torch.linalg.eigh((m + m.T) / 2)
        return (vecs * vals.clamp_min(0).sqrt()) @ vecs.T

    root_a = sqrtm_psd(s_a)
    cross = torch.linalg.eigvalsh(root_a @ s_b @ root_a).clamp_min(0).sqrt().sum()
    dist = (mu_a - mu_b).pow(2).sum() + s_a.trace() + s_b.trace() - 2 * cross
    return max(float(dist), 0.0)


def kernel_distance(a: torch.Tensor, b: torch.Tensor, bandwidth: float | None = None) -> float:
    """Unbiased squared MMD with a Gaussian kernel (Kernel Audio Distance, unscaled).

    Distribution-free and unbiased for small sets, which is the normal case for a
    client with a handful of songs. ``bandwidth`` defaults to the median pairwise
    distance of the pooled set. Can be slightly negative when the sets match.
    """
    a, b = _as64(a), _as64(b)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("kernel_distance needs at least 2 clips per set")
    if bandwidth is None:
        pooled = torch.cat([a, b])
        d = torch.cdist(pooled, pooled)
        off_diag = d[~torch.eye(len(pooled), dtype=torch.bool)]
        bandwidth = float(off_diag.median()) or 1.0

    def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(x, y).pow(2) / (2 * bandwidth**2))

    m, n = len(a), len(b)
    k_aa, k_bb = k(a, a), k(b, b)
    term_aa = (k_aa.sum() - k_aa.diagonal().sum()) / (m * (m - 1))
    term_bb = (k_bb.sum() - k_bb.diagonal().sum()) / (n * (n - 1))
    return float(term_aa + term_bb - 2 * k(a, b).mean())


def centroid_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between the mean normalized embeddings of two sets."""
    ca = _normalize(_as64(a)).mean(0, keepdim=True)
    cb = _normalize(_as64(b)).mean(0, keepdim=True)
    return float(cosine_matrix(ca, cb)[0, 0])


def pairwise_diversity(a: torch.Tensor) -> float:
    """Mean cosine distance between distinct clips. Near 0 means near-identical outputs."""
    if len(a) < 2:
        return math.nan
    sim = cosine_matrix(a, a)
    n = len(a)
    return float(1.0 - (sim.sum() - sim.diagonal().sum()) / (n * (n - 1)))


def prompt_adherence(audio: torch.Tensor, text: torch.Tensor) -> float:
    """Mean cosine similarity between each clip and its own prompt (row-aligned, CLAP space)."""
    if audio.shape[0] != text.shape[0]:
        raise ValueError("audio and text embeddings must be row-aligned")
    return float((_normalize(_as64(audio)) * _normalize(_as64(text))).sum(1).mean())


@dataclass(frozen=True)
class CopyReport:
    """Nearest-training-clip similarity of each generated clip."""

    mean_nn_similarity: float
    max_nn_similarity: float
    rate_above_threshold: float
    threshold: float


def copy_check(generated: torch.Tensor, training: torch.Tensor, threshold: float) -> CopyReport:
    """Flag generations whose nearest training clip is suspiciously close.

    A screen, not proof: high embedding similarity means "listen to this pair",
    and low similarity does not rule out copied melodies.
    """
    nn_sim = cosine_matrix(generated, training).max(dim=1).values
    return CopyReport(
        mean_nn_similarity=float(nn_sim.mean()),
        max_nn_similarity=float(nn_sim.max()),
        rate_above_threshold=float((nn_sim >= threshold).double().mean()),
        threshold=threshold,
    )


@dataclass(frozen=True)
class PersonalizationReport:
    """How much closer each client's generations are to its own songs than to others'.

    ``gap[i] = mean_{j != i} D[i, j] - D[i, i]`` for a distance matrix
    ``D[i, j] = distance(generated_i, reference_j)``. Positive is good.
    ``top1`` is the fraction of clients whose own reference set is the nearest one.
    """

    gap: list[float]
    mean_gap: float
    top1: float


def personalization(distances: Sequence[Sequence[float]]) -> PersonalizationReport:
    d = torch.as_tensor(distances, dtype=torch.float64)
    n = d.shape[0]
    if d.ndim != 2 or d.shape[1] != n or n < 2:
        raise ValueError("need a square distance matrix over at least 2 clients")
    own = d.diagonal()
    others = (d.sum(1) - own) / (n - 1)
    gap = others - own
    top1 = (d.argmin(1) == torch.arange(n)).double().mean()
    return PersonalizationReport(gap=gap.tolist(), mean_gap=float(gap.mean()), top1=float(top1))
