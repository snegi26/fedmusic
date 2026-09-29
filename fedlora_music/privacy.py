"""Client-side differential privacy for the federated adapter update.

Threat model: the server (and anyone observing the wire) is untrusted. Each client
clips its ``global`` adapter update to L2 norm ``C`` and adds Gaussian noise
``N(0, (sigma * C)^2)`` BEFORE sending, so the guarantee holds without trusting
aggregation.

Unit of privacy: the client's *entire* local dataset, in every round it
participates. Two neighbouring datasets can yield clipped updates up to ``2C``
apart, so each release is a Gaussian mechanism with sensitivity ``2C``. Privacy
loss is tracked with Renyi DP and converted to (epsilon, delta) using the
tight conversion of Balle et al. (2020), as used by Opacus.

Raw audio, latents, and the ``personal`` adapter are never released at all.
"""

from __future__ import annotations

import json
import math
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

_RDP_ORDERS: tuple[float, ...] = (
    *(1.0 + x / 10.0 for x in range(1, 100)),
    *(float(a) for a in range(12, 64)),
    128.0,
    256.0,
    512.0,
)


def gaussian_rdp(order: float, noise_multiplier: float, sensitivity_factor: float = 2.0) -> float:
    """RDP of one Gaussian release with noise std ``sigma*C`` and sensitivity ``k*C``."""
    return order * sensitivity_factor**2 / (2.0 * noise_multiplier**2)


def epsilon_for(noise_multipliers: Sequence[float], delta: float) -> float:
    """(epsilon, delta)-DP after one release per entry, each with its own sigma.

    RDP composes additively per order, so rounds run under different noise
    levels (e.g. the server raised sigma mid-run) are accounted exactly.
    """
    if not noise_multipliers:
        return 0.0
    best = math.inf
    for a in _RDP_ORDERS:
        rdp = sum(gaussian_rdp(a, s) for s in noise_multipliers)
        eps = rdp + math.log1p(-1.0 / a) - (math.log(delta) + math.log(a)) / (a - 1.0)
        best = min(best, eps)
    return max(best, 0.0)


def epsilon_after(releases: int, noise_multiplier: float, delta: float) -> float:
    """(epsilon, delta)-DP after ``releases`` updates at a fixed ``noise_multiplier``."""
    return epsilon_for([noise_multiplier] * max(releases, 0), delta)


def l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for t in tensors:
        total += t.detach().double().pow(2).sum()
    return float(total.sqrt())


@torch.no_grad()
def privatize_update(
    delta: Mapping[str, torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    """Clip the joint update to ``clip_norm`` and add calibrated Gaussian noise.

    Only the keys in ``delta`` (the trainable, released coordinates) are touched.
    Returns ``(noised_delta, pre_clip_norm)``.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(secrets.randbits(63))
    norm = l2_norm(delta.values())
    scale = min(1.0, clip_norm / (norm + 1e-12))
    std = noise_multiplier * clip_norm
    noised = {
        k: v.double() * scale + torch.randn(v.shape, generator=generator, dtype=torch.float64) * std
        for k, v in delta.items()
    }
    return {k: v.float() for k, v in noised.items()}, norm


@dataclass
class PrivacyLedger:
    """Per-client privacy accounting, persisted on the client only.

    Stores the noise multiplier of every release so accounting stays exact even
    if the noise level changes between rounds.
    """

    noise_multipliers: list[float] = field(default_factory=list)

    @property
    def releases(self) -> int:
        return len(self.noise_multipliers)

    def epsilon(self, delta: float, extra: float | None = None) -> float:
        """Epsilon spent so far, or after one more release at ``extra``."""
        sigmas = [*self.noise_multipliers, *(() if extra is None else (extra,))]
        return epsilon_for(sigmas, delta)

    def record(self, noise_multiplier: float) -> None:
        self.noise_multipliers.append(float(noise_multiplier))

    @classmethod
    def load(cls, path: Path) -> PrivacyLedger:
        if not path.is_file():
            return cls()
        return cls(noise_multipliers=list(json.loads(path.read_text())["noise_multipliers"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(path)
