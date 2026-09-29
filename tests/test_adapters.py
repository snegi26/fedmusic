"""Tests for dual-adapter injection and exact fusion (toy decoder, no ACE-Step needed)."""

from __future__ import annotations

import copy

import pytest
import torch
from peft import PeftModel
from torch import nn

from fedlora_music.adapters import (
    GLOBAL,
    PERSONAL,
    adapter_state,
    export_fused_adapter,
    inject_dual_lora,
    load_adapter_state,
    trainable_names,
)
from fedlora_music.config import AdapterSpec


class _Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.relu(self.q_proj(x)))


class _Toy(nn.Module):
    def __init__(self, d: int = 16) -> None:
        super().__init__()
        self.decoder = nn.Sequential(_Block(d), _Block(d))


SPEC = AdapterSpec(
    target_modules=("q_proj", "o_proj"),
    global_rank=4,
    global_alpha=8,
    personal_rank=2,
    personal_alpha=2,
    lora_dropout=0.0,
    freeze_global_a=True,
)


@pytest.fixture
def base() -> _Toy:
    torch.manual_seed(0)
    return _Toy()


def _randomize(model: nn.Module) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_" in n:
                p.normal_(0, 0.1)


def test_ffa_freezes_only_global_a(base: _Toy) -> None:
    model = inject_dual_lora(base, SPEC)
    g = trainable_names(model, GLOBAL)
    assert g and all(".lora_B." in n for n in g)
    assert any(".lora_A." in n for n in trainable_names(model, PERSONAL))
    assert not any(p.requires_grad for n, p in model.named_parameters() if ".lora_" not in n)


def test_state_roundtrip(base: _Toy) -> None:
    model = inject_dual_lora(base, SPEC)
    _randomize(model)
    state = adapter_state(model, GLOBAL)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if f".{GLOBAL}." in n:
                p.zero_()
    load_adapter_state(model, GLOBAL, state)
    for k, v in adapter_state(model, GLOBAL).items():
        assert torch.equal(v, state[k])


def test_fused_adapter_matches_dual_forward(base: _Toy, tmp_path) -> None:
    pristine = copy.deepcopy(base.decoder)
    model = inject_dual_lora(base, SPEC)
    _randomize(model)
    x = torch.randn(3, 16)
    with torch.no_grad():
        y_dual = model.decoder.base_model(x)

    out = export_fused_adapter(
        adapter_state(model, GLOBAL), adapter_state(model, PERSONAL), SPEC, tmp_path / "fused"
    )
    fused = PeftModel.from_pretrained(pristine, str(out))
    with torch.no_grad():
        y_fused = fused.base_model(x)
    assert torch.allclose(y_dual, y_fused, atol=1e-5)
