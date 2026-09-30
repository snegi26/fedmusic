"""Dual-adapter LoRA plumbing, independent of the base model.

Each model backend names the submodule that gets the adapters (its ``adapter_root``;
``decoder`` for ACE-Step's DiT). Two PEFT LoRA adapters live side by side on that
submodule's target projections:

* ``global``   - federated. Its (clipped, noised) update is the ONLY thing a client
                 ever sends. With ``freeze_global_a`` (FFA-LoRA) the random ``A``
                 matrices are fixed and shared, so FedAvg over ``B`` is exact.
* ``personal`` - captures the client's own style. Never leaves the device.

Both are active in the forward pass, so ``W' = W + s_g B_g A_g + s_p B_p A_p``.
For inference the pair is fused into one standard PEFT adapter by stacking along the
rank dimension: a standard PEFT adapter directory (ACE-Step's
``AceStepHandler.load_lora`` loads it as-is).
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import save_file
from torch import nn

from fedlora_music.config import AdapterSpec

GLOBAL = "global"
PERSONAL = "personal"

_KEY_RE = re.compile(r"^(?P<prefix>.+)\.lora_(?P<ab>[AB])\.(?P<adapter>[^.]+)\.weight$")

StateDict = OrderedDict[str, torch.Tensor]
DEFAULT_ROOT = "decoder"


def _lora_config(spec: AdapterSpec, rank: int, alpha: int) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=list(spec.target_modules),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,  # same as ACE-Step's own trainer
    )


def _parse(name: str) -> re.Match[str] | None:
    return _KEY_RE.match(name)


def is_param_of(name: str, adapter: str) -> bool:
    m = _parse(name)
    return m is not None and m["adapter"] == adapter


def inject_dual_lora(model: nn.Module, spec: AdapterSpec, root: str = DEFAULT_ROOT) -> nn.Module:
    """Wrap ``model.<root>`` with the ``global`` and ``personal`` adapters, both active.

    All base weights are frozen; adapter weights are kept in fp32 for stable
    optimisation (the base runs in bf16 under autocast).
    """
    peft_root = get_peft_model(
        getattr(model, root),
        _lora_config(spec, spec.global_rank, spec.global_alpha),
        adapter_name=GLOBAL,
    )
    peft_root.add_adapter(PERSONAL, _lora_config(spec, spec.personal_rank, spec.personal_alpha))
    # PeftModel.set_adapter takes a single name; the tuner accepts a list.
    peft_root.base_model.set_adapter([GLOBAL, PERSONAL])
    setattr(model, root, peft_root)

    for name, param in model.named_parameters():
        m = _parse(name)
        trainable = m is not None and m["adapter"] in (GLOBAL, PERSONAL)
        if trainable and spec.freeze_global_a and m["adapter"] == GLOBAL and m["ab"] == "A":
            trainable = False
        param.requires_grad_(trainable)
        if m is not None:
            param.data = param.data.float()
    return model


@contextmanager
def only_adapters(
    model: nn.Module, adapters: Sequence[str], root: str = DEFAULT_ROOT
) -> Iterator[None]:
    """Temporarily run the decoder with just ``adapters`` active (none = base model).

    Used for evaluation. ``requires_grad`` flags are restored afterwards, because
    PEFT's ``set_adapter`` re-enables gradients on every active adapter, which would
    silently unfreeze the shared FFA-LoRA ``A`` matrices.
    """
    peft_root = getattr(model, root)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    try:
        if adapters:
            peft_root.base_model.set_adapter(list(adapters))
            yield
        else:
            with peft_root.disable_adapter():
                yield
    finally:
        peft_root.base_model.set_adapter([GLOBAL, PERSONAL])
        for name, param in model.named_parameters():
            param.requires_grad_(flags[name])


def trainable_names(model: nn.Module, adapter: str) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad and is_param_of(n, adapter)]


def adapter_state(model: nn.Module, adapter: str) -> StateDict:
    """Detached fp32 CPU copy of one adapter's weights (numpy-serialisable for Flower)."""
    state = OrderedDict(
        (n, p.detach().to("cpu", torch.float32).clone())
        for n, p in model.named_parameters()
        if is_param_of(n, adapter)
    )
    if not state:
        raise RuntimeError(f"adapter {adapter!r} not found on model")
    return state


@torch.no_grad()
def load_adapter_state(model: nn.Module, adapter: str, state: dict[str, torch.Tensor]) -> None:
    """Copy ``state`` into the adapter in place. Keys and shapes must match exactly."""
    params = {n: p for n, p in model.named_parameters() if is_param_of(n, adapter)}
    if params.keys() != state.keys():
        missing, extra = params.keys() - state.keys(), state.keys() - params.keys()
        raise KeyError(
            f"adapter {adapter!r} key mismatch: missing={sorted(missing)[:3]} "
            f"extra={sorted(extra)[:3]}"
        )
    for name, param in params.items():
        src = state[name]
        if src.shape != param.shape:
            raise ValueError(f"{name}: shape {tuple(src.shape)} != {tuple(param.shape)}")
        param.copy_(src.to(param.device, param.dtype))


@torch.no_grad()
def reset_adapter(model: nn.Module, adapter: str, seed: int) -> None:
    """Re-initialise an adapter with PEFT defaults (A ~ Kaiming-uniform, B = 0)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in model.modules():
            if isinstance(module, LoraLayer) and adapter in module.lora_A:
                module.reset_lora_parameters(adapter, True)
    for name, param in model.named_parameters():
        if is_param_of(name, adapter):
            param.data = param.data.float()


def fuse_adapters(
    global_state: dict[str, torch.Tensor],
    personal_state: dict[str, torch.Tensor],
    spec: AdapterSpec,
    root: str = DEFAULT_ROOT,
) -> StateDict:
    """Fuse both adapters into one rank ``r_g + r_p`` adapter with scaling 1.

    ``s_g B_g A_g + s_p B_p A_p == [s_g B_g | s_p B_p] @ [A_g ; A_p]`` exactly.
    Returned keys follow the PEFT on-disk convention relative to ``model.<root>``
    (``base_model.model.<path>.lora_{A,B}.weight``).
    """
    s_g = spec.global_alpha / spec.global_rank
    s_p = spec.personal_alpha / spec.personal_rank

    def by_prefix(
        state: dict[str, torch.Tensor], adapter: str
    ) -> dict[str, dict[str, torch.Tensor]]:
        out: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in state.items():
            m = _parse(key)
            if m is None or m["adapter"] != adapter:
                raise KeyError(f"unexpected key for adapter {adapter!r}: {key}")
            out.setdefault(m["prefix"], {})[m["ab"]] = tensor
        return out

    g, p = by_prefix(global_state, GLOBAL), by_prefix(personal_state, PERSONAL)
    if g.keys() != p.keys():
        raise ValueError("global and personal adapters target different modules")

    fused: StateDict = OrderedDict()
    for prefix in sorted(g):
        a = torch.cat([g[prefix]["A"], p[prefix]["A"]], dim=0)  # (r_g + r_p, in)
        b = torch.cat([s_g * g[prefix]["B"], s_p * p[prefix]["B"]], dim=1)  # (out, r_g + r_p)
        rel = prefix.removeprefix(f"{root}.")
        fused[f"{rel}.lora_A.weight"] = a.contiguous()
        fused[f"{rel}.lora_B.weight"] = b.contiguous()
    return fused


def export_fused_adapter(
    global_state: dict[str, torch.Tensor],
    personal_state: dict[str, torch.Tensor],
    spec: AdapterSpec,
    out_dir: Path,
    root: str = DEFAULT_ROOT,
) -> Path:
    """Write a standard PEFT adapter directory (``PeftModel.from_pretrained`` loads it)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = spec.global_rank + spec.personal_rank
    fused = fuse_adapters(global_state, personal_state, spec, root)
    save_file(fused, str(out_dir / "adapter_model.safetensors"))
    cfg = _lora_config(spec, rank=rank, alpha=rank)  # alpha / r == 1: scales folded into B
    cfg.lora_dropout = 0.0
    cfg.save_pretrained(str(out_dir))
    (out_dir / "fedlora_meta.json").write_text(
        json.dumps({"global_rank": spec.global_rank, "personal_rank": spec.personal_rank}, indent=2)
    )
    return out_dir
