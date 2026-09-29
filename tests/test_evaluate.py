"""fedlora-eval building blocks on the toy model (no ACE-Step, no CLAP download)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from conftest import DIM, ROOT, Toy
from peft import PeftModel

from fedlora_music import trainer
from fedlora_music.adapters import (
    GLOBAL,
    PERSONAL,
    adapter_state,
    export_fused_adapter,
    inject_dual_lora,
    only_adapters,
)
from fedlora_music.config import AdapterSpec
from fedlora_music.evaluate import (
    DEFAULT_PROMPTS,
    VARIANTS,
    load_app_config,
    read_prompts,
    score_variant,
    variant_states,
)
from fedlora_music.model import Runtime

SPEC = AdapterSpec(("q_proj", "o_proj"), 4, 8, 2, 2, 0.0, True)


def _trained_toy() -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(0)
    base = Toy()
    pristine = copy.deepcopy(base.decoder)
    model = inject_dual_lora(base, SPEC)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_" in n:
                p.normal_(0, 0.1)
    return model, pristine


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_exported_variant_matches_live_adapter_switch(variant: str, tmp_path: Path) -> None:
    """What ACE-Step generates with for a variant == what the held-out loss measured."""
    model, pristine = _trained_toy()
    x = torch.randn(3, DIM)
    with torch.no_grad(), only_adapters(model, VARIANTS[variant]):
        live = model.decoder.base_model(x)
    if variant == "base":
        expected = pristine(x)
    else:
        states = variant_states(
            adapter_state(model, GLOBAL), adapter_state(model, PERSONAL), variant
        )
        out = export_fused_adapter(*states, SPEC, tmp_path / variant)
        expected = PeftModel.from_pretrained(pristine, str(out)).base_model(x)
    assert torch.allclose(live, expected.detach(), atol=1e-5)


def test_only_adapters_keeps_shared_a_frozen() -> None:
    model, _ = _trained_toy()
    before = {n: p.requires_grad for n, p in model.named_parameters()}
    for adapters in VARIANTS.values():
        with only_adapters(model, adapters):
            pass
    assert {n: p.requires_grad for n, p in model.named_parameters()} == before


def test_heldout_loss_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    def noisy_loss(rt: Runtime, batch: dict[str, torch.Tensor], cfg_ratio: float) -> torch.Tensor:
        assert cfg_ratio == 0.0
        noise = torch.randn_like(batch["x"])
        return torch.nn.functional.mse_loss(rt.model.decoder.base_model(batch["x"] + noise), noise)

    monkeypatch.setattr(trainer, "flow_matching_loss", noisy_loss)
    model, _ = _trained_toy()
    rt = Runtime(model, torch.device("cpu"), torch.float32, -0.4, 1.0, 0.0)
    batches = [{"x": torch.randn(4, DIM)} for _ in range(3)]

    first = trainer.heldout_loss(rt, batches, seed=7)
    torch.manual_seed(123)  # unrelated RNG use in between must not matter
    assert trainer.heldout_loss(rt, batches, seed=7) == first
    with only_adapters(model, ()):
        assert trainer.heldout_loss(rt, batches, seed=7) != first


def test_score_variant_reports_all_metrics() -> None:
    g = torch.Generator().manual_seed(0)
    gen, ref = torch.randn(6, 8, generator=g), torch.randn(5, 8, generator=g)
    out = score_variant(gen, gen.clone(), ref, ref, copy_threshold=0.9)
    assert out["clips"] == 6
    assert out["prompt_adherence"] == pytest.approx(1.0)
    assert {"kad_to_reference", "centroid_similarity", "diversity", "copy"} <= out.keys()
    assert out["fad_to_reference"] is None  # 6 clips in 8 dims: covariance is meaningless
    bare = score_variant(gen, gen, None, None, copy_threshold=0.9)
    assert "kad_to_reference" not in bare and "copy" not in bare


def test_read_prompts_precedence(tmp_path: Path) -> None:
    ds = tmp_path / "eval.json"
    ds.write_text(json.dumps({"samples": [{"caption": "jazz trio"}, {"caption": ""}]}))
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("# comment\n\nsynthwave\n")
    assert read_prompts(prompts, ds) == ["synthwave"]
    assert read_prompts(None, ds) == ["jazz trio"]
    assert read_prompts(None, tmp_path) == list(DEFAULT_PROMPTS)


def test_load_app_config_reads_run_config() -> None:
    cfg = load_app_config(ROOT)
    assert cfg.model.adapters.freeze_global_a is True
    assert cfg.privacy.delta == pytest.approx(1e-5)


def test_cli_writes_local_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Held-out loss only: adapters from the client's state, sources from `fedlora-prepare`."""
    from fedlora_music import evaluate
    from fedlora_music.store import ClientStore

    model, _ = _trained_toy()
    store = ClientStore(tmp_path / "client")
    store.save_tensors(store.local_global_path, adapter_state(model, GLOBAL))
    store.save_tensors(store.personal_adapter_path, adapter_state(model, PERSONAL))
    store.eval_tensor_dir.mkdir(parents=True)
    store.record_source("eval", tmp_path / "heldout")
    assert store.load_sources() == {"eval": str(tmp_path / "heldout")}

    seen = {}

    def fake_losses(spec, store_, states, variants, seed):
        seen["root"] = spec.ace_project_root
        return {v: float(i) for i, v in enumerate(variants)}

    monkeypatch.setattr(evaluate, "heldout_losses", fake_losses)
    out = tmp_path / "report"
    assert evaluate.main([
        "--client-dir", str(store.root), "--ace-project-root", str(tmp_path / "ace"),
        "--app-dir", str(ROOT), "--variants", "base,fused", "--skip-generation",
        "--out-dir", str(out),
    ]) == 0  # fmt: skip
    report = json.loads((out / "report.json").read_text())
    assert report["variants"] == {"base": {"heldout_loss": 0.0}, "fused": {"heldout_loss": 1.0}}
    assert seen["root"] == (tmp_path / "ace").resolve()


def test_cli_rejects_unknown_variant(tmp_path: Path) -> None:
    from fedlora_music import evaluate

    with pytest.raises(SystemExit):
        evaluate.main([
            "--client-dir", str(tmp_path), "--ace-project-root", str(tmp_path),
            "--variants", "nope",
        ])  # fmt: skip
