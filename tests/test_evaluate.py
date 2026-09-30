"""fedlora-eval building blocks on the toy model (no ACE-Step, no CLAP download)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import DIM, ROOT, Toy, make_songs
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
from fedlora_music.config import AdapterSpec, ModelSpec
from fedlora_music.evaluate import (
    DEFAULT_PROMPTS,
    VARIANTS,
    load_app_config,
    read_prompts,
    score_variant,
    variant_states,
)
from fedlora_music.store import ClientStore

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
    """What the backend generates with for a variant == what the held-out loss measured."""
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


def _toy_client(tmp_path: Path) -> tuple[Path, ModelSpec]:
    """A client folder on the toy backend with trained-looking adapters and held-out data."""
    from fedlora_music import prepare
    from fedlora_music.model import get_runtime

    spec = replace(load_app_config(ROOT).model, backend="toy", root=None, variant="")
    rt = get_runtime(spec)
    with torch.no_grad():
        for n, p in rt.model.named_parameters():
            if ".lora_B." in n:
                p.normal_(0, 0.1)
    client = ClientStore(tmp_path / "client")
    local = client.model("toy/tiny")
    local.save_tensors(local.local_global_path, adapter_state(rt.model, GLOBAL))
    local.save_tensors(local.personal_adapter_path, adapter_state(rt.model, PERSONAL))
    songs = make_songs(tmp_path / "heldout")
    assert prepare.main(["--audio-dir", str(songs), "--client-dir", str(client.root),
                         "--model-backend", "toy", "--split", "eval"]) == 0  # fmt: skip
    return client.root, spec


def test_heldout_loss_is_deterministic(tmp_path: Path) -> None:
    from fedlora_music.model import get_runtime

    _, spec = _toy_client(tmp_path)
    rt = get_runtime(spec)
    batches = [{"x": torch.randn(4, 16), "y": torch.randn(4, 16)} for _ in range(3)]
    first = trainer.heldout_loss(rt, batches, seed=7)
    torch.manual_seed(123)  # unrelated RNG use in between must not matter
    assert trainer.heldout_loss(rt, batches, seed=7) == first
    assert trainer.heldout_loss(rt, batches, seed=8) != first  # the toy loss has a noise term
    with only_adapters(rt.model, ()):
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


def test_cli_heldout_loss_on_toy_backend(tmp_path: Path) -> None:
    """Real held-out losses through the backend; adapters change the result."""
    from fedlora_music import evaluate

    client, _ = _toy_client(tmp_path)
    out = tmp_path / "report"
    assert evaluate.main([
        "--client-dir", str(client), "--model-backend", "toy", "--app-dir", str(ROOT),
        "--skip-generation", "--out-dir", str(out),
    ]) == 0  # fmt: skip
    report = json.loads((out / "report.json").read_text())
    assert report["model"] == "toy/tiny"
    losses = {v: r["heldout_loss"] for v, r in report["variants"].items()}
    assert losses.keys() == VARIANTS.keys()
    assert len(set(losses.values())) == 4  # each adapter combination is a different model


def test_generate_variants_exports_and_loads_each_adapter(tmp_path: Path) -> None:
    from fedlora_music import evaluate
    from fedlora_music.backends import get_backend

    client, spec = _toy_client(tmp_path)
    local = ClientStore(client).model("toy/tiny")
    states = evaluate._load_adapters(local)
    clips = evaluate.generate_variants(
        get_backend(spec), states, ["base", "fused"], ["a prompt"], 2, 0.05, 0, tmp_path / "gen"
    )
    assert [len(c) for c in clips.values()] == [2, 2]
    base, fused = (p.read_bytes() for p in (clips["base"][0][0], clips["fused"][0][0]))
    assert base != fused  # the exported adapter really changes the output
    assert (tmp_path / "gen" / "adapters" / "fused" / "adapter_config.json").is_file()


def test_cli_rejects_unknown_variant(tmp_path: Path) -> None:
    from fedlora_music import evaluate

    with pytest.raises(SystemExit):
        evaluate.main([
            "--client-dir", str(tmp_path), "--model-root", str(tmp_path),
            "--variants", "nope",
        ])  # fmt: skip
