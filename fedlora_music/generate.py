"""Generate music locally with a client's personalized (global + personal) adapter.

    fedlora-generate --client-dir ./clients/client-0 --ace-project-root ../ACE-Step-1.5 \
        --caption "warm lo-fi hip hop, dusty drums, rhodes" --duration 60
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fedlora_music.store import ClientStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--client-dir", type=Path, required=True)
    ap.add_argument("--ace-project-root", type=Path, required=True)
    ap.add_argument("--model-variant", default="acestep-v15-turbo")
    ap.add_argument("--caption", required=True)
    ap.add_argument("--lyrics", default="[Instrumental]")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path, default=None, help="Defaults to <client-dir>/generated")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    store = ClientStore(args.client_dir.expanduser().resolve())
    adapter_dir = store.fused_adapter_dir
    if not (adapter_dir / "adapter_config.json").is_file():
        logging.error(
            "No fused adapter at %s - run at least one federated round first", adapter_dir
        )
        return 1

    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(args.ace_project_root.expanduser().resolve()),
        config_path=args.model_variant,
        device="auto",
    )
    if not ok:
        logging.error("ACE-Step init failed: %s", status)
        return 1

    msg = dit.load_lora(str(adapter_dir))
    if not msg.startswith("✅"):
        logging.error("Adapter load failed: %s", msg)
        return 1
    dit.set_lora_scale(args.lora_scale)

    out_dir = (args.out_dir or store.root / "generated").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = generate_music(
        dit,
        None,  # no LM planner: DiT-only, caption drives the prompt
        GenerationParams(
            caption=args.caption,
            lyrics=args.lyrics,
            duration=args.duration,
            seed=args.seed,
            thinking=False,
        ),
        GenerationConfig(batch_size=args.batch_size, audio_format="flac"),
        save_dir=str(out_dir),
    )
    if not result.success:
        logging.error("Generation failed: %s", result.error)
        return 1
    for audio in result.audios:
        print(audio["path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
