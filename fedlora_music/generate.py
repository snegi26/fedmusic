"""Generate music locally with a client's personalized (global + personal) adapter.

    fedlora-generate --client-dir ./clients/client-0 --model-root ../ACE-Step-1.5 \\
        --caption "warm lo-fi hip hop, dusty drums, rhodes" --duration 60
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from fedlora_music.backends import get_backend
from fedlora_music.cli import add_model_args, model_spec
from fedlora_music.store import ClientStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--client-dir", type=Path, required=True)
    add_model_args(ap)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--lyrics", default="[Instrumental]")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=-1, help="-1: random")
    ap.add_argument("--out-dir", type=Path, default=None, help="Defaults to <client-dir>/generated")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    backend = get_backend(model_spec(args))
    store = ClientStore(args.client_dir.expanduser().resolve())
    adapter_dir = store.model(backend.key).fused_adapter_dir
    if not (adapter_dir / "adapter_config.json").is_file():
        logging.error(
            "No %s adapter at %s - run at least one federated round first", backend.key, adapter_dir
        )
        return 1

    out_dir = (args.out_dir or store.root / "generated").expanduser().resolve()
    seeds = (
        [-1] * args.batch_size if args.seed < 0 else [args.seed + i for i in range(args.batch_size)]
    )
    generator = backend.open_generator()
    try:
        paths = generator.generate(
            adapter_dir,
            args.caption,
            seeds,
            args.duration,
            out_dir,
            lyrics=args.lyrics,
            adapter_scale=args.lora_scale,
        )
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1
    finally:
        generator.close()
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
