"""Client-side preprocessing: turn the client's own audio into a model's training inputs.

Runs entirely on the client machine with the locally installed model; the audio
and the resulting inputs never leave ``<client-dir>``. Inputs are model-specific and
land in ``<client-dir>/models/<backend>/<variant>/``: prepare again after switching
models.

    fedlora-prepare --audio-dir ~/my_songs --client-dir ./clients/client-0 \\
        --model-root ../ACE-Step-1.5

Songs kept aside for evaluation go through ``--split eval``; they are written to
``.../eval/tensors`` and are never trained on:

    fedlora-prepare --audio-dir ~/my_songs_heldout --split eval \\
        --client-dir ./clients/client-0 --model-root ../ACE-Step-1.5
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
    ap.add_argument("--audio-dir", type=Path, help="Folder of the client's own audio files")
    ap.add_argument(
        "--dataset-json", type=Path, help="Optional ACE-Step dataset JSON with captions/lyrics"
    )
    ap.add_argument("--client-dir", type=Path, required=True)
    add_model_args(ap)
    ap.add_argument("--max-duration", type=float, default=240.0)
    ap.add_argument(
        "--split",
        choices=("train", "eval"),
        default="train",
        help="train: used in federated rounds; eval: held out for fedlora-eval only",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.audio_dir is None and args.dataset_json is None:
        ap.error("provide --audio-dir and/or --dataset-json")

    backend = get_backend(model_spec(args))
    store = ClientStore(args.client_dir.expanduser().resolve())
    local = store.model(backend.key)
    out_dir = local.tensor_dir if args.split == "train" else local.eval_tensor_dir
    result = backend.prepare(
        audio_dir=args.audio_dir.expanduser().resolve() if args.audio_dir else None,
        dataset_json=args.dataset_json.expanduser().resolve() if args.dataset_json else None,
        out_dir=out_dir,
        max_duration=args.max_duration,
    )
    logging.info("Prepared %s/%s files into %s", result.processed, result.total, out_dir)
    if result.processed == 0:
        return 1
    source = args.audio_dir or args.dataset_json
    store.record_source(args.split, source.expanduser().resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
