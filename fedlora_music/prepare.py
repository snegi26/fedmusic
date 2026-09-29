"""Client-side preprocessing: turn the client's own audio into ACE-Step training tensors.

Runs entirely on the client machine with the locally downloaded model; the audio
and the resulting tensors never leave ``<client-dir>``.

    fedlora-prepare --audio-dir ~/my_songs --client-dir ./clients/client-0 \
        --ace-project-root ../ACE-Step-1.5

Songs kept aside for evaluation go through ``--split eval``; they are written to
``<client-dir>/eval/tensors`` and are never trained on:

    fedlora-prepare --audio-dir ~/my_songs_heldout --split eval \
        --client-dir ./clients/client-0 --ace-project-root ../ACE-Step-1.5
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
    ap.add_argument("--audio-dir", type=Path, help="Folder of the client's own audio files")
    ap.add_argument(
        "--dataset-json", type=Path, help="Optional ACE-Step dataset JSON with captions/lyrics"
    )
    ap.add_argument("--client-dir", type=Path, required=True)
    ap.add_argument("--ace-project-root", type=Path, required=True)
    ap.add_argument("--model-variant", default="acestep-v15-turbo")
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

    from acestep.training_v2.preprocess import preprocess_audio_files

    store = ClientStore(args.client_dir.expanduser().resolve())
    out_dir = store.tensor_dir if args.split == "train" else store.eval_tensor_dir
    result = preprocess_audio_files(
        audio_dir=str(args.audio_dir) if args.audio_dir else None,
        output_dir=str(out_dir),
        checkpoint_dir=str(args.ace_project_root.expanduser().resolve() / "checkpoints"),
        variant=args.model_variant,
        max_duration=args.max_duration,
        dataset_json=str(args.dataset_json) if args.dataset_json else None,
    )
    logging.info("Preprocessed %s/%s files into %s", result["processed"], result["total"], out_dir)
    if result["processed"] == 0:
        return 1
    source = args.audio_dir or args.dataset_json
    store.record_source(args.split, source.expanduser().resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
