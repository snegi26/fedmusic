"""Client-side preprocessing: turn the client's own audio into ACE-Step training tensors.

Runs entirely on the client machine with the locally downloaded model; the audio
and the resulting tensors never leave ``<client-dir>``.

    fedlora-prepare --audio-dir ~/my_songs --client-dir ./clients/client-0 \
        --ace-project-root ../ACE-Step-1.5
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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.audio_dir is None and args.dataset_json is None:
        ap.error("provide --audio-dir and/or --dataset-json")

    from acestep.training_v2.preprocess import preprocess_audio_files

    store = ClientStore(args.client_dir.expanduser().resolve())
    result = preprocess_audio_files(
        audio_dir=str(args.audio_dir) if args.audio_dir else None,
        output_dir=str(store.tensor_dir),
        checkpoint_dir=str(args.ace_project_root.expanduser().resolve() / "checkpoints"),
        variant=args.model_variant,
        max_duration=args.max_duration,
        dataset_json=str(args.dataset_json) if args.dataset_json else None,
    )
    logging.info(
        "Preprocessed %s/%s files into %s", result["processed"], result["total"], store.tensor_dir
    )
    return 0 if result["processed"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
