"""Command-line options shared by the ``fedlora-*`` tools."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fedlora_music.config import ModelSpec


def add_model_args(ap: argparse.ArgumentParser) -> None:
    """``--model-backend``, ``--model-root`` (alias ``--ace-project-root``), ``--model-variant``."""
    from fedlora_music.backends import backend_names

    ap.add_argument(
        "--model-backend",
        default=None,
        help=f"one of: {', '.join(backend_names())} (default: acestep, or the run config's)",
    )
    ap.add_argument(
        "--model-root",
        "--ace-project-root",
        dest="model_root",
        type=Path,
        default=os.environ.get("FEDLORA_MODEL_ROOT") or None,
        help="the backend's install folder (ACE-Step: the checkout holding checkpoints/); "
        "default: $FEDLORA_MODEL_ROOT",
    )
    ap.add_argument(
        "--model-variant", default=None, help="model inside the backend (default: its default)"
    )


def model_spec(args: argparse.Namespace, run: ModelSpec | None = None) -> ModelSpec:
    """Model from CLI flags, falling back to the run config's ``run`` spec if given."""
    root = args.model_root.expanduser().resolve() if args.model_root else None
    return ModelSpec(
        backend=args.model_backend or (run.backend if run else "acestep"),
        root=root or (run.root if run else None),
        variant=args.model_variant
        if args.model_variant is not None
        else (run.variant if run else ""),
        adapters=run.adapters if run else None,
    )
