"""Audio discovery, loading, and CLAP embeddings for evaluation.

Heavy imports (``soundfile``, ``torchaudio``, ``transformers``) are deferred so the
rest of the package imports without them. All of these ship with ACE-Step's
environment; ``pip install -e ".[eval]"`` adds them elsewhere.

CLAP embeds audio and text in one space, so a single model covers style distance
(audio vs audio) and prompt adherence (audio vs text). The default checkpoint,
``laion/larger_clap_music``, was trained on music.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"})
DEFAULT_CLAP = "laion/larger_clap_music"
CLAP_SAMPLE_RATE = 48_000


def list_audio(source: Path) -> list[Path]:
    """Audio files under a folder (recursive), or listed in an ACE-Step dataset JSON."""
    source = source.expanduser().resolve()
    if source.is_file() and source.suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        samples = raw if isinstance(raw, list) else raw.get("samples", [])
        paths = []
        for s in samples:
            ap = s.get("audio_path") or s.get("filename")
            if ap:
                p = Path(ap)
                paths.append(p if p.is_absolute() else source.parent / p)
        return sorted(p for p in paths if p.is_file())
    if source.is_dir():
        return sorted(
            p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
    raise FileNotFoundError(f"{source} is neither an audio folder nor a dataset JSON")


def load_mono(path: Path, sample_rate: int, max_seconds: float | None = None) -> torch.Tensor:
    """Decode ``path`` to mono float32 at ``sample_rate``, center-cropped to ``max_seconds``."""
    import soundfile as sf
    import torchaudio.functional as AF

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != sample_rate:
        wav = AF.resample(wav, sr, sample_rate)
    if max_seconds is not None:
        n = int(max_seconds * sample_rate)
        if wav.numel() > n:
            start = (wav.numel() - n) // 2
            wav = wav[start : start + n]
    return wav


class ClapEmbedder:
    """Audio and text embeddings from a Hugging Face CLAP checkpoint (L2-normalized)."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CLAP,
        device: str | torch.device = "cpu",
        max_seconds: float = 30.0,
        batch_size: int = 8,
    ) -> None:
        from transformers import ClapModel, ClapProcessor

        self.device = torch.device(device)
        self.model = ClapModel.from_pretrained(checkpoint).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(checkpoint)
        self.max_seconds = max_seconds
        self.batch_size = batch_size

    @staticmethod
    def _features(out: object) -> torch.Tensor:
        # transformers < 5 returns the projected tensor; >= 5 may wrap it in an output.
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output  # type: ignore[attr-defined]

    @torch.no_grad()
    def embed_audio(self, paths: Sequence[Path]) -> torch.Tensor:
        chunks = []
        for i in range(0, len(paths), self.batch_size):
            wavs = [
                load_mono(p, CLAP_SAMPLE_RATE, self.max_seconds).numpy()
                for p in paths[i : i + self.batch_size]
            ]
            inputs = self.processor.feature_extractor(
                wavs, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
            ).to(self.device)
            chunks.append(self._features(self.model.get_audio_features(**inputs)))
        return torch.nn.functional.normalize(torch.cat(chunks).float().cpu(), dim=1)

    @torch.no_grad()
    def embed_text(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        feats = self._features(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(feats.float().cpu(), dim=1)


def aesthetics_scores(paths: Sequence[Path]) -> list[dict[str, float]]:
    """Meta Audiobox Aesthetics (CE, CU, PC, PQ) per clip. Needs ``audiobox_aesthetics``."""
    try:
        from audiobox_aesthetics.infer import initialize_predictor
    except ImportError as exc:
        raise RuntimeError(
            "--aesthetics needs the audiobox_aesthetics package: "
            'pip install -e ".[eval-aesthetics]"'
        ) from exc
    predictor = initialize_predictor()
    return [
        {k: float(v) for k, v in row.items()}
        for row in predictor.forward([{"path": str(p)} for p in paths])
    ]
