"""Reuse WeSpeaker frame features within one batch, without sharing speaker masks.

Heavy optional dependencies are imported only after pyannote has been loaded.
No model, feature, or waveform is cached across batches or jobs.
"""
from __future__ import annotations

import logging
import os
import warnings
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)
SHARED_EMBEDDING_ENV = "MTD_SHARED_SPEAKER_EMBEDDING"


def _can_share(model) -> bool:
    # Training BatchNorm mixes samples; dither introduces per-sample randomness.
    return (
        not any(module.training for module in model.modules())
        and getattr(model.hparams, "dither", None) == 0.0
        and not model.fbank_only
        and callable(getattr(model, "forward_frames", None))
        and callable(getattr(model, "forward_embedding", None))
    )


class _SharedEmbedding:
    """Delegate the inference interface and keep pipeline.to() working.

    The installer adds pyannote's BaseInference marker to this adapter so that
    Pipeline.__setattr__ registers it normally, including future device moves.
    """

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def to(self, device):
        self.original.to(device)
        return self

    def __call__(self, waveforms, masks=None):
        import torch

        model = self.original.model_
        if (
            masks is None
            or not _can_share(model)
            or waveforms.ndim != 3
            or waveforms.shape[1] != 1
            or waveforms.shape[0] < 2
            or masks.ndim != 2
            or masks.shape[0] != waveforms.shape[0]
        ):
            return self.original(waveforms, masks=masks)

        indices, inverse = [], []
        for i in range(waveforms.shape[0]):
            # Never merge merely similar audio, even when masks are identical.
            if not indices or not torch.equal(waveforms[i], waveforms[indices[-1]]):
                indices.append(i)
            inverse.append(len(indices) - 1)
        if len(indices) == waveforms.shape[0]:
            return self.original(waveforms, masks=masks)

        # Match the original inference precision and warning policy. Errors must
        # reach the existing diarization error handler, not silently change models.
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames = model.forward_frames(waveforms[indices].to(self.original.device))
            expanded = frames.index_select(0, torch.tensor(inverse, device=frames.device))
            embeddings = model.forward_embedding(expanded, weights=masks.to(self.original.device))
        return embeddings.cpu().numpy()


def enable_shared_speaker_embedding(pipeline) -> bool:
    """Install a per-pipeline adapter only on the validated pyannote implementation.

    Set MTD_SHARED_SPEAKER_EMBEDDING=0 before starting the app to use the original
    path. New pyannote versions deliberately require revalidation before enabling.
    Batch shape changes can cause tiny floating-point differences on CUDA; this
    is not a promise of bitwise-identical embeddings for every recording.
    """
    if os.environ.get(SHARED_EMBEDDING_ENV, "1").strip().lower() in {"0", "false", "off", "no"}:
        return False
    original = getattr(pipeline, "_embedding", None)
    if isinstance(original, _SharedEmbedding):
        return True
    try:
        if version("pyannote.audio") != "4.0.7":
            return False
        from pyannote.audio.core.inference import BaseInference
        from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
        from pyannote.audio.pipelines.speaker_verification import PyannoteAudioPretrainedSpeakerEmbedding
    except (ImportError, PackageNotFoundError):
        return False
    if type(original) is not PyannoteAudioPretrainedSpeakerEmbedding:
        return False
    if type(original.model_) is not WeSpeakerResNet34 or not _can_share(original.model_):
        return False
    if original.device.type != "cuda":
        return False

    class SharedWeSpeakerEmbedding(_SharedEmbedding, BaseInference):
        pass

    pipeline._embedding = SharedWeSpeakerEmbedding(original)
    logger.info("WeSpeaker: sharing identical audio windows within embedding batches (%s=0 disables)", SHARED_EMBEDDING_ENV)
    return True
