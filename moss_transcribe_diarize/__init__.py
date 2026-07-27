from __future__ import annotations

from .subtitle import (
    SubtitleSegment,
    SubtitleStyle,
    coerce_subtitle_segments,
    export_ass,
    export_json,
    export_srt,
    normalize_segments,
    subtitle_segments_from_transcript,
)
from .transcript_parser import (
    TranscriptParseError,
    TranscriptSegment,
    TranscriptStreamParser,
    iter_transcript_segments,
    parse_transcript,
)


_MOSS_EXPORTS = {
    "MossTranscribeDiarizeConfig": ("configuration_moss_transcribe_diarize", "MossTranscribeDiarizeConfig"),
    "MossTranscribeDiarizeForConditionalGeneration": (
        "modeling_moss_transcribe_diarize",
        "MossTranscribeDiarizeForConditionalGeneration",
    ),
    "MossTranscribeDiarizeModel": ("modeling_moss_transcribe_diarize", "MossTranscribeDiarizeModel"),
    "MossTranscribeDiarizePreTrainedModel": (
        "modeling_moss_transcribe_diarize",
        "MossTranscribeDiarizePreTrainedModel",
    ),
    "MossTranscribeDiarizeProcessor": ("processing_moss_transcribe_diarize", "MossTranscribeDiarizeProcessor"),
    "VQAdaptor": ("modeling_moss_transcribe_diarize", "VQAdaptor"),
}


def __getattr__(name: str):
    if name not in _MOSS_EXPORTS:
        raise AttributeError(name)
    module_name, attr = _MOSS_EXPORTS[name]
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[attr])
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the legacy MOSS/Transformers runtime. Install the torch-runtime extra "
            "or use the default Whisper backend."
        ) from exc
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = [
    "SubtitleSegment",
    "SubtitleStyle",
    "TranscriptParseError",
    "TranscriptSegment",
    "TranscriptStreamParser",
    "MossTranscribeDiarizeConfig",
    "MossTranscribeDiarizeForConditionalGeneration",
    "MossTranscribeDiarizeModel",
    "MossTranscribeDiarizePreTrainedModel",
    "MossTranscribeDiarizeProcessor",
    "VQAdaptor",
    "coerce_subtitle_segments",
    "export_ass",
    "export_json",
    "export_srt",
    "iter_transcript_segments",
    "normalize_segments",
    "parse_transcript",
    "subtitle_segments_from_transcript",
]
