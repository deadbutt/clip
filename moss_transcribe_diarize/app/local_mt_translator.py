from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from moss_transcribe_diarize.subtitle import SubtitleSegment

from .text_translator import translation_skip_reason


DEFAULT_OPUS_MODEL_DIR = Path("models/opus-mt-en-zh-ct2-int8")
DEFAULT_OPUS_TOKENIZER_DIR = Path("models/opus-mt-en-zh")
DEFAULT_TARGET_PREFIX = ">>cmn_Hans<<"


@dataclass(slots=True)
class LocalMtTranslator:
    model_dir: str | Path = DEFAULT_OPUS_MODEL_DIR
    tokenizer_dir: str | Path | None = DEFAULT_OPUS_TOKENIZER_DIR
    model: str = "Helsinki-NLP/opus-mt-en-zh"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 1
    target_prefix: str = DEFAULT_TARGET_PREFIX
    _translator: object | None = field(default=None, init=False, repr=False)
    _source_sp: object | None = field(default=None, init=False, repr=False)
    _target_sp: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.tokenizer_dir = Path(self.tokenizer_dir) if self.tokenizer_dir else self.model_dir

    def runtime_info(self) -> dict[str, object]:
        return {
            "available": True,
            "backend": "local-mt",
            "model": self.model,
            "model_dir": str(self.model_dir),
            "tokenizer_dir": str(self.tokenizer_dir),
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "target_prefix": self.target_prefix,
        }

    def translate_segments(
        self,
        segments: Iterable[SubtitleSegment],
        *,
        target_language: str = "简体中文",
        batch_size: int = 64,
        context_window: int = 0,
        semantic_units: bool = False,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
    ) -> list[str]:
        del target_language, context_window, semantic_units
        items = list(segments)
        self._ensure_loaded()
        batch_size = max(1, int(batch_size))
        translations: list[str] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            translations.extend(self._translate_batch(batch))
            if progress_callback is not None:
                progress_callback(min(len(translations), len(items)), len(items), start, len(batch))
        return translations

    def _ensure_loaded(self) -> None:
        if self._translator is not None and self._source_sp is not None and self._target_sp is not None:
            return
        try:
            import ctranslate2
            import sentencepiece
        except ImportError as exc:
            raise RuntimeError(
                "Local OPUS-MT translation requires ctranslate2 and sentencepiece. "
                "Run: uv pip install sentencepiece"
            ) from exc
        model_dir = Path(self.model_dir)
        tokenizer_dir = Path(self.tokenizer_dir or model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Local MT model directory does not exist: {model_dir}")
        source_spm = tokenizer_dir / "source.spm"
        target_spm = tokenizer_dir / "target.spm"
        if not source_spm.exists() or not target_spm.exists():
            raise FileNotFoundError(f"source.spm and target.spm are required in {tokenizer_dir}")
        self._translator = ctranslate2.Translator(
            str(model_dir),
            device=self.device,
            compute_type=self.compute_type,
        )
        self._source_sp = sentencepiece.SentencePieceProcessor(model_file=str(source_spm))
        self._target_sp = sentencepiece.SentencePieceProcessor(model_file=str(target_spm))

    def _translate_batch(self, segments: list[SubtitleSegment]) -> list[str]:
        assert self._translator is not None
        assert self._source_sp is not None
        assert self._target_sp is not None
        passthrough: dict[int, str] = {}
        source_tokens: list[list[str]] = []
        source_indexes: list[int] = []
        for index, segment in enumerate(segments):
            text = str(segment.text or "").strip()
            if _should_preserve_text(text):
                passthrough[index] = text
                continue
            tokens = self._source_sp.encode(text, out_type=str)
            if self.target_prefix:
                tokens = [self.target_prefix, *tokens]
            source_tokens.append(tokens)
            source_indexes.append(index)

        translated_by_index = dict(passthrough)
        if source_tokens:
            results = self._translator.translate_batch(
                source_tokens,
                beam_size=max(1, int(self.beam_size)),
                max_batch_size=max(1, len(source_tokens)),
            )
            for index, result in zip(source_indexes, results):
                hypothesis = result.hypotheses[0] if result.hypotheses else []
                translated_by_index[index] = self._decode(hypothesis)
        return [translated_by_index.get(index, str(segment.text or "")) for index, segment in enumerate(segments)]

    def _decode(self, tokens: list[str]) -> str:
        assert self._target_sp is not None
        filtered = [
            token
            for token in tokens
            if token
            and token not in {"</s>", "<pad>"}
            and not (token.startswith(">>") and token.endswith("<<"))
        ]
        return self._target_sp.decode(filtered).strip()


def _should_preserve_text(text: str) -> bool:
    return translation_skip_reason(text) is not None
