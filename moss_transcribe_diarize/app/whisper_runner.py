from __future__ import annotations

import threading
import time
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from moss_transcribe_diarize.defaults import DEFAULT_PROMPT

from .ffmpeg import detect_ffmpeg

StatusCallback = Callable[[str, float | None, int | None], None]

_PART_HEAD_RE = re.compile(r"^\[(\d+(?:\.\d+)?)\]")
_PART_TAIL_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]$")


def _shift_part_times(part: str, offset: float) -> str:
    """把 part（形如 "[start][S00]text[end]"）的首尾时间戳平移 offset 秒。"""
    head = _PART_HEAD_RE.match(part)
    if head:
        part = f"[{float(head.group(1)) + offset:.2f}]" + part[head.end():]
    tail = _PART_TAIL_RE.search(part)
    if tail:
        part = part[: tail.start()] + f"[{float(tail.group(1)) + offset:.2f}]"
    return part


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    prompt_len: int
    generated_tokens: int
    elapsed_sec: float
    model: str
    audio: str
    decoding: str
    temperature: float | None
    top_p: float | None = None
    top_k: int | None = None
    # 词级时间戳 [(start, end, token), ...]，仅在转录内部使用、不序列化；
    # 供词级断句重组取精确时间。引擎不支持时为 None，走 segment 级降级。
    words: list[tuple[float, float, str]] | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "prompt_len": self.prompt_len,
            "generated_tokens": self.generated_tokens,
            "elapsed_sec": self.elapsed_sec,
            "model": self.model,
            "audio": self.audio,
            "decoding": self.decoding,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


class WhisperRunner:
    """Fast, practical transcription runner with a Whisper fallback chain."""

    def __init__(
        self,
        model_path: str | Path = "medium",
        *,
        device: str = "auto",
        dtype: str = "auto",
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        condition_on_previous_text: bool = True,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 3,
        # VAD 调优参数：默认 None 用 faster-whisper 原生默认（min_silence 2000/pad 400/threshold 0.5）。
        # 实测调小会经 condition_on_previous_text 连锁引入撇号缺失/个别错字（如 don't→don d），得不偿失，
        # 故默认不开启；需要时按视频显式传参。
        vad_min_silence_duration_ms: int | None = None,
        vad_speech_pad_ms: int | None = None,
        vad_threshold: float | None = None,
        # 静音超过该阈值时跳过幻觉输出（whisper 在长静音易生成 "Thank you./Bye."），纯增益无副作用
        hallucination_silence_threshold: float = 2.0,
    ):
        self.model_path = str(model_path)
        self.device_name = device
        self.dtype_name = dtype
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self.vad_threshold = vad_threshold
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self._model = None
        self._engine = None
        self._openai_fp16 = False
        self._fallback_errors: list[str] = []
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def runtime_info(self) -> dict:
        return {
            "backend": "whisper",
            "path": self.model_path,
            "engine": self._engine or "unloaded",
            "device": self.device_name,
            "dtype": self.dtype_name,
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "vad_min_silence_duration_ms": self.vad_min_silence_duration_ms,
            "vad_speech_pad_ms": self.vad_speech_pad_ms,
            "vad_threshold": self.vad_threshold,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "fallback_errors": list(self._fallback_errors),
        }

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_length: int = 0,
        max_new_tokens: int = 0,
        decoding: str = "greedy",
        temperature: float | None = None,
        hotwords: str | None = None,
        status_callback: StatusCallback | None = None,
        **_: object,
    ) -> TranscriptionResult:
        del max_length, max_new_tokens, decoding, temperature
        with self._lock:
            if status_callback is not None:
                status_callback("loading_model", 0.05, None)
            self._ensure_loaded()
            if status_callback is not None:
                status_callback("transcribing", 0.10, None)

            started = time.time()
            if self._engine == "faster-whisper":
                parts, segment_count, words = self._transcribe_faster_whisper(
                    audio_path, prompt, status_callback, hotwords=hotwords
                )
                # 即使 vad_filter=False，no_speech_threshold 也可能把段落判为无声而提前结束，
                # 所以这里不再以 vad_filter 为前提，只看 _looks_sparse 的判定结果（含覆盖率检查）
                if self._looks_sparse(parts, segment_count, audio_path):
                    fallback_parts, fallback_segment_count, fallback_words = self._transcribe_faster_whisper(
                        audio_path,
                        prompt,
                        status_callback,
                        vad_filter=False,
                        hotwords=hotwords,
                    )
                    if len("".join(fallback_parts).strip()) > len("".join(parts).strip()):
                        parts = fallback_parts
                        segment_count = fallback_segment_count
                        words = fallback_words
                # 定向缺口恢复：VAD 可能局部误杀语音（如短句被判无声而跳过），
                # 在词时间轴上找出 >3s 的无词缺口，检测该区段音频能量，
                # 若有语音特征则单独重转录（vad_filter=False）并补回词表。
                recovered = self._recover_gaps(words, audio_path, prompt, status_callback)
                if recovered:
                    parts.extend(recovered[0])
                    words.extend(recovered[1])
                    segment_count += len(recovered[0])
                    words.sort(key=lambda w: w[0])
                    parts.sort(key=lambda p: float(p[1:p.find("]")]))
            else:
                parts, segment_count, words = self._transcribe_openai_whisper(audio_path, prompt, status_callback)

            if status_callback is not None:
                status_callback("transcribing", 0.85, segment_count)
            return TranscriptionResult(
                text="".join(parts),
                prompt_len=0,
                generated_tokens=segment_count,
                elapsed_sec=time.time() - started,
                model=self.model_path,
                audio=str(Path(audio_path).expanduser()),
                decoding="beam_search",
                temperature=None,
                words=words,
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _ensure_ffmpeg_on_path()
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        errors: list[str] = []
        self._fallback_errors = []
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            errors.append(f"faster-whisper unavailable: {exc}")
            self._fallback_errors = list(errors)
        else:
            try:
                device, compute_type = _resolve_runtime(self.device_name, self.dtype_name)
                self._model = WhisperModel(self.model_path, device=device, compute_type=compute_type)
                self._engine = "faster-whisper"
                return
            except Exception as exc:
                errors.append(f"faster-whisper load failed: {exc}")
                self._fallback_errors = list(errors)

        try:
            import whisper as openai_whisper
        except ImportError as exc:
            errors.append(f"openai-whisper unavailable: {exc}")
        else:
            try:
                device, fp16 = _resolve_openai_runtime(self.device_name, self.dtype_name)
                self._model = openai_whisper.load_model(self.model_path, device=device)
                self._engine = "openai-whisper"
                self._openai_fp16 = fp16
                if self._fallback_errors:
                    print(
                        "[WARN] faster-whisper was not used; falling back to openai-whisper: "
                        + " | ".join(self._fallback_errors),
                        flush=True,
                    )
                return
            except Exception as exc:
                errors.append(f"openai-whisper load failed: {exc}")

        raise RuntimeError("Whisper model load failed: " + " | ".join(errors))

    def _transcribe_faster_whisper(
        self,
        audio_path: str | Path,
        prompt: str,
        status_callback: StatusCallback | None,
        *,
        vad_filter: bool | None = None,
        hotwords: str | None = None,
    ) -> tuple[list[str], int, list[tuple[float, float, str]]]:
        path = str(Path(audio_path).expanduser())
        kwargs = {
            "language": self.language,
            "beam_size": int(self.beam_size),
            "vad_filter": bool(self.vad_filter if vad_filter is None else vad_filter),
            "initial_prompt": _whisper_initial_prompt(prompt, self.language),
            "condition_on_previous_text": bool(self.condition_on_previous_text),
            "repetition_penalty": float(self.repetition_penalty),
            "no_repeat_ngram_size": int(self.no_repeat_ngram_size),
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "word_timestamps": True,
            # 静音超过该阈值时跳过幻觉输出（whisper 在长静音易生成 "Thank you./Bye."），纯增益无副作用
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
        }
        # 热词表（人名/专名/领域术语，空格分隔）: whisper 框架内唯一能针对性救错听的杠杆
        if hotwords and hotwords.strip():
            kwargs["hotwords"] = hotwords.strip()
        # VAD 调优：仅在显式传参时覆盖 faster-whisper 默认（默认 None 保持原生行为，
        # 实测调小会经 condition_on_previous_text 连锁引入撇号缺失/个别错字）
        _vad_params: dict[str, object] = {}
        if self.vad_min_silence_duration_ms is not None:
            _vad_params["min_silence_duration_ms"] = self.vad_min_silence_duration_ms
        if self.vad_speech_pad_ms is not None:
            _vad_params["speech_pad_ms"] = self.vad_speech_pad_ms
        if self.vad_threshold is not None:
            _vad_params["threshold"] = self.vad_threshold
        if _vad_params:
            kwargs["vad_parameters"] = _vad_params
        # fallback 调用（vad_filter=False 被显式传入）时，放宽 no_speech_threshold，
        # 避免再次因为"判无声"提前结束 segment 迭代
        if vad_filter is False:
            kwargs["no_speech_threshold"] = 0.9
            kwargs["log_prob_threshold"] = -1.5
        try:
            segments_iter, info = self._model.transcribe(path, **kwargs)
        except TypeError as exc:
            if not _looks_like_unsupported_transcribe_option(exc):
                raise
            for key in (
                "condition_on_previous_text",
                "repetition_penalty",
                "no_repeat_ngram_size",
                "compression_ratio_threshold",
                "log_prob_threshold",
                "no_speech_threshold",
                "word_timestamps",
                "hotwords",
            ):
                kwargs.pop(key, None)
            segments_iter, info = self._model.transcribe(path, **kwargs)
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        repeat_guard = _RepeatedPhraseGuard()
        parts: list[str] = []
        words: list[tuple[float, float, str]] = []
        segment_count = 0
        for segment in segments_iter:
            segment_count += 1
            start = max(0.0, float(segment.start))
            end = max(start, float(segment.end))
            text = str(segment.text or "").strip()
            if not text:
                continue
            if repeat_guard.should_skip(text):
                if status_callback is not None:
                    progress = _duration_progress(end, duration)
                    status_callback("transcribing", progress, segment_count)
                continue
            parts.append(f"[{start:.2f}][S00]{text}[{end:.2f}]")
            # 复读段落的词也一并跳过，只收集保留段落的词时间戳
            for word in getattr(segment, "words", None) or []:
                token = str(getattr(word, "word", "") or "")
                if not token.strip():
                    continue
                words.append((max(0.0, float(word.start)), max(0.0, float(word.end)), token))
            if status_callback is not None:
                progress = _duration_progress(end, duration)
                status_callback("transcribing", progress, segment_count)
        return parts, segment_count, words

    def _looks_sparse(self, parts: list[str], segment_count: int, audio_path: str | Path) -> bool:
        text = "".join(parts).strip()
        duration = _probe_duration(audio_path)
        # 新增：基于"转录结束时间占音频总时长"判断是否被 VAD 提前截断
        # 长音频场景下，即使前半段转录出大量内容，VAD 仍可能把后半段判为无声而跳过，
        # 导致整段尾部丢失。只要覆盖时间不足 80%，就认为 sparse，触发无 VAD 重试。
        if duration >= 12.0:
            last_end = 0.0
            for chunk in parts:
                # parts 形如 "[start][S00]text[end]"，找最后一个 [end]
                idx = chunk.rfind("]")
                if idx <= 0:
                    continue
                head = chunk.rfind("[", 0, idx)
                if head <= 0:
                    continue
                try:
                    last_end = max(last_end, float(chunk[head + 1:idx]))
                except ValueError:
                    continue
            if last_end > 0 and duration > 0 and (last_end / duration) < 0.8:
                return True
        if len(text) >= 40:
            return False
        if segment_count > 2:
            return False
        return duration >= 12.0

    def _recover_gaps(
        self,
        words: list[tuple[float, float, str]],
        audio_path: str | Path,
        prompt: str,
        status_callback: StatusCallback | None,
    ) -> tuple[list[str], list[tuple[float, float, str]]] | None:
        """检测词时间轴上的无词缺口，对有语音能量的缺口单独重转录。

        VAD 可能局部误杀短句（如 "Whoa!" 被判无声），整体覆盖率检查无法发现。
        这里在词列表中找 >3s 的缺口，用 RMS + 自相关判断该区段是否有语音，
        有则提取该片段用 vad_filter=False 重转录，把词和文本补回。
        """
        if len(words) < 2:
            return None
        gap_threshold = 3.0
        gaps: list[tuple[float, float]] = []
        for i in range(1, len(words)):
            prev_end = words[i - 1][1]
            cur_start = words[i][0]
            gap = cur_start - prev_end
            if gap >= gap_threshold:
                gaps.append((prev_end, cur_start))
        if not gaps:
            return None
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            return None
        try:
            audio, sr = sf.read(str(Path(audio_path).expanduser()))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception:
            # 视频容器（mp4/mkv/webm…）libsndfile 读不了，此时缺口恢复会整体静默失效；
            # 改用 ffmpeg 解码成 16k 单声道 wav 再读。
            decoded = _decode_audio_pcm(audio_path)
            if decoded is None:
                return None
            audio, sr = decoded
        overall_rms = float(np.sqrt(np.mean(audio ** 2)))
        if overall_rms < 1e-6:
            return None
        recovered_parts: list[str] = []
        recovered_words: list[tuple[float, float, str]] = []
        for gap_start, gap_end in gaps:
            s = int(gap_start * sr)
            e = int(gap_end * sr)
            if e - s < int(0.5 * sr):
                continue
            chunk = audio[s:e].astype(np.float32)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < overall_rms * 0.08:
                continue
            voiced = 0
            n = 0
            for j in range(0, len(chunk) - 640, 320):
                fr = chunk[j:j + 640].astype(np.float64)
                fr = fr - fr.mean()
                if float(np.abs(fr).mean()) < 0.004:
                    continue
                n += 1
                c = np.correlate(fr, fr, "full")[640:]
                c /= (c[0] + 1e-9)
                k = int(np.argmax(c[40:230])) + 40
                if c[k] > 0.3:
                    voiced += 1
            if n == 0 or voiced / n < 0.2:
                continue
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, chunk, sr)
                gap_parts, _, gap_words = self._transcribe_faster_whisper(
                    tmp_path, prompt, status_callback, vad_filter=False,
                )
                if gap_words:
                    offset = gap_start
                    adjusted_words = [
                        (w[0] + offset, w[1] + offset, w[2]) for w in gap_words
                    ]
                    recovered_words.extend(adjusted_words)
                    # parts 的时间戳是相对临时片段的(≈0),必须同样加偏移,
                    # 否则 transcribe() 按首时间戳排序时这些段会被排到全文最前。
                    recovered_parts.extend(
                        _shift_part_times(part, offset) for part in gap_parts
                    )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        if not recovered_words:
            return None
        return recovered_parts, recovered_words

    def _transcribe_openai_whisper(
        self,
        audio_path: str | Path,
        prompt: str,
        status_callback: StatusCallback | None,
    ) -> tuple[list[str], int, list[tuple[float, float, str]]]:
        result = self._model.transcribe(
            str(Path(audio_path).expanduser()),
            language=self.language,
            fp16=bool(self._openai_fp16),
            initial_prompt=_whisper_initial_prompt(prompt, self.language),
            verbose=False,
            condition_on_previous_text=bool(self.condition_on_previous_text),
            beam_size=int(self.beam_size),
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            word_timestamps=True,
        )
        duration = float(result.get("duration") or 0.0)
        segments = result.get("segments") or []
        repeat_guard = _RepeatedPhraseGuard()
        parts: list[str] = []
        words: list[tuple[float, float, str]] = []
        segment_count = 0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_count += 1
            start = max(0.0, float(segment.get("start") or 0.0))
            end = max(start, float(segment.get("end") or start))
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            if repeat_guard.should_skip(text):
                if status_callback is not None:
                    progress = _duration_progress(end, duration)
                    status_callback("transcribing", progress, segment_count)
                continue
            parts.append(f"[{start:.2f}][S00]{text}[{end:.2f}]")
            for word in segment.get("words") or []:
                token = str(word.get("word") or "")
                if not token.strip():
                    continue
                words.append(
                    (
                        max(0.0, float(word.get("start") or 0.0)),
                        max(0.0, float(word.get("end") or 0.0)),
                        token,
                    )
                )
            if status_callback is not None:
                progress = _duration_progress(end, duration)
                status_callback("transcribing", progress, segment_count)
        return parts, segment_count, words


def _resolve_runtime(device: str, dtype: str) -> tuple[str, str]:
    device = (device or "auto").lower()
    dtype = (dtype or "auto").lower()
    if dtype in {"auto", "bf16", "bfloat16"} and device == "auto":
        dtype = "default"
    elif dtype == "auto":
        dtype = "float16" if device == "cuda" else "int8"
    if dtype == "bf16":
        dtype = "float16"
    if dtype == "bfloat16":
        dtype = "float16"
    if dtype == "fp16":
        dtype = "float16"
    if dtype == "fp32":
        dtype = "float32"
    return device, dtype


def _resolve_openai_runtime(device: str, dtype: str) -> tuple[str, bool]:
    try:
        import torch
    except ImportError:
        torch = None
    device = (device or "auto").lower()
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    dtype = (dtype or "auto").lower()
    fp16 = device == "cuda" and dtype not in {"fp32", "float32"}
    return device, fp16


def _whisper_initial_prompt(prompt: str, language: str | None) -> str | None:
    custom = (prompt or "").strip()
    if custom and custom != DEFAULT_PROMPT:
        return custom
    # 无标点的转录输出会让断句后处理失效（检测不到句末标点就合不成完整句），
    # 用一段带标点的示例文本引导模型稳定输出标点和大小写。
    # 实测（runs/regroup_compare）：英文音频句末标点覆盖率 12% -> 45%。
    lang = (language or "").lower()
    if lang.startswith("zh"):
        return _PUNCT_PROMPT_ZH
    if not lang or lang.startswith("en"):
        return _PUNCT_PROMPT_EN
    return None


_PUNCT_PROMPT_EN = (
    "Here is a properly punctuated transcript. She said: \"I made this decision last year. "
    "It was hard, but it was right.\" We talked for a while, and then she left. I agree."
)

_PUNCT_PROMPT_ZH = (
    "以下是带标点的转录文本。她说：“这是我去年做的决定，虽然很难，但是对的。”"
    "我们聊了一会儿，然后她离开了。我觉得很有道理。"
)


class _RepeatedPhraseGuard:
    def __init__(self, *, max_consecutive: int = 3, max_total: int = 24):
        self.max_consecutive = max_consecutive
        self.max_total = max_total
        self._last = ""
        self._consecutive = 0
        self._totals: dict[str, int] = {}

    def should_skip(self, text: str) -> bool:
        normalized = _normalize_repeated_phrase(text)
        if not _is_repeated_phrase_candidate(normalized):
            self._last = ""
            self._consecutive = 0
            return False

        if normalized == self._last:
            self._consecutive += 1
        else:
            self._last = normalized
            self._consecutive = 1

        total = self._totals.get(normalized, 0) + 1
        self._totals[normalized] = total
        return self._consecutive > self.max_consecutive or total > self.max_total


def _normalize_repeated_phrase(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_repeated_phrase_candidate(normalized: str) -> bool:
    if not normalized or len(normalized) > 64:
        return False
    words = normalized.split()
    return 1 <= len(words) <= 6


def _looks_like_unsupported_transcribe_option(exc: TypeError) -> bool:
    message = str(exc).lower()
    return "unexpected keyword" in message or "got an unexpected" in message


def _ensure_ffmpeg_on_path() -> None:
    tools = detect_ffmpeg()
    ffmpeg_path = tools.ffmpeg or tools.ffprobe
    if not ffmpeg_path:
        return
    ffmpeg_dir = str(Path(ffmpeg_path).resolve().parent)
    current = os.environ.get("PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if ffmpeg_dir in parts:
        return
    os.environ["PATH"] = os.pathsep.join([ffmpeg_dir, *parts]) if parts else ffmpeg_dir


def _duration_progress(position: float, duration: float) -> float:
    if duration <= 0:
        return 0.25
    ratio = max(0.0, min(1.0, position / duration))
    return 0.10 + (0.85 - 0.10) * ratio


def _decode_audio_pcm(audio_path: str | Path) -> tuple["np.ndarray", int] | None:
    """用 ffmpeg 把（视频容器的）媒体解成 16k 单声道波形，供缺口检测使用。

    返回 (waveform, sample_rate)；ffmpeg 不可用或解码失败返回 None。
    """
    import subprocess
    import tempfile

    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    try:
        ffmpeg = detect_ffmpeg().ffmpeg
    except Exception:
        return None
    if not ffmpeg:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(Path(audio_path).expanduser()),
             "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", tmp_path, "-y"],
            capture_output=True, timeout=600,
        )
        if result.returncode != 0:
            return None
        audio, sr = sf.read(tmp_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio), int(sr)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _probe_duration(audio_path: str | Path) -> float:
    try:
        from .ffmpeg import probe_media_duration
    except Exception:
        return 0.0
    try:
        return float(probe_media_duration(audio_path) or 0.0)
    except Exception:
        return 0.0
