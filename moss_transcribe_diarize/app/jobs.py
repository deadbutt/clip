from __future__ import annotations

import difflib
import json
import logging
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from moss_transcribe_diarize.subtitle import (
    SubtitleItem,
    SubtitleSegment,
    SubtitleStyle,
    clean_source_captions,
    coerce_subtitle_items,
    coerce_subtitle_segments,
    drop_repeated_hallucinations,
    export_ass,
    export_json,
    export_srt,
    parse_ass,
    parse_srt,
    regroup_sentences,
    regroup_sentences_from_words,
    subtitle_segments_from_transcript,
    write_text,
)

from . import vocal_separator
from .audio_timeline import (
    HOLE_FIX_THRESHOLD,
    analyze_audio_timeline,
    extract_timeline_fixed_audio,
    total_hole_seconds,
)
from .clips import generate_clip_candidates, rebase_segments_for_clip
from .ffmpeg import (
    FFmpegCancelled,
    burn_ass_subtitles,
    burn_ass_subtitles_clip,
    detect_ffmpeg,
    probe_media_duration,
    probe_video_size,
)
from .speaker_labeler import label_speakers
from .text_search import apply_replacements, search_segment_texts
from .text_translator import (
    apply_translations,
    collect_pretranslation_skips,
    validate_translation_outputs,
)

RUNNING_STATES = {"queued", "downloading", "loading_model", "transcribing", "postprocessing", "labeling_speakers", "translating", "proofreading", "rendering"}
TERMINAL_STATES = {"waiting_review", "done", "failed", "cancelled"}
# API 触发的用户后处理状态: {status: (进度信息字段, 中文名)}。重启中断时回
# waiting_review 而非重新入队,避免 worker 重跑后处理覆盖用户编辑。
_POSTPROCESS_STATES: dict[str, tuple[str | None, str]] = {
    "translating": ("translation_info", "翻译"),
    "proofreading": ("proofread_info", "校对"),
    "rendering": (None, "烧录"),
}

logger = logging.getLogger(__name__)

# 词级 diff 对齐的最低匹配率:低于它说明文本被重写,插值时间戳没有意义。
_MIN_ALIGN_MATCH_RATIO = 0.4
# segments 解析缓存条目上限:每条可达数 MB(词级时间戳),不设限的话
# 长期运行的服务会随历史任务数无限增长;8 个活跃编辑任务已绰绰有余。
_SEGMENTS_CACHE_MAX = 8

# 字幕展示/导出只修正 Whisper 词级时间戳偏早的问题，工程 JSON 仍保留原始边界。
_TAIL_FRAME_SECONDS = 0.02
_TAIL_ANALYSIS_RATE = 8000
_TAIL_MAX_EXTENSION = 0.80
_TAIL_FALLBACK_PADDING = 0.50


class _AppendFileHandler(logging.Handler):
    """每条日志即开即写即关。

    常驻 FileHandler 会在 Windows 上一直锁住 job 目录（删除/清理时
    PermissionError）；本任务日志频率很低（状态迁移、批次进度），
    逐条开关文件的代价可以忽略。"""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def _normalize_words(text: str) -> list[str]:
    return [word.strip(".,!?;:\"'()[]") for word in str(text or "").split() if word.strip(".,!?;:\"'()[]")]


def _interpolate_item(text: str, start: float, end: float) -> SubtitleItem:
    return SubtitleItem(text=text, start=start, end=max(end, start + 0.01))


def align_items_to_text(
    old_text: str,
    old_items: list[SubtitleItem],
    new_text: str,
    *,
    min_match_ratio: float = _MIN_ALIGN_MATCH_RATIO,
) -> list[SubtitleItem] | None:
    """把旧词级 items 演化成新文本的 items(text 被编辑后调用)。

    规则: 相同词继承时间戳;替换区间均分被替换词的总时长;
    插入词在前后邻居时间之间插值;删除词丢弃。匹配率过低(整句重写)返回 None,
    让调用方诚实降级为无词级数据,而不是编造时间戳。
    """
    # items 词文本(带标点)归一化后与 items 建立索引映射。
    old_words: list[str] = []
    old_item_index: list[int] = []
    for idx, item in enumerate(old_items):
        for word in _normalize_words(item.text):
            old_words.append(word)
            old_item_index.append(idx)
    new_words = _normalize_words(new_text)
    if not old_words or not new_words:
        return None

    matcher = difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    matched = sum(size for _, _, size in matcher.get_matching_blocks())
    if matched / max(len(old_words), len(new_words)) < min_match_ratio:
        return None

    def item_at(word_pos: int) -> SubtitleItem | None:
        if 0 <= word_pos < len(old_item_index):
            return old_items[old_item_index[word_pos]]
        return None

    out: list[SubtitleItem] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                item = item_at(i1 + k) or old_items[0]
                out.append(SubtitleItem(text=new_words[j1 + k], start=item.start, end=item.end))
        elif tag in ("replace", "insert"):
            if tag == "replace" and i1 < len(old_item_index) and i2 - 1 < len(old_item_index):
                seg_start = old_items[old_item_index[i1]].start
                seg_end = old_items[old_item_index[i2 - 1]].end
            else:
                # 插入: 前驱末尾与后继开头之间插值。
                prev_end = out[-1].end if out else (item_at(0).start if item_at(0) else 0.0)
                nxt = item_at(i1) or (item_at(min(i1, len(old_item_index) - 1)) if old_item_index else None)
                next_start = nxt.start if nxt else prev_end
                seg_start, seg_end = prev_end, max(next_start, prev_end + 0.01)
            span = max(seg_end - seg_start, 0.01)
            count = j2 - j1
            for k in range(count):
                s = seg_start + span * k / count
                e = seg_start + span * (k + 1) / count
                out.append(_interpolate_item(new_words[j1 + k], s, e))
        # delete: 丢弃旧词。
    return out


class JobCancelled(BaseException):
    """任务被用户取消。

    继承 BaseException 而非 RuntimeError：管线里大量 `except Exception` /
    `except RuntimeError` 是刻意的容错/重试设计（如 LLM 批次重试），
    取消必须穿透它们直达各阶段入口的 `except JobCancelled`。"""
    pass

# Audio consumes ~12.5 prompt tokens/sec (Whisper 30s -> 375 tokens after 4x merge).
# Transcript output is ~10 generated tokens/sec of speech; use 14 with a safety margin
# so a single pass is unlikely to hit the limit and force a costly re-run.
_AUDIO_TOKENS_PER_SECOND = 12.5
_OUTPUT_TOKENS_PER_SECOND = 14.0
_MIN_MAX_NEW_TOKENS = 2048
_MAX_MAX_NEW_TOKENS = 65536
_TOKEN_ROUNDING = 512


def _apply_transcription_quality(segments: list[SubtitleSegment], metrics: list[dict[str, float]] | None) -> None:
    """Attach explainable, segment-level ASR quality metadata without extra inference."""
    if not metrics:
        return
    for segment in segments:
        overlaps = [m for m in metrics if float(m.get("end", 0)) > segment.start and float(m.get("start", 0)) < segment.end]
        if not overlaps:
            continue
        avg_logprob = sum(float(m.get("avg_logprob", 0)) for m in overlaps) / len(overlaps)
        no_speech = max(float(m.get("no_speech_prob", 0)) for m in overlaps)
        compression = max(float(m.get("compression_ratio", 0)) for m in overlaps)
        score = max(0.0, min(1.0, (avg_logprob + 1.5) / 1.5))
        score *= max(0.0, 1.0 - no_speech)
        flags: list[str] = []
        reasons: list[str] = []
        if avg_logprob < -0.8:
            flags.append("low_confidence")
            reasons.append("ASR 平均对数概率较低")
        if no_speech >= 0.6:
            flags.append("possible_hallucination")
            reasons.append("无语音概率较高")
        if compression >= 2.4:
            flags.append("possible_repetition")
            reasons.append("文本压缩比异常")
        segment.confidence = round(score, 4)
        segment.quality_flags = flags or None
        segment.quality_reasons = reasons or None


def _padded_export_segments(
    segments: list[SubtitleSegment],
    padding: float = _TAIL_FALLBACK_PADDING,
    end_overrides: dict[str, float] | None = None,
) -> list[SubtitleSegment]:
    """Apply sentence-tail buffers while keeping segment source times untouched."""
    out: list[SubtitleSegment] = []
    for index, segment in enumerate(segments):
        next_start = segments[index + 1].start if index + 1 < len(segments) else segment.end + padding
        requested_end = (end_overrides or {}).get(segment.id, segment.end + padding)
        end = min(next_start, max(segment.end, requested_end))
        out.append(replace(segment, end=max(segment.start + 0.01, end)))
    return out


def _rms_frames_from_audio(path: str | Path) -> list[float] | None:
    """Decode a small mono PCM stream and return 20 ms RMS values.

    The stream is intentionally low rate and short-lived: a 1 hour file is only
    about 58 MB of raw PCM, and normal runs are much smaller. Failures simply
    disable adaptive padding and leave the fixed fallback in place.
    """
    tools = detect_ffmpeg()
    if not tools.ffmpeg or not Path(path).is_file():
        return None
    try:
        completed = subprocess.run(
            [
                tools.ffmpeg,
                "-v",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(_TAIL_ANALYSIS_RATE),
                "-f",
                "s16le",
                "-",
            ],
            capture_output=True,
            timeout=90,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = completed.stdout
    samples_per_frame = max(1, round(_TAIL_ANALYSIS_RATE * _TAIL_FRAME_SECONDS))
    bytes_per_frame = samples_per_frame * 2
    frames: list[float] = []
    for offset in range(0, len(raw) - 1, bytes_per_frame):
        chunk = raw[offset : offset + bytes_per_frame]
        count = len(chunk) // 2
        if not count:
            continue
        values = struct.unpack(f"<{count}h", chunk[: count * 2])
        frames.append((sum(value * value for value in values) / count) ** 0.5 / 32768.0)
    return frames or None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _adaptive_tail_endings(
    path: str | Path,
    segments: list[SubtitleSegment],
    *,
    max_extension: float = _TAIL_MAX_EXTENSION,
) -> dict[str, float] | None:
    """Find safe sentence-tail extensions from low-rate audio energy.

    A tail must contain consecutive energy above both the local noise floor and
    a fraction of the segment's preceding energy. This rejects long quiet gaps
    and most music/game-only tails while preserving a conservative fallback.
    """
    frames = _rms_frames_from_audio(path)
    if not frames:
        return None
    frame_seconds = _TAIL_FRAME_SECONDS
    noise = _percentile(frames, 0.20)
    result: dict[str, float] = {}
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    for index, segment in enumerate(ordered):
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else end + max_extension
        limit = min(end + max_extension, float(next_start))
        if limit <= end + frame_seconds * 0.5:
            result[segment.id] = min(limit, end + _TAIL_FALLBACK_PADDING)
            continue
        before_start = max(start, end - 0.40)
        before = frames[round(before_start / frame_seconds) : round(end / frame_seconds)]
        reference = _percentile(before, 0.75)
        threshold = max(0.008, noise * 1.8, reference * 0.22)
        first_frame = max(0, int(end / frame_seconds))
        last_frame = min(len(frames), int(limit / frame_seconds + 0.999))
        last_active = -1
        quiet_run = 0
        for frame_index in range(first_frame, last_frame):
            if frames[frame_index] >= threshold:
                last_active = frame_index
                quiet_run = 0
            else:
                quiet_run += 1
                if last_active >= 0 and quiet_run >= 2:
                    break
        if last_active < first_frame:
            extension = _TAIL_FALLBACK_PADDING
        else:
            extension = max(0.0, min(max_extension, (last_active + 1) * frame_seconds - end))
            # Adaptive analysis may refine a longer tail, but never shorten the
            # established 0.5 s fallback that users already see in the preview.
            extension = max(_TAIL_FALLBACK_PADDING, extension)
        result[segment.id] = min(limit, end + extension)
    return result

# demucs(~2GB) + whisper(~3GB) 同卡并行所需的最小空闲显存。
# 低于它强行并行会触发驱动 sysmem fallback，两负载超线性互拖。
_PARALLEL_DEMUCS_MIN_FREE_VRAM = 5_000_000_000


def _vram_allows_parallel_demucs() -> bool:
    """空闲显存是否够 demucs 与 whisper 安全并行。无法判断时保守串行。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free, _total = torch.cuda.mem_get_info()
        return free >= _PARALLEL_DEMUCS_MIN_FREE_VRAM
    except Exception:
        return False


def recommend_max_new_tokens(
    duration_sec: float | None,
    *,
    max_length: int = 131072,
) -> int | None:
    """Suggest a non-truncating ``max_new_tokens`` for an audio of the given duration.

    Returns ``None`` when the duration is unknown, leaving any caller-provided
    value untouched. The result is clamped to a safe range and rounded so that
    ``prompt + output`` stays within ``max_length``.
    """
    if duration_sec is None or duration_sec <= 0:
        return None
    raw = duration_sec * _OUTPUT_TOKENS_PER_SECOND
    recommended = max(_MIN_MAX_NEW_TOKENS, int((raw + _TOKEN_ROUNDING - 1) // _TOKEN_ROUNDING) * _TOKEN_ROUNDING)
    # Reserve room for the audio prompt so prompt + output fits the context window.
    prompt_estimate = int(duration_sec * _AUDIO_TOKENS_PER_SECOND)
    context_ceiling = max(_MIN_MAX_NEW_TOKENS, max_length - prompt_estimate - _TOKEN_ROUNDING)
    recommended = min(recommended, context_ceiling, _MAX_MAX_NEW_TOKENS)
    return max(_MIN_MAX_NEW_TOKENS, recommended)


@dataclass(slots=True)
class JobRecord:
    id: str
    status: str
    media_name: str
    input_path: str
    job_dir: str
    inference_prompt: str
    max_length: int
    max_new_tokens: int
    decoding: str
    temperature: float | None
    progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model: str | None = None
    prompt_len: int | None = None
    generated_tokens: int | None = None
    elapsed_sec: float | None = None
    subtitle_style: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    speaker_labeling: dict[str, Any] = field(default_factory=dict)
    speaker_count: int | None = None
    diarization_backend: str = "none"
    translation_info: dict[str, Any] = field(default_factory=dict)
    proofread_info: dict[str, Any] = field(default_factory=dict)
    alignment_info: dict[str, Any] = field(default_factory=dict)
    hotwords: str = ""
    source: str = "upload"
    source_url: str | None = None
    cookies_config: dict[str, Any] = field(default_factory=dict)
    download_info: dict[str, Any] = field(default_factory=dict)
    force_transcribe: bool = False
    transcript_source: str | None = None
    source_subtitles: list[dict[str, Any]] = field(default_factory=list)
    # 应用自己写出 srt/ass 时的 mtime 戳;用于区分"应用写的"与"外部编辑过的"文件,
    # 防止 list_segments 把应用刚写好的字幕误判为外部编辑而反向覆盖(丢失 speaker/items)。
    subtitle_file_stamps: dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    phase_times: dict[str, float] = field(default_factory=dict)
    error_kind: str | None = None

    @property
    def raw_transcript_path(self) -> Path:
        return Path(self.job_dir) / "raw_transcript.txt"

    @property
    def raw_words_path(self) -> Path:
        return Path(self.job_dir) / "raw_words.json"

    @property
    def segments_path(self) -> Path:
        return Path(self.job_dir) / "segments.json"

    @property
    def source_segments_path(self) -> Path:
        return Path(self.job_dir) / "segments.source.json"

    @property
    def proofread_path(self) -> Path:
        return Path(self.job_dir) / "proofread.json"

    @property
    def alignment_path(self) -> Path:
        return Path(self.job_dir) / "alignment.json"

    @property
    def srt_path(self) -> Path:
        return Path(self.job_dir) / "subtitle.srt"

    @property
    def ass_path(self) -> Path:
        return Path(self.job_dir) / "subtitle.ass"

    @property
    def output_path(self) -> Path:
        return Path(self.job_dir) / "output.mp4"

    @property
    def clips_dir(self) -> Path:
        return Path(self.job_dir) / "clips"

    @property
    def job_path(self) -> Path:
        return Path(self.job_dir) / "job.json"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        possibly_truncated = (
            self.generated_tokens is not None
            and self.max_new_tokens > 0
            and self.generated_tokens >= self.max_new_tokens
        )
        data["inference"] = {
            "prompt": self.inference_prompt,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "decoding": self.decoding,
            "temperature": self.temperature,
        }
        data["usage"] = {
            "prompt_tokens": self.prompt_len,
            "generated_tokens": self.generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "possibly_truncated": possibly_truncated,
            "elapsed_sec": self.elapsed_sec,
        }
        data["files"] = {
            "raw_transcript": str(self.raw_transcript_path),
            "segments": str(self.segments_path),
            "srt": str(self.srt_path),
            "ass": str(self.ass_path),
            "mp4": str(self.output_path),
            "clips": str(self.clips_dir),
        }
        data["backend"] = self.backend
        data["translation"] = {
            **self.translation_info,
            "source_available": self.source_segments_path.exists(),
        }
        data["proofread"] = {
            **self.proofread_info,
            "result_available": self.proofread_path.exists(),
        }
        data["alignment"] = {
            **self.alignment_info,
            "result_available": self.alignment_path.exists(),
        }
        data["source"] = self.source
        if self.source_url:
            data["source_url"] = self.source_url
        if self.cookies_config:
            data["cookies_config"] = self.cookies_config
        if self.download_info:
            data["download_info"] = self.download_info
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        inference = data.get("inference") or {}
        temperature = data.get("temperature", inference.get("temperature"))
        return cls(
            id=str(data["id"]),
            status=str(data.get("status") or "failed"),
            media_name=str(data.get("media_name") or "input.media"),
            input_path=str(data.get("input_path") or ""),
            job_dir=str(data.get("job_dir") or ""),
            inference_prompt=str(data.get("inference_prompt") or inference.get("prompt") or ""),
            max_length=int(data.get("max_length") or inference.get("max_length") or 0),
            max_new_tokens=int(data.get("max_new_tokens") or inference.get("max_new_tokens") or 0),
            decoding=str(data.get("decoding") or inference.get("decoding") or "greedy"),
            temperature=None if temperature is None else float(temperature),
            progress=float(data.get("progress") or 0.0),
            error=data.get("error"),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            model=data.get("model"),
            prompt_len=data.get("prompt_len"),
            generated_tokens=data.get("generated_tokens"),
            elapsed_sec=data.get("elapsed_sec"),
            subtitle_style=dict(data.get("subtitle_style") or {}),
            backend=str(data.get("backend") or ""),
            speaker_labeling=dict(data.get("speaker_labeling") or {}),
            speaker_count=None if data.get("speaker_count") in ("", None) else int(data.get("speaker_count")),
            diarization_backend=str(data.get("diarization_backend") or "auto"),
            translation_info=dict(data.get("translation_info") or data.get("translation") or {}),
            proofread_info=dict(data.get("proofread_info") or data.get("proofread") or {}),
            alignment_info=dict(data.get("alignment_info") or data.get("alignment") or {}),
            hotwords=str(data.get("hotwords") or ""),
            source=str(data.get("source") or "upload"),
            source_url=data.get("source_url"),
            cookies_config=dict(data.get("cookies_config") or {}),
            download_info=dict(data.get("download_info") or {}),
            subtitle_file_stamps=dict(data.get("subtitle_file_stamps") or {}),
        )


class JobManager:
    def __init__(
        self,
        runs_dir: str | Path,
        model_runner: Any,
        *,
        prompt: str,
        max_length: int,
        max_new_tokens: int,
        decoding: str = "greedy",
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str = "none",
        hf_token: str | None = None,
        pyannote_model: str = "pyannote/speaker-diarization-3.1",
        diarization_device: str = "auto",
    ):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.model_runner = model_runner
        self.prompt = prompt
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.decoding = decoding
        self.temperature = temperature
        self.speaker_count = self._resolve_speaker_count(speaker_count)
        self.diarization_backend = diarization_backend
        self.hf_token = hf_token
        self.pyannote_model = pyannote_model
        self.diarization_device = diarization_device
        self._jobs: dict[str, JobRecord] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._queue_paused = False
        self._queue_condition = threading.Condition()
        self._phase_starts: dict[str, float] = {}
        self._render_lock = threading.Lock()
        self._translate_lock = threading.Lock()
        self._proofread_lock = threading.Lock()
        self._progress_save_times: dict[str, float] = {}
        self._cancelled_jobs: set[str] = set()
        # 正在渲染的片段(job_id/clip_id):两个 ffmpeg 写同一输出文件会互相损坏。
        self._rendering_clips: set[str] = set()
        self._rendering_clips_lock = threading.Lock()
        # worker 正在处理的任务 id：delete_job 据此有界等待 worker 停下,
        # 避免 rmtree 与 worker 写文件竞态(Windows 上文件被占用删不干净)。
        self._active_jobs: set[str] = set()
        self._event_hub: Any = None
        self._event_loop: Any = None
        self._emit_times: dict[str, float] = {}
        self._emit_status: dict[str, str] = {}
        self._job_loggers: dict[str, logging.Logger] = {}
        # segments.json 的解析缓存: job_id -> (stat 指纹, 解析结果)。
        # 指纹 = (mtime_ns, size),任何写入(应用保存/外部 srt 反向同步)都会改变它,缓存自动失效。
        self._segments_cache: dict[str, tuple[tuple[int, int], list[dict[str, Any]]]] = {}
        # job_id -> (input stat, segment time signature, adaptive end map).
        self._tail_padding_cache: dict[str, tuple[tuple[int, int], tuple[tuple[str, float, float], ...], dict[str, float] | None]] = {}
        self._load_existing_jobs()
        self._worker = threading.Thread(target=self._worker_loop, name="mtd-job-worker", daemon=True)
        self._worker.start()

    def create_job_from_file(
        self,
        source_path: str | Path,
        media_name: str | None = None,
        *,
        prompt: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        decoding: str | None = None,
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str | None = None,
        hotwords: str | None = None,
    ) -> JobRecord:
        options = self._resolve_inference_options(
            prompt=prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
        )
        job_id = uuid.uuid4().hex[:12]
        source_path = Path(source_path)
        suffix = source_path.suffix or ".media"
        job_dir = self.runs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path = job_dir / f"input{suffix}"
        shutil.copyfile(source_path, input_path)
        job = JobRecord(
            id=job_id,
            status="queued",
            progress=0.0,
            media_name=media_name or source_path.name,
            input_path=str(input_path),
            job_dir=str(job_dir),
            inference_prompt=options["prompt"],
            max_length=options["max_length"],
            max_new_tokens=options["max_new_tokens"],
            decoding=options["decoding"],
            temperature=options["temperature"],
            model=self.model_runner.model_path,
            backend=self._runner_backend(),
            speaker_count=self._resolve_speaker_count(speaker_count),
            diarization_backend=self._resolve_diarization_backend(diarization_backend),
            hotwords=str(hotwords or ""),
        )
        self._jobs[job.id] = job
        self._save_job(job)
        self._queue.put(job.id)
        return job

    def create_job_for_upload(
        self,
        filename: str,
        *,
        prompt: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        decoding: str | None = None,
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str | None = None,
        hotwords: str | None = None,
    ) -> tuple[JobRecord, Path]:
        options = self._resolve_inference_options(
            prompt=prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
        )
        job_id = uuid.uuid4().hex[:12]
        suffix = Path(filename).suffix or ".media"
        job_dir = self.runs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path = job_dir / f"input{suffix}"
        job = JobRecord(
            id=job_id,
            status="queued",
            progress=0.0,
            media_name=filename,
            input_path=str(input_path),
            job_dir=str(job_dir),
            inference_prompt=options["prompt"],
            max_length=options["max_length"],
            max_new_tokens=options["max_new_tokens"],
            decoding=options["decoding"],
            temperature=options["temperature"],
            model=self.model_runner.model_path,
            backend=self._runner_backend(),
            speaker_count=self._resolve_speaker_count(speaker_count),
            diarization_backend=self._resolve_diarization_backend(diarization_backend),
            hotwords=str(hotwords or ""),
        )
        self._jobs[job.id] = job
        self._save_job(job)
        self.job_log(
            job,
            f"任务创建(上传): media={job.media_name}, model={job.model}, speaker_count={job.speaker_count}, "
            f"diarization={job.diarization_backend}, max_new_tokens={job.max_new_tokens}, decoding={job.decoding}",
        )
        return job, input_path

    def create_job_for_url(
        self,
        url: str,
        *,
        cookies_browser: str | None = "firefox",
        cookies_file: str | None = None,
        prompt: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        decoding: str | None = None,
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str | None = None,
        force_transcribe: bool = False,
        hotwords: str | None = None,
    ) -> JobRecord:
        options = self._resolve_inference_options(
            prompt=prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
        )
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.runs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        cookies_config: dict[str, Any] = {"browser": cookies_browser, "file": cookies_file}
        job = JobRecord(
            id=job_id,
            status="queued",
            progress=0.0,
            media_name=url[:80],
            input_path=str(job_dir / "input.media"),
            job_dir=str(job_dir),
            inference_prompt=options["prompt"],
            max_length=options["max_length"],
            max_new_tokens=options["max_new_tokens"],
            decoding=options["decoding"],
            temperature=options["temperature"],
            model=self.model_runner.model_path,
            backend=self._runner_backend(),
            speaker_count=self._resolve_speaker_count(speaker_count),
            diarization_backend=self._resolve_diarization_backend(diarization_backend),
            source="url",
            source_url=url,
            cookies_config=cookies_config,
            force_transcribe=bool(force_transcribe),
            hotwords=str(hotwords or ""),
        )
        self._jobs[job.id] = job
        self._save_job(job)
        self.job_log(
            job,
            f"任务创建(链接): url={job.source_url}, model={job.model}, speaker_count={job.speaker_count}, "
            f"diarization={job.diarization_backend}, force_transcribe={job.force_transcribe}",
        )
        self._queue.put(job.id)
        return job

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def queue_info(self) -> dict[str, Any]:
        """Return queue state without exposing the internal queue object."""
        with self._queue_condition:
            paused = self._queue_paused
        return {"paused": paused, "pending": self._queue.qsize(), "active": len(self._active_jobs)}

    def pause_queue(self) -> dict[str, Any]:
        with self._queue_condition:
            self._queue_paused = True
        return self.queue_info()

    def resume_queue(self) -> dict[str, Any]:
        with self._queue_condition:
            self._queue_paused = False
            self._queue_condition.notify_all()
        return self.queue_info()

    # ------------------------------------------------------------ 事件与日志

    def bind_events(self, loop: Any, hub: Any) -> None:
        """server 启动时注入事件循环与 SSE hub；worker 线程经 call_soon_threadsafe 桥接。"""
        self._event_loop = loop
        self._event_hub = hub

    def _emit_job_event(self, job: JobRecord, *, force: bool = False) -> None:
        hub = self._event_hub
        loop = self._event_loop
        if hub is None or loop is None:
            return
        now = time.time()
        if not force and now - self._emit_times.get(job.id, 0.0) < 0.4:
            return
        self._emit_times[job.id] = now
        try:
            loop.call_soon_threadsafe(hub.publish, f"job:{job.id}", "job", job.to_dict())
        except RuntimeError:
            pass  # 关机时事件循环已关闭

    def _logger_for(self, job: JobRecord) -> logging.Logger:
        """每任务一个独立 logger，写 job_dir/pipeline.log，互不串扰。

        propagate=True 让消息同时走 root logger 上控制台：长静默阶段
        （pyannote 全程无进度输出）终端不再"假死"，用户不至于误删任务。
        """
        log = self._job_loggers.get(job.id)
        if log is not None:
            return log
        log = logging.getLogger(f"mtd.job.{job.id}")
        log.setLevel(logging.INFO)
        log.propagate = True
        log.handlers.clear()
        try:
            handler = _AppendFileHandler(Path(job.job_dir) / "pipeline.log")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(handler)
        except OSError:
            pass  # 目录不可写时退化为只走控制台
        self._job_loggers[job.id] = log
        return log

    def job_log(self, job: JobRecord, message: str, *, level: int = logging.INFO, exc_info: bool = False) -> None:
        try:
            self._logger_for(job).log(level, message, exc_info=exc_info)
        except Exception:
            pass

    def _close_job_logger(self, job_id: str) -> None:
        log = self._job_loggers.pop(job_id, None)
        if log is None:
            return
        for handler in log.handlers:
            try:
                handler.close()
            except Exception:
                pass
        log.handlers.clear()

    # ------------------------------------------------------------ 取消

    def cancel_job(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if job.status not in RUNNING_STATES:
            raise RuntimeError("任务当前不在运行中。")
        self._cancelled_jobs.add(job_id)
        self.job_log(job, "用户请求取消任务")
        self._emit_job_event(job, force=True)
        return job

    # ------------------------------------------------------------ 热词词表

    @property
    def hotwords_glossary_path(self) -> Path:
        return Path(self.runs_dir).parent / "config" / "hotwords.json"

    def load_hotwords_glossary(self) -> list[str]:
        path = self.hotwords_glossary_path
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        terms = data.get("terms") if isinstance(data, dict) else data
        if not isinstance(terms, list):
            return []
        seen: list[str] = []
        for term in terms:
            term = str(term or "").strip()
            if term and term not in seen:
                seen.append(term)
        return seen

    def save_hotwords_glossary(self, terms: list[str]) -> list[str]:
        cleaned: list[str] = []
        for term in terms:
            term = str(term or "").strip()
            if term and term not in cleaned:
                cleaned.append(term)
        path = self.hotwords_glossary_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"terms": cleaned}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return cleaned

    def add_hotwords_to_glossary(self, new_terms: list[str]) -> list[str]:
        if not new_terms:
            return self.load_hotwords_glossary()
        merged = self.load_hotwords_glossary()
        changed = False
        for term in new_terms:
            term = str(term or "").strip()
            if term and term not in merged:
                merged.append(term)
                changed = True
        if changed:
            self.save_hotwords_glossary(merged)
        return merged

    def _effective_hotwords(self, job: JobRecord) -> str:
        """job 手填热词 + 全局词表 合并（去重、保序）。"""
        tokens: list[str] = []
        for raw in (job.hotwords, " ".join(self.load_hotwords_glossary())):
            for token in re.split(r"[\s,，;；]+", str(raw or "")):
                token = token.strip()
                if token and token not in tokens:
                    tokens.append(token)
        return " ".join(tokens)

    def rerun_job(
        self,
        job_id: str,
        *,
        prompt: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        decoding: str | None = None,
        temperature: float | None = None,
        speaker_count: int | None = None,
        diarization_backend: str | None = None,
        hotwords: str | None = None,
    ) -> JobRecord:
        source = self.get_job(job_id)
        input_path = Path(source.input_path)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        return self.create_job_from_file(
            input_path,
            media_name=source.media_name,
            prompt=source.inference_prompt if prompt is None else prompt,
            max_length=source.max_length if max_length is None else max_length,
            max_new_tokens=source.max_new_tokens if max_new_tokens is None else max_new_tokens,
            decoding=source.decoding if decoding is None else decoding,
            temperature=source.temperature if temperature is None else temperature,
            speaker_count=source.speaker_count if speaker_count is None else speaker_count,
            diarization_backend=source.diarization_backend if diarization_backend is None else diarization_backend,
            hotwords=source.hotwords if hotwords is None else hotwords,
        )

    def list_jobs(self) -> list[JobRecord]:
        return sorted(self._jobs.values(), key=lambda job: job.updated_at, reverse=True)

    def get_job(self, job_id: str) -> JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"Unknown job: {job_id}") from exc

    def delete_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            self._cancelled_jobs.add(job_id)
            self._jobs.pop(job_id, None)
            # 有界等待 worker 真正停下再删目录，避免 rmtree 与文件写入竞态；
            # 超时（worker 卡在不可中断的子进程里）则退回旧行为：尽力删。
            deadline = time.time() + 15.0
            while time.time() < deadline and job_id in self._active_jobs:
                time.sleep(0.1)
        self._close_job_logger(job_id)
        self._jobs.pop(job_id, None)
        self._emit_times.pop(job_id, None)
        self._emit_status.pop(job_id, None)
        self._segments_cache.pop(job_id, None)
        self._tail_padding_cache.pop(job_id, None)
        self._progress_save_times.pop(job_id, None)
        self._cancelled_jobs.discard(job_id)
        shutil.rmtree(job.job_dir, ignore_errors=True)

    def list_segments(self, job_id: str) -> list[dict[str, Any]]:
        """读取任务字幕段落(带词级 items)。

        命中缓存时返回的是共享对象——调用方不得就地修改返回值或其嵌套结构
        (现有调用点均为只读或 from_dict/{**seg} 拷贝,新增调用点必须遵守)。
        """
        data, _version = self._load_segments(job_id)
        return self._decorate_display_ends(self.get_job(job_id), data)

    def list_segments_with_version(self, job_id: str) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
        """同 list_segments,并附带 segments.json 的 stat 指纹(供 ETag/304 判断)。

        指纹与缓存 key 来自同一次调用,保证"数据与指纹"一致,不会出现
        ETag 比数据新导致客户端 304 卡在旧稿的情况。
        """
        data, version = self._load_segments(job_id)
        return self._decorate_display_ends(self.get_job(job_id), data), version

    def _adaptive_end_map(self, job: JobRecord, segments: list[SubtitleSegment]) -> dict[str, float] | None:
        input_path = Path(job.input_path)
        try:
            stat = input_path.stat()
            input_key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return None
        signature = tuple((segment.id, float(segment.start), float(segment.end)) for segment in segments)
        cached = self._tail_padding_cache.get(job.id)
        if cached is not None and cached[0] == input_key and cached[1] == signature:
            return cached[2]
        result = _adaptive_tail_endings(input_path, segments)
        self._tail_padding_cache[job.id] = (input_key, signature, result)
        return result

    def _decorate_display_ends(self, job: JobRecord, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        segments = [SubtitleSegment.from_dict(item) for item in data if isinstance(item, dict)]
        end_map = self._adaptive_end_map(job, segments)
        for index, (item, segment) in enumerate(zip(data, segments)):
            next_start = segments[index + 1].start if index + 1 < len(segments) else segment.end + _TAIL_FALLBACK_PADDING
            fallback = min(next_start, segment.end + _TAIL_FALLBACK_PADDING)
            item["display_end"] = round(float((end_map or {}).get(segment.id, fallback)), 3)
        # Keep the parsed list identity: callers use it as an immutable cache value
        # and existing API code relies on repeated reads returning the same object.
        return data

    def _export_segments_for_job(self, job: JobRecord, segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
        end_map = self._adaptive_end_map(job, segments)
        return _padded_export_segments(segments, end_overrides=end_map)

    def _load_segments(self, job_id: str) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
        job = self.get_job(job_id)
        self._maybe_sync_segments_from_subtitle_files(job)
        path = job.segments_path
        if not path.exists():
            self._segments_cache.pop(job_id, None)
            return [], None
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._segments_cache.get(job_id)
        if cached is not None and cached[0] == key:
            # 命中即移到末尾:配合上限弹出最早条目,行为等价 LRU。
            self._segments_cache[job_id] = self._segments_cache.pop(job_id)
            return cached[1], key
        data = json.loads(path.read_text(encoding="utf-8"))
        self._segments_cache[job_id] = (key, data)
        while len(self._segments_cache) > _SEGMENTS_CACHE_MAX:
            oldest = next(iter(self._segments_cache))
            if oldest == job_id:
                break
            self._segments_cache.pop(oldest, None)
        return data, key

    def update_segments(
        self,
        job_id: str,
        payload: list[dict[str, Any]],
        style_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        job = self.get_job(job_id)
        segments = coerce_subtitle_segments(payload)
        self._backfill_items_from_disk(job, segments)
        style = SubtitleStyle.from_dict(style_payload) if style_payload is not None else None
        if style is not None:
            job.subtitle_style = style.to_dict()
        self._write_subtitle_files(job, segments, style=style)
        if job.status == "done":
            self._set_status(job, "waiting_review", 0.95, error=None)
        else:
            self._touch(job, error=None)
        return self._decorate_display_ends(job, [segment.to_dict() for segment in segments])

    # ------------------------------------------------------------ 查找与替换

    def search_segments(
        self,
        job_id: str,
        query: str,
        *,
        mode: str = "literal",
        limit: int = 500,
    ) -> dict[str, Any]:
        if mode not in {"literal", "pinyin"}:
            raise ValueError("mode must be 'literal' or 'pinyin'.")
        segments = self.list_segments(job_id)
        matches = search_segment_texts(segments, query, mode=mode, limit=limit)
        return {
            "query": query,
            "mode": mode,
            "total": len(matches),
            "matches": [
                {
                    "segment_id": match.segment_id,
                    "index": match.index,
                    "char_start": match.char_start,
                    "char_end": match.char_end,
                    "snippet": match.snippet,
                }
                for match in matches
            ],
        }

    def replace_segments(
        self,
        job_id: str,
        query: str,
        replacement: str,
        *,
        mode: str = "literal",
        segment_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot edit segments while the job is running.")
        if mode not in {"literal", "pinyin"}:
            raise ValueError("mode must be 'literal' or 'pinyin'.")
        if not str(query):
            raise ValueError("query must not be empty.")
        segments = self.list_segments(job_id)
        allowed = set(segment_ids) if segment_ids is not None else None
        payload: list[dict[str, Any]] = []
        hits = 0
        touched_ids: list[str] = []
        for segment in segments:
            if allowed is not None and str(segment.get("id")) not in allowed:
                payload.append(segment)
                continue
            text = str(segment.get("text") or "")
            matches = search_segment_texts([segment], query, mode=mode)
            if not matches:
                payload.append(segment)
                continue
            ranges = [(match.char_start, match.char_end) for match in matches]
            payload.append({**segment, "text": apply_replacements(text, ranges, str(replacement))})
            hits += len(ranges)
            touched_ids.append(str(segment.get("id")))
        if not hits:
            return {"replacements": 0, "segments": segments, "segment_ids": []}
        # 走 update_segments：词级 items 演化 + srt/ass 同步重写。
        segments_out = self.update_segments(job_id, payload)
        self.job_log(job, f"批量替换 mode={mode}: {query!r} -> {replacement!r}，共 {hits} 处 / {len(touched_ids)} 段")
        return {"replacements": hits, "segments": segments_out, "segment_ids": touched_ids}

    def _backfill_items_from_disk(self, job: JobRecord, segments: list[SubtitleSegment]) -> None:
        """整包保存的 payload 通常不带 items(前端只缓存 5 个字段)。
        按 id 从现有 segments.json 回填词级数据并夹到新边界内,避免词级真源被静默清空。
        text 被编辑过时,回填后立即用词级 diff 把旧 items 演化成新文本的 items。"""
        if not job.segments_path.exists():
            return
        try:
            existing = json.loads(job.segments_path.read_text(encoding="utf-8"))
        except Exception:
            return
        by_id = {str(item.get("id")): item for item in existing if isinstance(item, dict)}
        for segment in segments:
            if segment.items is not None:
                continue
            record = by_id.get(segment.id)
            if not record:
                continue
            items = coerce_subtitle_items(record.get("items"))
            if not items:
                continue
            items = [
                SubtitleItem(
                    text=item.text,
                    start=min(max(item.start, segment.start), segment.end),
                    end=min(max(item.end, segment.start), segment.end),
                )
                for item in items
            ]
            old_text = str(record.get("text") or "")
            items_norm = _normalize_words(" ".join(i.text.strip() for i in items if i.text.strip()))
            text_norm = _normalize_words(segment.text)
            disk_norm = _normalize_words(old_text)
            if items_norm == text_norm or items_norm != disk_norm:
                # items 与当前文本一致(未编辑/已对齐),或 items 与文本本就错配
                # (如已翻译任务: 中文 text + 英文 items): 原样保留,不做对齐。
                segment.items = items
            else:
                # items 对应磁盘旧文本且 text 已被编辑: 词级 diff 演化(重写降级 None)。
                segment.items = align_items_to_text(old_text, items, segment.text)

    def _load_segments_records(self, job: JobRecord) -> list[SubtitleSegment]:
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")
        return [SubtitleSegment.from_dict(item) for item in json.loads(job.segments_path.read_text(encoding="utf-8"))]

    @staticmethod
    def _unique_segment_id(segments: list[SubtitleSegment], base: str) -> str:
        existing = {segment.id for segment in segments}
        if base not in existing:
            return base
        n = 2
        while f"{base}~{n}" in existing:
            n += 1
        return f"{base}~{n}"

    @staticmethod
    def _split_record_at(segments: list[SubtitleSegment], index: int, t: float, new_id: str) -> tuple[SubtitleSegment, SubtitleSegment]:
        """在时间 t 处把 segments[index] 拆成两条;有词级 items 时吸附到最近词边界。
        text 是内容真源: 编辑过的段落拆分时保留编辑文本(按前半段词时长占比映射字符切点),
        绝不从旧 items 重建文本回退用户修改。"""
        seg = segments[index]
        duration = seg.end - seg.start
        t = min(max(t, seg.start + 0.1), seg.end - 0.1)

        if seg.items:
            best_k = None
            best_dist = None
            for k in range(1, len(seg.items)):
                dist = abs(seg.items[k].start - t)
                if best_dist is None or dist < best_dist:
                    best_k, best_dist = k, dist
            if best_k is None:
                raise ValueError("Segment has only one word; cannot split.")
            left_items = seg.items[:best_k]
            right_items = seg.items[best_k:]
            items_text = " ".join(i.text.strip() for i in seg.items if i.text.strip())
            if _normalize_words(items_text) == _normalize_words(seg.text):
                # 未编辑过: 从 items 重建文本(最精确)。
                left_text = " ".join(i.text.strip() for i in left_items if i.text.strip())
                right_text = " ".join(i.text.strip() for i in right_items if i.text.strip())
            else:
                # 编辑过: 时间按词边界切,文本按前半段词时长占比映射切点,保留用户修改。
                left_dur = sum(
                    (min(i.end, seg.end) - max(i.start, seg.start)) for i in left_items
                ) or 1.0
                total_dur = max(
                    sum((min(i.end, seg.end) - max(i.start, seg.start)) for i in seg.items), left_dur
                )
                ratio = min(1.0, max(0.0, left_dur / total_dur))
                cut = max(1, min(len(seg.text) - 1, round(len(seg.text) * ratio)))
                left_text = seg.text[:cut].rstrip()
                right_text = seg.text[cut:].lstrip()
            left = SubtitleSegment(
                id=seg.id, start=seg.start, end=max(left_items[-1].end, seg.start),
                speaker=seg.speaker, text=left_text or seg.text, items=left_items, confidence=seg.confidence, quality_flags=seg.quality_flags, quality_reasons=seg.quality_reasons,
            )
            right = SubtitleSegment(
                id=new_id,
                start=min(right_items[0].start, seg.end), end=seg.end,
                speaker=seg.speaker, text=right_text or seg.text, items=right_items, confidence=seg.confidence, quality_flags=seg.quality_flags, quality_reasons=seg.quality_reasons,
            )
        else:
            ratio = (t - seg.start) / duration
            cut = max(1, min(len(seg.text) - 1, round(len(seg.text) * ratio)))
            left = SubtitleSegment(
                id=seg.id, start=seg.start, end=t, speaker=seg.speaker,
                text=seg.text[:cut].rstrip(), items=None, confidence=seg.confidence, quality_flags=seg.quality_flags, quality_reasons=seg.quality_reasons,
            )
            right = SubtitleSegment(
                id=new_id, start=t, end=seg.end,
                speaker=seg.speaker, text=seg.text[cut:].lstrip(), items=None, confidence=seg.confidence, quality_flags=seg.quality_flags, quality_reasons=seg.quality_reasons,
            )
        return left, right

    def _sync_source_split(self, job: JobRecord, segment_id: str, t: float, new_id: str) -> bool:
        """翻译过的 job:对源稿备份做同样的拆分,保证重译时结构一致。"""
        if not job.source_segments_path.exists():
            return False
        try:
            source = [
                SubtitleSegment.from_dict(item)
                for item in json.loads(job.source_segments_path.read_text(encoding="utf-8"))
            ]
            s_index = next((i for i, s in enumerate(source) if s.id == segment_id), None)
            if s_index is None:
                return False
            left, right = self._split_record_at(source, s_index, t, new_id)
            source[s_index : s_index + 1] = [left, right]
            write_text(job.source_segments_path, export_json(source))
            return True
        except Exception:
            return False

    def _sync_source_merge(self, job: JobRecord, segment_ids: list[str]) -> bool:
        """翻译过的 job:对源稿备份做同样的合并。"""
        if not job.source_segments_path.exists():
            return False
        try:
            source = [
                SubtitleSegment.from_dict(item)
                for item in json.loads(job.source_segments_path.read_text(encoding="utf-8"))
            ]
            id_set = set(segment_ids)
            indexes = sorted(i for i, s in enumerate(source) if s.id in id_set)
            if len(indexes) != len(id_set) or indexes != list(range(indexes[0], indexes[0] + len(indexes))):
                return False
            group = [source[i] for i in indexes]
            first = group[0]
            items = (
                [item for s in group for item in (s.items or [])]
                if all(s.items is not None for s in group)
                else None
            )
            merged = SubtitleSegment(
                id=first.id,
                start=min(s.start for s in group),
                end=max(s.end for s in group),
                speaker=first.speaker,
                text=" ".join(s.text.strip() for s in group if s.text.strip()),
                items=items,
            )
            source[indexes[0] : indexes[0] + len(indexes)] = [merged]
            write_text(job.source_segments_path, export_json(source))
            return True
        except Exception:
            return False

    def _mark_structure_changed(self, job: JobRecord) -> None:
        info = dict(job.translation_info or {})
        if info:
            info["structure_changed"] = True
            job.translation_info = info

    def split_segment(
        self,
        job_id: str,
        segment_id: str,
        split_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """在指定时间点(缺省取中点)把一条字幕拆成两条。

        有词级 items 时在最近的词边界下刀,文本按词分配;
        没有时按时长比例切字符(fallback)。
        翻译过的 job 会同步拆源稿备份,重译时保留新结构。"""
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot edit segments while the job is running.")
        segments = self._load_segments_records(job)
        index = next((i for i, s in enumerate(segments) if s.id == segment_id), None)
        if index is None:
            raise KeyError(f"Segment {segment_id} not found.")
        seg = segments[index]
        duration = seg.end - seg.start
        if duration < 0.4:
            raise ValueError("Segment is too short to split.")

        t = seg.start + duration / 2 if split_time is None else float(split_time)
        new_id = self._unique_segment_id(segments, seg.id)
        left, right = self._split_record_at(segments, index, t, new_id)
        segments[index : index + 1] = [left, right]

        source_synced = self._sync_source_split(job, segment_id, t, new_id)
        if source_synced:
            self._mark_structure_changed(job)
            # 已翻译任务: 拆出的两条退回源文文本(译文按旧结构已失真,structure_changed 会提示重译)。
            try:
                source_by_id = {
                    str(item.get("id")): item
                    for item in json.loads(job.source_segments_path.read_text(encoding="utf-8"))
                }
                for pos in (index, index + 1):
                    piece = segments[pos]
                    record = source_by_id.get(piece.id)
                    if record is not None and str(record.get("text") or "") != piece.text:
                        segments[pos] = SubtitleSegment(
                            id=piece.id, start=piece.start, end=piece.end,
                            speaker=piece.speaker, text=str(record.get("text") or piece.text),
                            items=piece.items,
                        )
            except Exception:
                logger.debug("split: fallback to source text skipped", exc_info=True)
        self._write_subtitle_files(job, segments)
        self._touch(job, error=None)
        return self._decorate_display_ends(job, [segment.to_dict() for segment in segments])

    def merge_segments(self, job_id: str, segment_ids: list[str]) -> list[dict[str, Any]]:
        """把多条**相邻**字幕合并成一条;说话人取第一条,items 依序拼接。
        翻译过的 job 会同步合并源稿备份,重译时保留新结构。"""
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot edit segments while the job is running.")
        segments = self._load_segments_records(job)
        wanted = [str(sid) for sid in segment_ids]
        if not wanted:
            raise ValueError("No segments to merge.")
        id_set = set(wanted)
        indexes = sorted(i for i, s in enumerate(segments) if s.id in id_set)
        if len(indexes) != len(id_set):
            missing = id_set - {segments[i].id for i in indexes}
            raise KeyError(f"Segment(s) not found: {', '.join(sorted(missing))}")
        if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            raise ValueError("Only adjacent segments can be merged.")

        group = [segments[i] for i in indexes]
        first = group[0]
        text = " ".join(s.text.strip() for s in group if s.text.strip())
        items = (
            [item for s in group for item in (s.items or [])]
            if all(s.items is not None for s in group)
            else None
        )
        merged = SubtitleSegment(
            id=first.id,
            start=min(s.start for s in group),
            end=max(s.end for s in group),
            speaker=first.speaker,
            text=text,
            items=items,
        )
        segments[indexes[0] : indexes[0] + len(indexes)] = [merged]

        source_synced = self._sync_source_merge(job, wanted)
        if source_synced:
            self._mark_structure_changed(job)
        self._write_subtitle_files(job, segments)
        self._touch(job, error=None)
        return self._decorate_display_ends(job, [segment.to_dict() for segment in segments])

    def sync_segments_from_subtitle_files(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        result = self._maybe_sync_segments_from_subtitle_files(job, force=True)
        segments = self.list_segments(job_id)
        return {
            "synced": bool(result),
            "source": result["source"] if result else None,
            "count": len(segments),
            "segments": segments,
        }

    def render(self, job_id: str, style_payload: dict[str, Any] | None = None) -> JobRecord:
        job = self.get_job(job_id)
        if not detect_ffmpeg().available:
            raise RuntimeError("ffmpeg and ffprobe are not available on PATH.")
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")
        if job.status == "rendering":
            return job
        if job.status in RUNNING_STATES - {"rendering"}:
            raise RuntimeError("Cannot render before transcription is ready.")
        self._set_status(job, "rendering", 0.95, error=None)
        threading.Thread(
            target=self._render_job,
            args=(job.id, SubtitleStyle.from_dict(style_payload)),
            name=f"mtd-render-{job.id}",
            daemon=True,
        ).start()
        return job

    def list_clip_candidates(
        self,
        job_id: str,
        *,
        min_duration: float = 45.0,
        target_duration: float = 120.0,
        max_duration: float | None = 180.0,
        limit: int = 24,
        selector: Any | None = None,
    ) -> list[dict[str, Any]]:
        segments = [SubtitleSegment.from_dict(item) for item in self.list_segments(job_id)]
        rule_limit = max(limit, min(48, limit * 4)) if selector is not None else limit
        candidates = [
            candidate.to_dict()
            for candidate in generate_clip_candidates(
                segments,
                min_duration=min_duration,
                target_duration=target_duration,
                max_duration=max_duration,
                limit=rule_limit,
            )
        ]
        if selector is not None:
            return selector.rank_clip_candidates(candidates, limit=limit)
        return candidates

    def render_clip(
        self,
        job_id: str,
        *,
        start: float,
        end: float,
        style_payload: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not detect_ffmpeg().available:
            raise RuntimeError("ffmpeg and ffprobe are not available on PATH.")
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")

        start = max(0.0, float(start))
        end = max(start + 0.25, float(end))
        style = SubtitleStyle.from_dict(style_payload)
        clip_id = _safe_clip_name(name or f"clip_{start:.2f}_{end:.2f}")
        clip_key = f"{job.id}/{clip_id}"
        with self._rendering_clips_lock:
            if clip_key in self._rendering_clips:
                raise RuntimeError("该片段正在渲染中,请等当前渲染完成后再试。")
            self._rendering_clips.add(clip_key)
        try:
            job.clips_dir.mkdir(parents=True, exist_ok=True)
            ass_path = job.clips_dir / f"{clip_id}.ass"
            srt_path = job.clips_dir / f"{clip_id}.srt"
            metadata_path = job.clips_dir / f"{clip_id}.json"
            output_path = job.clips_dir / f"{clip_id}.mp4"
            segments = self._clip_segments(job, start=start, end=end)
            if not segments:
                raise RuntimeError("The selected range does not contain any subtitle segments.")
            width, height = probe_video_size(job.input_path)
            write_text(
                ass_path,
                export_ass(_padded_export_segments(segments), style=style, video_width=width, video_height=height),
                encoding="utf-8-sig",
            )
            write_text(
                srt_path,
                export_srt(_padded_export_segments(segments), show_speaker=style.show_speaker, speaker_names=style.speaker_names),
                encoding="utf-8-sig",
            )
            write_text(
                metadata_path,
                json.dumps(
                    {
                        "source_media": job.media_name,
                        "source_start": start,
                        "source_end": end,
                        "duration": end - start,
                        "clip_timeline_start": 0.0,
                        "clip_timeline_end": end - start,
                        "segments": [segment.to_dict() for segment in segments],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            burn_ass_subtitles_clip(
                job.input_path,
                ass_path,
                output_path,
                start=start,
                end=end,
                style=style,
                # 切片烧录不进任务状态机（不占 RUNNING_STATES），取消检查主要
                # 兜住任务被删除的情况，避免给已删除目录继续写产物。
                cancel_check=lambda: job.id not in self._jobs,
            )
        finally:
            self._rendering_clips.discard(clip_key)
        return {
            "filename": output_path.name,
            "path": str(output_path),
            "start": start,
            "end": end,
            "duration": end - start,
            "segments": len(segments),
            "files": {
                "mp4": output_path.name,
                "srt": srt_path.name,
                "ass": ass_path.name,
                "metadata": metadata_path.name,
            },
        }

    def translate(
        self,
        job_id: str,
        translator: Any,
        *,
        target_language: str = "简体中文",
        mode: str = "replace",
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot translate while the job is running.")
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")

        with self._translate_lock:
            self._set_status(job, "translating", max(job.progress, 0.95), error=None)
            try:
                if not job.source_segments_path.exists():
                    shutil.copyfile(job.segments_path, job.source_segments_path)
                source_payload = json.loads(job.source_segments_path.read_text(encoding="utf-8"))
                segments = [SubtitleSegment.from_dict(item) for item in source_payload]
                pretranslation_skips = collect_pretranslation_skips(segments)
                started = time.time()

                def update_translation_progress(done: int, total: int, batch_start: int, batch_count: int) -> None:
                    ratio = 1.0 if total <= 0 else max(0.0, min(1.0, done / total))
                    self.job_log(job, f"翻译进度 {done}/{total}（批 {batch_start + 1}/{batch_count}）")
                    job.translation_info = {
                        "applied": False,
                        "in_progress": True,
                        "model": getattr(translator, "model", None),
                        "target_language": target_language,
                        "mode": mode,
                        "done": done,
                        "total": total,
                        "pretranslation_skip_count": len(pretranslation_skips),
                        "batch_start": batch_start,
                        "batch_count": batch_count,
                        "percent": round(ratio * 100, 1),
                        "elapsed_sec": round(time.time() - started, 3),
                    }
                    self._set_status(job, "translating", 0.95 + 0.04 * ratio, error=None)

                job.translation_info = {
                    "applied": False,
                    "in_progress": True,
                    "model": getattr(translator, "model", None),
                    "target_language": target_language,
                    "mode": mode,
                    "done": 0,
                    "total": len(segments),
                    "pretranslation_skip_count": len(pretranslation_skips),
                    "percent": 0.0,
                    "elapsed_sec": 0.0,
                }
                self._touch(job, error=None)
                translate_kwargs = {
                    "target_language": target_language,
                    "progress_callback": update_translation_progress,
                }
                if batch_size is not None:
                    translate_kwargs["batch_size"] = batch_size
                translations = translator.translate_segments(segments, **translate_kwargs)
                elapsed = time.time() - started
                validation_issues = validate_translation_outputs(segments, translations)
                translated = apply_translations(segments, translations, mode=mode)
                self._write_subtitle_files(job, translated)
                job.translation_info = {
                    "applied": True,
                    "in_progress": False,
                    "model": getattr(translator, "model", None),
                    "target_language": target_language,
                    "mode": mode,
                    "done": len(translated),
                    "total": len(segments),
                    "percent": 100.0,
                    "elapsed_sec": round(elapsed, 3),
                    "pretranslation_skip_count": len(pretranslation_skips),
                    "pretranslation_skips": pretranslation_skips[:50],
                    "validation_issue_count": len(validation_issues),
                    "validation_issues": validation_issues[:20],
                }
                job.status = "waiting_review"
                job.progress = 0.95
                self._touch(job, error=None)
                return {
                    "segments": [segment.to_dict() for segment in translated],
                    "count": len(translated),
                    "target_language": target_language,
                    "mode": mode,
                    "elapsed_sec": round(elapsed, 3),
                    "pretranslation_skip_count": len(pretranslation_skips),
                    "pretranslation_skips": pretranslation_skips[:50],
                    "validation_issue_count": len(validation_issues),
                    "validation_issues": validation_issues[:20],
                }
            except JobCancelled:
                self._cancelled_jobs.discard(job.id)
                job.translation_info = {**job.translation_info, "in_progress": False, "cancelled": True}
                job.status = "waiting_review"
                job.progress = max(job.progress, 0.95)
                self._touch(job, error=None)
                self.job_log(job, "翻译已取消，回到待校对")
                return {"cancelled": True}
            except Exception as exc:
                job.translation_info = {
                    **job.translation_info,
                    "in_progress": False,
                    "error": str(exc),
                }
                self._set_status(job, "waiting_review", max(job.progress, 0.95), error=f"Translation failed: {exc}")
                raise

    def restore_source_segments(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot restore subtitles while the job is running.")
        if not job.source_segments_path.exists():
            raise FileNotFoundError("No source subtitle backup is available.")
        segments = [
            SubtitleSegment.from_dict(item)
            for item in json.loads(job.source_segments_path.read_text(encoding="utf-8"))
        ]
        self._write_subtitle_files(job, segments)
        job.translation_info = {"applied": False}
        self._touch(job, error=None)
        return {"segments": [segment.to_dict() for segment in segments], "count": len(segments)}

    def _proofread_target(self, job: JobRecord) -> tuple[Path, str]:
        """校对目标: 已翻译过用源稿(译文不受影响), 否则用当前稿。"""
        if job.source_segments_path.exists():
            return job.source_segments_path, "source"
        return job.segments_path, "segments"

    def proofread(
        self,
        job_id: str,
        proofreader: Any,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot proofread while the job is running.")
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")

        with self._proofread_lock:
            self._set_status(job, "proofreading", max(job.progress, 0.95), error=None)
            try:
                target_path, target_kind = self._proofread_target(job)
                segments = [
                    SubtitleSegment.from_dict(item)
                    for item in json.loads(target_path.read_text(encoding="utf-8"))
                ]
                started = time.time()
                translated = target_kind == "source"

                def update_proofread_progress(phase: str, done: int, total: int) -> None:
                    if phase == "pass1":
                        if translated:
                            ratio = 0.5 * (done / total) if total else 0.5
                        else:
                            ratio = 0.7 * (done / total) if total else 0.7
                    else:
                        ratio = 0.6 if translated else 1.0
                    self.job_log(job, f"校对进度 phase={phase} {done}/{total}")
                    job.proofread_info = {
                        **job.proofread_info,
                        "in_progress": True,
                        "applied": False,
                        "phase": phase,
                        "done": done,
                        "total": total,
                        "percent": round(ratio * 100, 1),
                        "target": target_kind,
                    }
                    self._set_status(job, "proofreading", 0.95 + 0.04 * ratio, error=None)

                job.proofread_info = {
                    **job.proofread_info,
                    "in_progress": True,
                    "applied": False,
                    "phase": "pass1",
                    "done": 0,
                    "total": 0,
                    "percent": 0.0,
                    "target": target_kind,
                }
                self._touch(job, error=None)
                result = proofreader.proofread(segments, progress_callback=update_proofread_progress)
                # 一键校对: 已翻译任务在校对源稿之后追加译文对照检查(只读标注)。
                alignment_payload: dict[str, Any] | None = None
                if translated:
                    try:
                        pairs = self._alignment_pairs(job)
                        if pairs:
                            def update_alignment_progress(done: int, total: int) -> None:
                                if job.id in self._cancelled_jobs:
                                    raise JobCancelled()
                                ratio = (done / total) if total else 1.0
                                self.job_log(job, f"对照检查进度 {done}/{total}")
                                job.proofread_info = {
                                    **job.proofread_info,
                                    "phase": "alignment",
                                    "done": done,
                                    "total": total,
                                    "percent": round((0.6 + 0.39 * ratio) * 100, 1),
                                }
                                self._set_status(job, "proofreading", 0.95 + 0.04 * (0.6 + 0.39 * ratio), error=None)

                            issues = proofreader.check_alignment(pairs, progress_callback=update_alignment_progress)
                            alignment_payload = {
                                "issues": issues,
                                "issue_count": len(issues),
                                "pair_count": len(pairs),
                            }
                    except JobCancelled:
                        raise
                    except Exception:
                        # 对照检查是附加质检,失败不作废主校对已产出的修正。
                        logger.exception("Alignment check after proofread failed")
                result["target"] = target_kind
                result["created_at"] = time.time()
                result["elapsed_sec"] = round(time.time() - started, 1)
                if alignment_payload is not None:
                    result["alignment"] = alignment_payload
                    write_text(
                        job.alignment_path,
                        json.dumps({**alignment_payload, "created_at": time.time()}, ensure_ascii=False, indent=2),
                    )
                    job.alignment_info = {
                        "in_progress": False,
                        "issue_count": alignment_payload["issue_count"],
                        "pair_count": alignment_payload["pair_count"],
                    }
                write_text(job.proofread_path, json.dumps(result, ensure_ascii=False, indent=2))
                job.proofread_info = {
                    "in_progress": False,
                    "applied": False,
                    "target": target_kind,
                    "typo_count": len(result.get("suggestions") or []),
                    "term_count": len(result.get("term_corrections") or []),
                    "alignment_count": (result.get("alignment") or {}).get("issue_count", 0),
                    "elapsed_sec": result["elapsed_sec"],
                }
                job.status = "waiting_review"
                job.progress = 0.95
                self._touch(job, error=None)
                return result
            except JobCancelled:
                self._cancelled_jobs.discard(job.id)
                job.proofread_info = {**job.proofread_info, "in_progress": False, "cancelled": True}
                job.status = "waiting_review"
                job.progress = max(job.progress, 0.95)
                self._touch(job, error=None)
                self.job_log(job, "AI 校对已取消，回到待校对")
                return {"cancelled": True}
            except Exception as exc:
                job.proofread_info = {
                    **job.proofread_info,
                    "in_progress": False,
                    "error": str(exc),
                }
                self._set_status(job, "waiting_review", max(job.progress, 0.95), error=f"Proofread failed: {exc}")
                raise

    def get_proofread_result(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not job.proofread_path.exists():
            raise FileNotFoundError("No proofread result is available for this job.")
        result = json.loads(job.proofread_path.read_text(encoding="utf-8"))
        result["applied"] = bool(job.proofread_info.get("applied"))
        return result

    def apply_proofread(self, job_id: str, ids: list[str], terms: list[dict[str, Any]]) -> dict[str, Any]:
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot apply proofread while the job is running.")
        if not job.proofread_path.exists():
            raise FileNotFoundError("No proofread result is available for this job.")
        result = json.loads(job.proofread_path.read_text(encoding="utf-8"))
        target_path, target_kind = self._proofread_target(job)
        segments = [
            SubtitleSegment.from_dict(item)
            for item in json.loads(target_path.read_text(encoding="utf-8"))
        ]

        wanted = {str(item) for item in ids or []}
        applied_ids: set[str] = set()
        for suggestion in result.get("suggestions") or []:
            seg_id = str(suggestion.get("id") or "")
            if seg_id not in wanted:
                continue
            for index, segment in enumerate(segments):
                if segment.id == seg_id and segment.text == suggestion.get("original"):
                    corrected = str(suggestion.get("corrected") or segment.text)
                    new_items = segment.items
                    if segment.items:
                        items_norm = _normalize_words(
                            " ".join(i.text.strip() for i in segment.items if i.text.strip())
                        )
                        if items_norm == _normalize_words(segment.text) and items_norm != _normalize_words(corrected):
                            # items 对应当前文本且校对改了文本: 同步演化词级 items。
                            # 错配场景(已翻译任务)原样保留,不强行对齐。
                            new_items = align_items_to_text(segment.text, segment.items, corrected)
                    segments[index] = SubtitleSegment(
                        id=segment.id,
                        start=segment.start,
                        end=segment.end,
                        speaker=segment.speaker,
                        text=corrected,
                        items=new_items,
                    )
                    applied_ids.add(seg_id)
                    break

        term_hits = 0
        applied_terms: list[dict[str, Any]] = []
        valid_terms = {
            (str(t.get("wrong") or ""), str(t.get("right") or ""))
            for t in result.get("term_corrections") or []
        }
        for term in terms or []:
            wrong = str(term.get("wrong") or "").strip()
            right = str(term.get("right") or "").strip()
            if not wrong or not right or (wrong, right) not in valid_terms:
                continue
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            changed = 0
            for index, segment in enumerate(segments):
                # lambda 提供 replacement,避免 right 含 \ 时被当替换模板解析。
                new_text, count = pattern.subn(lambda _match: right, segment.text)
                if count and new_text != segment.text:
                    segments[index] = SubtitleSegment(
                        id=segment.id,
                        start=segment.start,
                        end=segment.end,
                        speaker=segment.speaker,
                        text=new_text,
                        items=segment.items,
                    )
                    changed += count
            if changed:
                term_hits += changed
                applied_terms.append({"wrong": wrong, "right": right, "hits": changed})

        needs_retranslate = False
        if target_kind == "segments":
            self._write_subtitle_files(job, segments)
        else:
            write_text(target_path, export_json(segments))
            needs_retranslate = True
        # 应用的术语自动沉淀进全局热词词表，下次转录 whisper 直接带进去
        glossary_terms = self.add_hotwords_to_glossary(
            [t.get("right") for t in applied_terms]
        )
        job.proofread_info = {
            **job.proofread_info,
            "applied": True,
            "applied_ids": sorted(applied_ids),
            "applied_terms": applied_terms,
            "term_hits": term_hits,
            "needs_retranslate": needs_retranslate,
        }
        self._touch(job, error=None)
        return {
            "applied_count": len(applied_ids),
            "applied_ids": sorted(applied_ids),
            "applied_terms": applied_terms,
            "term_hits": term_hits,
            "needs_retranslate": needs_retranslate,
            "glossary_terms": glossary_terms,
            "segments": [segment.to_dict() for segment in segments] if target_kind == "segments" else None,
        }

    # ------------------------------------------------------------ 对照检查

    def _alignment_pairs(self, job: JobRecord) -> list[dict[str, Any]]:
        """当前稿与源稿按 id 配对,产出送 LLM 的对照对。

        双语模式的当前稿是"译文\\n源文",取首行作译文;
        译文与源文相同(预翻译跳过保留原文)或任一侧为空的对不送检。"""
        current = [
            SubtitleSegment.from_dict(item)
            for item in json.loads(job.segments_path.read_text(encoding="utf-8"))
        ]
        source_by_id = {
            str(item.get("id")): str(item.get("text") or "")
            for item in json.loads(job.source_segments_path.read_text(encoding="utf-8"))
        }
        pairs: list[dict[str, Any]] = []
        for index, segment in enumerate(current):
            source_text = source_by_id.get(segment.id)
            if source_text is None or not source_text.strip():
                continue
            text = segment.text
            translated_text = text.split("\n", 1)[0].strip() if "\n" in text else text.strip()
            if not translated_text or translated_text == source_text.strip():
                continue
            pairs.append(
                {
                    "id": segment.id,
                    "index": index,
                    "start": segment.start,
                    "end": segment.end,
                    "source_text": source_text.strip(),
                    "translated_text": translated_text,
                }
            )
        return pairs

    def alignment_check(self, job_id: str, proofreader: Any) -> dict[str, Any]:
        """翻译后的源文-译文对照检查:LLM 只产只读标注,不改任何文本。"""
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot run the check while the job is running.")
        if not job.source_segments_path.exists():
            raise RuntimeError("Only translated jobs can run the alignment check.")
        if not job.segments_path.exists():
            raise RuntimeError("No subtitle segments are available for this job.")

        with self._proofread_lock:
            self._set_status(job, "proofreading", max(job.progress, 0.95), error=None)
            try:
                pairs = self._alignment_pairs(job)
                started = time.time()

                def update_progress(done: int, total: int) -> None:
                    if job.id in self._cancelled_jobs:
                        raise JobCancelled()
                    ratio = 1.0 if total <= 0 else done / total
                    self.job_log(job, f"对照检查进度 {done}/{total}")
                    job.alignment_info = {
                        **job.alignment_info,
                        "in_progress": True,
                        "done": done,
                        "total": total,
                        "percent": round(ratio * 100, 1),
                    }
                    self._set_status(job, "proofreading", 0.95 + 0.04 * ratio, error=None)

                job.alignment_info = {
                    **job.alignment_info,
                    "in_progress": True,
                    "done": 0,
                    "total": 0,
                    "percent": 0.0,
                }
                self._touch(job, error=None)
                issues = proofreader.check_alignment(pairs, progress_callback=update_progress)
                result = {
                    "issues": issues,
                    "issue_count": len(issues),
                    "pair_count": len(pairs),
                    "segment_count": len(json.loads(job.segments_path.read_text(encoding="utf-8"))),
                    "elapsed_sec": round(time.time() - started, 1),
                    "created_at": time.time(),
                }
                write_text(job.alignment_path, json.dumps(result, ensure_ascii=False, indent=2))
                job.alignment_info = {
                    "in_progress": False,
                    "issue_count": len(issues),
                    "pair_count": len(pairs),
                    "elapsed_sec": result["elapsed_sec"],
                }
                job.status = "waiting_review"
                job.progress = 0.95
                self._touch(job, error=None)
                return result
            except JobCancelled:
                self._cancelled_jobs.discard(job.id)
                job.alignment_info = {**job.alignment_info, "in_progress": False, "cancelled": True}
                job.status = "waiting_review"
                job.progress = max(job.progress, 0.95)
                self._touch(job, error=None)
                self.job_log(job, "对照检查已取消，回到待校对")
                return {"cancelled": True}
            except Exception as exc:
                job.alignment_info = {
                    **job.alignment_info,
                    "in_progress": False,
                    "error": str(exc),
                }
                self._set_status(job, "waiting_review", max(job.progress, 0.95), error=f"Alignment check failed: {exc}")
                raise

    def get_alignment_result(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        if not job.alignment_path.exists():
            raise FileNotFoundError("No alignment check result is available for this job.")
        return json.loads(job.alignment_path.read_text(encoding="utf-8"))

    def apply_alignment(self, job_id: str, ids: list[str]) -> dict[str, Any]:
        """把对照检查的建议译文应用到当前稿(双语模式保留源文行)。"""
        job = self.get_job(job_id)
        if job.status in RUNNING_STATES:
            raise RuntimeError("Cannot apply alignment fixes while the job is running.")
        if not job.alignment_path.exists():
            raise FileNotFoundError("No alignment check result is available for this job.")
        result = json.loads(job.alignment_path.read_text(encoding="utf-8"))
        issues = [item for item in result.get("issues") or [] if isinstance(item, dict)]
        by_id = {str(item.get("id") or ""): item for item in issues if item.get("suggested")}
        wanted = {str(item) for item in ids or []}
        if not wanted:
            return {"applied_count": 0, "applied_ids": [], "segments": None}

        segments = [
            SubtitleSegment.from_dict(item)
            for item in json.loads(job.segments_path.read_text(encoding="utf-8"))
        ]
        applied_ids: list[str] = []
        for index, segment in enumerate(segments):
            issue = by_id.get(segment.id)
            if issue is None or segment.id not in wanted:
                continue
            suggested = str(issue["suggested"]).strip()
            if not suggested:
                continue
            # 双语模式当前稿是 "译文\n源文":建议只替换译文行,源文行原样保留。
            if "\n" in segment.text:
                parts = segment.text.split("\n", 1)
                parts[0] = suggested
                new_text = "\n".join(parts)
            else:
                new_text = suggested
            if new_text == segment.text:
                continue
            segments[index] = SubtitleSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                text=new_text,
                items=segment.items,
            )
            applied_ids.append(segment.id)

        if applied_ids:
            self._write_subtitle_files(job, segments)
            result["issues"] = [
                item for item in issues if str(item.get("id")) not in set(applied_ids)
            ]
            result["issue_count"] = len(result["issues"])
            result["applied_ids"] = sorted({*result.get("applied_ids", []), *applied_ids})
            write_text(job.alignment_path, json.dumps(result, ensure_ascii=False, indent=2))
            job.alignment_info = {
                **job.alignment_info,
                "issue_count": result["issue_count"],
                "applied_count": len(result["applied_ids"]),
            }
            self._touch(job, error=None)
        return {
            "applied_count": len(applied_ids),
            "applied_ids": sorted(applied_ids),
            "segments": [segment.to_dict() for segment in segments] if applied_ids else None,
        }

    def clip_download_path(self, job_id: str, filename: str) -> Path:
        job = self.get_job(job_id)
        path = (job.clips_dir / Path(filename).name).resolve()
        root = job.clips_dir.resolve()
        if root not in path.parents or path.suffix.lower() not in {".mp4", ".srt", ".ass", ".json"}:
            raise FileNotFoundError(filename)
        if not path.exists():
            raise FileNotFoundError(filename)
        return path

    def download_path(self, job_id: str, kind: str) -> Path:
        job = self.get_job(job_id)
        table = {
            "json": job.segments_path,
            "segments": job.segments_path,
            "srt": job.srt_path,
            "ass": job.ass_path,
            "mp4": job.output_path,
            "transcript": job.raw_transcript_path,
        }
        if kind not in table:
            raise KeyError(f"Unsupported download kind: {kind}")
        path = table[kind]
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._queue_condition:
                while self._queue_paused:
                    self._queue_condition.wait()
            self._active_jobs.add(job_id)
            try:
                self._process_job(self.get_job(job_id))
            except Exception:
                # 排队中被删除的任务(get_job 抛 KeyError)或任何内部意外
                # 都不得杀死唯一的 worker 线程,否则后续任务永远停在 queued。
                logger.exception("Job worker failed on job %s", job_id)
            finally:
                self._active_jobs.discard(job_id)
                self._queue.task_done()

    def _load_existing_jobs(self) -> None:
        for path in sorted(self.runs_dir.glob("*/job.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = JobRecord.from_dict(data)
                if not job.job_dir:
                    job.job_dir = str(path.parent)
                if job.status in _POSTPROCESS_STATES:
                    # 烧录/翻译/校对由 API 触发,不走 worker 队列;中断时转录与
                    # 用户编辑早已落盘,重新入队会重跑后处理并覆盖 segments.json,
                    # 回 waiting_review 让用户点一下重试即可。
                    info_key, label = _POSTPROCESS_STATES[job.status]
                    if info_key:
                        info = dict(getattr(job, info_key) or {})
                        info["in_progress"] = False
                        setattr(job, info_key, info)
                    job.status = "waiting_review"
                    job.progress = max(job.progress, 0.95)
                    job.error = f"服务重启导致{label}中断，请重试"
                    job.updated_at = time.time()
                    self._save_job(job)
                    self._jobs[job.id] = job
                    continue
                if job.status in RUNNING_STATES:
                    # 断点续传：中断的任务自动重新入队，而不是标 failed
                    job.status = "queued"
                    job.progress = 0.0
                    job.error = None
                    job.updated_at = time.time()
                    self._save_job(job)
                    self._jobs[job.id] = job
                    self.enqueue(job.id)
                    continue
                self._jobs[job.id] = job
            except Exception:
                continue

    def _prepare_audio_timeline(self, job: JobRecord) -> str:
        """检测音频轨时间戳窟窿（录屏丢帧），必要时重抽修复时间轴。

        返回实际供转录/分离使用的音频路径；修复后的时间轴与原视频
        播放时间轴对齐，烧录出的字幕不会漂移。
        """
        try:
            holes = analyze_audio_timeline(job.input_path)
        except Exception:
            holes = None
        if not holes or total_hole_seconds(holes) < HOLE_FIX_THRESHOLD:
            return str(job.input_path)
        fixed_path = Path(job.job_dir) / "audio_fixed.wav"
        if extract_timeline_fixed_audio(job.input_path, fixed_path):
            self.job_log(
                job,
                f"检测到音频时间轴窟窿 {total_hole_seconds(holes):.2f}s"
                f"（{len(holes)} 处，疑似录制丢帧），已补静音修复时间轴",
            )
            return str(fixed_path)
        self.job_log(
            job,
            "音频时间轴窟窿修复失败，按原始音频继续转录",
            level=logging.WARNING,
        )
        return str(job.input_path)

    def _process_job(self, job: JobRecord) -> None:
        # 提前绑定：except 分支里要引用（下载阶段取消时 try 内还没执行到赋值）
        vocals_path = None
        vocals_future = None
        vocals_executor = None
        try:
            if job.source == "url" and job.source_url and not Path(job.input_path).exists():
                self._download_phase(job)

            # 录屏文件音频轨可能有 PTS 窟窿（丢帧），会让整条字幕时间轴前移。
            # 转录/人声分离/说话人标记统一改吃修复后的音频，保证时间轴一致。
            audio_path = self._prepare_audio_timeline(job)

            # 缺口恢复/无 VAD 重试会对小片段从 0.1 重新计进度，导致进度条
            # 反复跳回开头；转录阶段进度单调只升不降。
            last_transcribe_progress = 0.0

            def update(status: str, progress: float | None, generated_tokens: int | None = None) -> None:
                nonlocal last_transcribe_progress
                self._raise_if_cancelled(job.id)
                if status == "transcribing" and job.generated_tokens is None:
                    job.generated_tokens = 0
                if generated_tokens is not None:
                    job.generated_tokens = generated_tokens
                if status == "transcribing" and progress is not None:
                    progress = max(progress, last_transcribe_progress)
                    last_transcribe_progress = progress
                save = generated_tokens is None or self._should_save_live_progress(job.id)
                self._set_status(job, status, progress if progress is not None else job.progress, save=save)

            # 人工平台字幕优先：URL 任务若有人工 CC，直接当文稿跳过 whisper；
            # auto-CC 仍走转录。force_transcribe 可强制转录。
            caption_segments = self._load_source_captions(job)
            if caption_segments is None:
                self._apply_auto_max_new_tokens(job)
            else:
                self.job_log(job, f"使用人工字幕作为文稿，跳过转录（{len(caption_segments)} 段）")
            self._raise_if_cancelled(job.id)

            # 人声分离与 whisper 并行：demucs(~2GB) + whisper(~3GB) 显存
            # 可同卡共存时，转录期间顺手把 BGM 剥掉，总耗时几乎不变。
            # pyannote 改吃人声轨（BGM 会把说话人嵌入拉偏导致人物融合），
            # whisper 仍跑原始音频（分离伪影会重伤 ASR 召回率）。
            # 显存不宽松时（后台 ollama/浏览器等常驻吃卡）强行并行会触发
            # 驱动 sysmem fallback，两个负载超线性互拖（实测 demucs 单独
            # 63s / 并行 281s）——改走转录后串行，反而更快更稳。
            diarization_enabled = self._resolve_diarization_backend(job.diarization_backend) not in {"none", "off", "disabled"}
            demucs_needed = (
                diarization_enabled
                and job.speaker_count != 0
                and vocal_separator.vocal_separation_available()
            )
            parallel_demucs = demucs_needed and _vram_allows_parallel_demucs()
            if parallel_demucs:
                from concurrent.futures import ThreadPoolExecutor

                self.job_log(job, "后台启动 demucs 人声分离（与转录并行）")
                vocals_executor = ThreadPoolExecutor(max_workers=1)
                vocals_future = vocals_executor.submit(
                    vocal_separator.separate_vocals, audio_path, job.job_dir
                )
            elif demucs_needed:
                self.job_log(job, "空闲显存不足，demucs 人声分离改为转录完成后串行执行")

            # 断点续传：检查转录是否已完成（raw_words.json 存在则跳过 whisper）
            checkpoint_words = None
            if caption_segments is None and job.raw_words_path.exists():
                try:
                    _words_data = json.loads(job.raw_words_path.read_text(encoding="utf-8"))
                    checkpoint_words = [(float(s), float(e), str(t)) for s, e, t in _words_data]
                except Exception:
                    checkpoint_words = None

            if caption_segments is not None:
                result = None
                job.generated_tokens = 0
                self._set_status(job, "postprocessing", 0.85, error=None)
                job.raw_transcript_path.write_text(
                    "\n".join(segment.text for segment in caption_segments), encoding="utf-8"
                )
                segments = caption_segments
            elif checkpoint_words is not None:
                # 断点续传：转录已完成，从 checkpoint 恢复，跳过 whisper
                result = None
                segments = regroup_sentences_from_words(checkpoint_words)
                segments = drop_repeated_hallucinations(segments)
                job.generated_tokens = job.generated_tokens or 0
                self._set_status(job, "postprocessing", 0.85, error=None)
            else:
                result = self.model_runner.transcribe(
                    audio_path,
                    prompt=job.inference_prompt,
                    max_length=job.max_length,
                    max_new_tokens=job.max_new_tokens,
                    decoding=job.decoding,
                    temperature=job.temperature,
                    hotwords=self._effective_hotwords(job),
                    status_callback=update,
                )
                self._raise_if_cancelled(job.id)
                job.generated_tokens = result.generated_tokens
                self.job_log(
                    job,
                    f"转录完成: {len(result.text)} 字符, generated_tokens={result.generated_tokens}, "
                    f"elapsed={result.elapsed_sec:.1f}s",
                )
                self._set_status(job, "postprocessing", 0.85, error=None)
                job.raw_transcript_path.write_text(result.text, encoding="utf-8")
                # 保存 words checkpoint（断点续传用）
                if result.words:
                    job.raw_words_path.write_text(
                        json.dumps([[s, e, t] for s, e, t in result.words], ensure_ascii=False),
                        encoding="utf-8",
                    )
                    segments = regroup_sentences_from_words(result.words)
                else:
                    segments = regroup_sentences(subtitle_segments_from_transcript(result.text, postprocess=False))
                # 重复幻觉过滤：音乐段里同一句"不存在的话"反复出现的模式。
                segments = drop_repeated_hallucinations(segments)
                _apply_transcription_quality(segments, getattr(result, "segment_metrics", None))
            self._raise_if_cancelled(job.id)

            # 串行路径：转录已完成、whisper 显存已让出，现在才启动 demucs
            if demucs_needed and not parallel_demucs and vocals_future is None:
                from concurrent.futures import ThreadPoolExecutor

                self.job_log(job, "开始 demucs 人声分离（串行）")
                vocals_executor = ThreadPoolExecutor(max_workers=1)
                vocals_future = vocals_executor.submit(
                    vocal_separator.separate_vocals, audio_path, job.job_dir
                )

            if vocals_future is not None:
                try:
                    vocals_path = vocals_future.result()
                    self.job_log(job, "demucs 人声分离完成")
                except Exception as exc:
                    vocals_path = None
                    self.job_log(job, f"demucs 人声分离失败，回退原始音频: {exc}", level=logging.WARNING, exc_info=True)
                finally:
                    vocals_executor.shutdown(wait=False)
                # 背景音门控：纯对白音轨（无 BGM）不值得喂分离人声，
                # 丢弃结果直接用原始音频。
                if vocals_path is not None and not vocal_separator.has_background_audio(
                    result.words if result else None, audio_path, job.job_dir
                ):
                    self.job_log(job, "背景音门控判定无 BGM，丢弃人声轨，pyannote 改用原始音频")
                    try:
                        Path(vocals_path).unlink(missing_ok=True)
                    except OSError:
                        pass
                    vocals_path = None

            self._set_status(job, "labeling_speakers", 0.88, error=None)
            self.job_log(
                job,
                "开始说话人标记（diarization 对全音频扫描，期间约 1-2 分钟无进度输出，属正常）",
            )
            resolved_speakers = self._resolve_speaker_count(job.speaker_count)
            segments, speaker_info = label_speakers(
                audio_path,
                segments,
                work_dir=job.job_dir,
                # 未指定说话人数时给 pyannote 自动检测留空间（原值 2 会把
                # 三人及以上素材静默压进两个标签）；显式指定时按指定值钳定。
                max_speakers=resolved_speakers or 4,
                target_speakers=resolved_speakers,
                backend=job.diarization_backend,
                hf_token=self.hf_token,
                pyannote_model=self.pyannote_model,
                device=self.diarization_device,
                words=(result.words if result else checkpoint_words),
                audio_path=vocals_path,
            )
            # 说话人标记可能返回新建的 SubtitleSegment，重新附加 ASR 质量信息，
            # 确保最终写入 segments.json 的对象包含置信度。
            _apply_transcription_quality(segments, getattr(result, "segment_metrics", None))
            if vocals_path is not None:
                try:
                    Path(vocals_path).unlink(missing_ok=True)
                except OSError:
                    pass
            # 时间轴修复音频的使命结束：后续剪辑/烧录用原视频（修复后的
            # 时间轴与视频播放轴对齐，字幕直接可用）。删掉省磁盘。
            if audio_path != str(job.input_path):
                try:
                    Path(audio_path).unlink(missing_ok=True)
                except OSError:
                    pass
            job.speaker_labeling = speaker_info.to_dict()
            self.job_log(job, "说话人标记完成")
            self._write_subtitle_files(job, segments)
            if result is not None:
                job.prompt_len = result.prompt_len
                job.elapsed_sec = result.elapsed_sec
            self._set_status(job, "waiting_review", 0.95, error=None)
            self._progress_save_times.pop(job.id, None)
        except JobCancelled:
            if vocals_executor is not None:
                # 取消路径同样要关 demucs 线程池，否则孤儿线程占着 ~2GB 显存
                # 直到推理自然跑完。正在跑的推理无法中断（demucs 无取消钩子），
                # shutdown(wait=False) 只保证不阻塞取消流程。
                vocals_future.cancel()
                vocals_executor.shutdown(wait=False)
            self._cancelled_jobs.discard(job.id)
            self._progress_save_times.pop(job.id, None)
            self._emit_times.pop(job.id, None)
            if job.id in self._jobs:
                # 用户主动取消：任务与产物保留，落 cancelled 终态。
                # 删除触发的取消 job 已移出 _jobs，不能回写 job.json 复活目录。
                job.status = "cancelled"
                self._touch(job, error=None)
        except Exception as exc:
            if vocals_executor is not None:
                vocals_future.cancel()
                vocals_executor.shutdown(wait=False)
            if job.id in self._cancelled_jobs:
                self._cancelled_jobs.discard(job.id)
                self._progress_save_times.pop(job.id, None)
                return
            self.job_log(job, f"任务失败: {exc}", level=logging.ERROR, exc_info=True)
            self._set_status(job, "failed", 1.0, error=str(exc))
            self._progress_save_times.pop(job.id, None)

    def _download_phase(self, job: JobRecord) -> None:
        from .downloader import download_with_yt_dlp

        self._raise_if_cancelled(job.id)
        self._set_status(job, "downloading", 0.0, error=None)
        cc = job.cookies_config or {}
        cookies_browser = cc.get("browser") or "firefox"
        cookies_file = cc.get("file")

        def on_progress(ratio: float, info: dict[str, Any]) -> None:
            self._raise_if_cancelled(job.id)
            job.download_info = info
            save = self._should_save_live_progress(job.id)
            self._set_status(job, "downloading", ratio * 0.05, error=None, save=save)

        def cancel_check() -> bool:
            return job.id in self._cancelled_jobs or job.id not in self._jobs

        def on_title(raw_title: str) -> None:
            job.media_name = _sanitize_display_name(raw_title)
            self._save_job(job)

        result = None
        try:
            result = download_with_yt_dlp(
                job.source_url,
                job.job_dir,
                cookies_browser=cookies_browser,
                cookies_file=cookies_file,
                progress_callback=on_progress,
                cancel_check=cancel_check,
                title_callback=on_title,
            )
        finally:
            # 上传的 cookies 临时文件(.cookies.txt)是浏览器登录凭据,下载结束立即删除;
            # 用户自备路径不在此列,原样保留。
            if cookies_file and str(cookies_file).endswith(".cookies.txt"):
                try:
                    Path(cookies_file).unlink(missing_ok=True)
                except OSError:
                    logger.debug("cookies temp file cleanup failed: %s", cookies_file, exc_info=True)
        self._raise_if_cancelled(job.id)
        downloaded = result.path
        if result.title:
            job.media_name = _sanitize_display_name(result.title)
        else:
            # 标题比下载先到时 on_title 已写入，这里只在没拿到标题时回退文件名。
            job.media_name = downloaded.name
        job.input_path = str(downloaded)
        job.download_info = {"completed": True, "filename": downloaded.name}
        job.source_subtitles = list(result.subtitles)
        self._save_job(job)

    def _load_source_captions(self, job: JobRecord) -> list[SubtitleSegment] | None:
        """URL 任务带人工平台字幕时返回可作文稿的段落；否则返回 None 走转录。

        只信任人工字幕：auto-CC 无标点、全小写，质量通常不如 whisper，
        所以 auto 字幕不作为文稿来源。
        """
        if job.force_transcribe or job.source != "url" or not job.source_subtitles:
            return None
        from .downloader import pick_best_subtitle

        manual = [s for s in job.source_subtitles if s.get("kind") == "manual"]
        if not manual:
            return None
        picked = pick_best_subtitle(manual)
        if not picked:
            return None
        path = Path(picked["path"])
        if not path.is_file():
            return None
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            segments = clean_source_captions(parse_srt(raw))
        except Exception:
            return None
        if not segments:
            return None
        job.transcript_source = f"captions:{picked.get('kind', 'manual')}:{picked.get('lang', '')}"
        self._save_job(job)
        return segments

    def _apply_auto_max_new_tokens(self, job: JobRecord) -> None:
        """Raise ``max_new_tokens`` to a non-truncating floor based on audio duration.

        Only ever increases the value, so an explicit user override that is already
        large enough is left untouched. If the duration cannot be probed (e.g. a
        non-media file or ffprobe unavailable), the configured value is kept as-is.
        """
        duration = probe_media_duration(job.input_path)
        recommended = recommend_max_new_tokens(duration, max_length=job.max_length)
        if recommended is not None and recommended > job.max_new_tokens:
            job.max_new_tokens = recommended
            self._save_job(job)

    def _render_job(self, job_id: str, style: SubtitleStyle) -> None:
        job = self.get_job(job_id)
        with self._render_lock:
            try:
                self._set_status(job, "rendering", 0.95, error=None)
                segments = [SubtitleSegment.from_dict(item) for item in self.list_segments(job.id)]
                width, height = probe_video_size(job.input_path)
                write_text(job.ass_path, export_ass(self._export_segments_for_job(job, segments), style=style, video_width=width, video_height=height))
                def on_render_progress(ratio: float) -> None:
                    progress = max(job.progress, 0.95 + max(0.0, min(1.0, ratio)) * 0.049)
                    self._set_status(
                        job,
                        "rendering",
                        progress,
                        error=None,
                        save=self._should_save_live_progress(job.id),
                    )

                burn_ass_subtitles(
                    job.input_path,
                    job.ass_path,
                    job.output_path,
                    style=style,
                    progress_callback=on_render_progress,
                    cancel_check=lambda: job.id in self._cancelled_jobs or job.id not in self._jobs,
                )
                self._set_status(job, "done", 1.0, error=None)
            except FFmpegCancelled:
                self._cancelled_jobs.discard(job.id)
                job.status = "waiting_review"
                job.progress = 0.95
                self._touch(job, error=None)
                self.job_log(job, "烧录已取消，回到待校对")
            except JobCancelled:
                self._cancelled_jobs.discard(job.id)
                job.status = "waiting_review"
                job.progress = 0.95
                self._touch(job, error=None)
                self.job_log(job, "烧录已取消，回到待校对")
            except Exception as exc:
                self._set_status(job, "waiting_review", 0.95, error=f"Render failed: {exc}")

    def _clip_segments(self, job: JobRecord, *, start: float, end: float) -> list[SubtitleSegment]:
        source = [SubtitleSegment.from_dict(item) for item in self.list_segments(job.id)]
        return rebase_segments_for_clip(source, start=start, end=end)

    def _write_subtitle_files(
        self,
        job: JobRecord,
        segments: list[SubtitleSegment],
        *,
        style: SubtitleStyle | None = None,
    ) -> None:
        write_text(job.segments_path, export_json(segments))
        # 导出字幕给句尾留少量缓冲；工程 JSON 保留 ASR 原始时间，便于继续编辑。
        self._tail_padding_cache.pop(job.id, None)
        export_segments = self._export_segments_for_job(job, segments)
        if style is None:
            style = SubtitleStyle.from_dict(job.subtitle_style) if job.subtitle_style else SubtitleStyle(font_size=48)
        write_text(
            job.srt_path,
            export_srt(export_segments, show_speaker=style.show_speaker, speaker_names=style.speaker_names),
            encoding="utf-8-sig",
        )
        width, height = probe_video_size(job.input_path)
        write_text(
            job.ass_path,
            export_ass(export_segments, style=style, video_width=width, video_height=height),
            encoding="utf-8-sig",
        )
        # 记录应用写出时的 mtime,sync 时据此跳过应用自己写的文件。
        job.subtitle_file_stamps = {
            "srt": job.srt_path.stat().st_mtime,
            "ass": job.ass_path.stat().st_mtime,
        }
        self._save_job(job)

    def _maybe_sync_segments_from_subtitle_files(self, job: JobRecord, *, force: bool = False) -> dict[str, Any] | None:
        source_path, source_kind = self._select_subtitle_source(job, force=force)
        if source_path is None:
            return None
        source_mtime = source_path.stat().st_mtime
        if not force:
            # 应用自己写出的文件(mtime 与记录的戳一致)不是外部编辑,跳过;
            # 否则 srt 写在 segments.json 之后,每次 list_segments 都会误触发反向同步。
            if job.subtitle_file_stamps.get(source_kind) == source_mtime:
                return None
            if job.segments_path.exists() and job.segments_path.stat().st_mtime >= source_mtime:
                return None
        text = source_path.read_text(encoding="utf-8-sig" if source_kind == "srt" else "utf-8")
        segments = parse_srt(text) if source_kind == "srt" else parse_ass(text)
        if not segments and text.strip():
            return None
        style = SubtitleStyle.from_dict(job.subtitle_style) if job.subtitle_style else SubtitleStyle(font_size=48)
        self._write_subtitle_files(job, segments, style=style)
        write_text(job.segments_path, export_json(segments))
        self._touch(job, error=None)
        return {"source": source_kind, "count": len(segments)}

    def _select_subtitle_source(self, job: JobRecord, *, force: bool = False) -> tuple[Path | None, str | None]:
        if job.srt_path.exists():
            return job.srt_path, "srt"
        if job.ass_path.exists():
            return job.ass_path, "ass"
        if force and job.segments_path.exists():
            return job.segments_path, "json"
        return None, None

    def _set_status(
        self,
        job: JobRecord,
        status: str,
        progress: float,
        *,
        error: str | None = None,
        save: bool = True,
    ) -> None:
        self._raise_if_cancelled(job.id)
        previous = job.status
        job.status = status
        job.progress = max(0.0, min(1.0, progress))
        if status != previous:
            # Persist coarse phase durations for queue diagnostics.
            phase_started = getattr(self, "_phase_starts", {}).get(job.id)
            if phase_started is not None:
                job.phase_times[previous] = round(max(0.0, time.time() - phase_started), 3)
            self._phase_starts[job.id] = time.time()
            detail = f" -> {status} ({job.progress:.0%})" + (f" error={error}" if error else "")
            self.job_log(job, previous + detail, level=logging.ERROR if error else logging.INFO)
        self._touch(job, error=error, save=save)

    def _resolve_inference_options(
        self,
        *,
        prompt: str | None,
        max_length: int | None,
        max_new_tokens: int | None,
        decoding: str | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        prompt_value = self.prompt if prompt is None or not prompt.strip() else prompt
        max_length_value = self.max_length if max_length is None else int(max_length)
        max_new_tokens_value = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        decoding_value = decoding or self.decoding
        if decoding_value not in {"greedy", "sample"}:
            raise ValueError("decoding must be greedy or sample.")
        if max_length_value <= 0:
            raise ValueError("max_length must be greater than 0.")
        if max_new_tokens_value <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        temperature_value = self.temperature if temperature is None else float(temperature)
        if decoding_value == "greedy":
            temperature_value = None
        else:
            if temperature_value is None:
                temperature_value = 1.0
            if temperature_value <= 0:
                raise ValueError("temperature must be greater than 0.")

        return {
            "prompt": prompt_value,
            "max_length": max_length_value,
            "max_new_tokens": max_new_tokens_value,
            "decoding": decoding_value,
            "temperature": temperature_value,
        }

    def _resolve_speaker_count(self, speaker_count: int | None) -> int | None:
        value = getattr(self, "speaker_count", None) if speaker_count is None else speaker_count
        if value in ("", None):
            return None
        value = int(value)
        if value <= 0:
            return None
        return max(1, min(value, 10))

    def _resolve_diarization_backend(self, diarization_backend: str | None) -> str:
        value = (diarization_backend or self.diarization_backend or "auto").lower()
        if value in {"auto", "pyannote", "cluster", "none", "off", "disabled"}:
            return "none" if value in {"off", "disabled"} else value
        return "auto"

    def _touch(self, job: JobRecord, *, error: str | None = None, save: bool = True) -> None:
        job.error = error
        job.updated_at = time.time()
        if save:
            self._save_job(job)
        self._emit_job_event(job, force=self._emit_status.get(job.id) != job.status)
        self._emit_status[job.id] = job.status

    def _save_job(self, job: JobRecord) -> None:
        job.job_path.write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _should_save_live_progress(self, job_id: str) -> bool:
        now = time.time()
        last_saved = self._progress_save_times.get(job_id, 0.0)
        if now - last_saved < 0.5:
            return False
        self._progress_save_times[job_id] = now
        return True

    def _runner_backend(self) -> str:
        if hasattr(self.model_runner, "runtime_info"):
            try:
                return str(self.model_runner.runtime_info().get("backend") or "")
            except Exception:
                return ""
        return ""

    def _raise_if_cancelled(self, job_id: str) -> None:
        if job_id in self._cancelled_jobs or job_id not in self._jobs:
            raise JobCancelled(job_id)


_DISPLAY_NAME_ILLEGAL_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_display_name(name: str) -> str:
    """URL 任务的展示名：剔除文件系统非法字符并截断，仅用于界面显示，不影响磁盘文件名。"""
    cleaned = _DISPLAY_NAME_ILLEGAL_RE.sub(" ", str(name))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return cleaned[:80]


def _safe_clip_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
    safe = safe.strip("._")
    return safe[:80] or f"clip_{uuid.uuid4().hex[:8]}"
