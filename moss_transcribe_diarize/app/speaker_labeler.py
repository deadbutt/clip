from __future__ import annotations

import math
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from moss_transcribe_diarize.subtitle import SubtitleItem, SubtitleSegment
from moss_transcribe_diarize.subtitle.postprocess import _ends_sentence

from .ffmpeg import detect_ffmpeg


@dataclass(slots=True)
class SpeakerLabelingInfo:
    enabled: bool
    applied: bool
    method: str
    speakers: int
    segments: int
    reason: str = ""
    fallback: str = ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "method": self.method,
            "speakers": self.speakers,
            "segments": self.segments,
            "reason": self.reason,
            "fallback": self.fallback,
        }


def label_speakers(
    media_path: str | Path,
    segments: Iterable[SubtitleSegment],
    *,
    work_dir: str | Path,
    enabled: bool = True,
    max_speakers: int = 4,
    target_speakers: int | None = None,
    words: list[tuple[float, float, str]] | None = None,
    backend: str = "auto",
    hf_token: str | None = None,
    pyannote_model: str = "pyannote/speaker-diarization-3.1",
    device: str = "auto",
    audio_path: str | Path | None = None,
) -> tuple[list[SubtitleSegment], SpeakerLabelingInfo]:
    """audio_path: 预备好的 16k 音频（如 demucs 分离出的人声）。

    提供时直接使用，跳过从媒体抽取；whisper 侧仍跑原始音频。
    """
    prepared = [segment for segment in segments if segment.text and segment.end > segment.start]
    backend = (backend or "auto").lower()
    if backend in {"off", "none", "disabled"}:
        enabled = False
    if not enabled:
        return prepared, SpeakerLabelingInfo(False, False, backend, 0, len(prepared), "disabled")
    if len(prepared) < 4:
        return prepared, SpeakerLabelingInfo(True, False, backend, 1, len(prepared), "too few segments")
    existing = {segment.speaker for segment in prepared if segment.speaker}
    if len(existing) > 1:
        return prepared, SpeakerLabelingInfo(True, False, backend, len(existing), len(prepared), "speaker labels already exist")

    if backend in {"auto", "pyannote"}:
        try:
            return _label_speakers_pyannote(
                media_path,
                prepared,
                work_dir=work_dir,
                target_speakers=target_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
                model_name=pyannote_model,
                device=device,
                words=words,
                audio_path=audio_path,
            )
        except Exception as exc:
            if backend == "pyannote":
                return prepared, SpeakerLabelingInfo(True, False, "pyannote", 1, len(prepared), str(exc))
            cluster_segments, cluster_info = _label_speakers_cluster(
                media_path,
                prepared,
                work_dir=work_dir,
                max_speakers=max_speakers,
                target_speakers=target_speakers,
            )
            cluster_info.fallback = f"pyannote failed: {exc}"
            return cluster_segments, cluster_info

    if backend != "cluster":
        return prepared, SpeakerLabelingInfo(True, False, backend, 1, len(prepared), f"unsupported backend: {backend}")

    return _label_speakers_cluster(
        media_path,
        prepared,
        work_dir=work_dir,
        max_speakers=max_speakers,
        target_speakers=target_speakers,
    )


def _label_speakers_cluster(
    media_path: str | Path,
    prepared: list[SubtitleSegment],
    *,
    work_dir: str | Path,
    max_speakers: int,
    target_speakers: int | None,
) -> tuple[list[SubtitleSegment], SpeakerLabelingInfo]:
    try:
        sample_rate, audio = _extract_audio(media_path, work_dir=work_dir)
        features, usable_indices = _segment_features(audio, sample_rate, prepared)
        if len(usable_indices) < 4:
            return prepared, SpeakerLabelingInfo(True, False, "audio_cluster", 1, len(prepared), "not enough voiced audio")
        labels = _cluster_features(features, max_speakers=max_speakers, target_speakers=target_speakers)
        if len(set(labels)) <= 1:
            return _with_single_speaker(prepared), SpeakerLabelingInfo(True, True, "audio_cluster", 1, len(prepared), "single speaker estimated")

        output = _apply_cluster_labels(prepared, usable_indices, labels)
        speaker_count = len({segment.speaker for segment in output})
        return output, SpeakerLabelingInfo(True, True, "audio_cluster", speaker_count, len(output))
    except Exception as exc:
        return prepared, SpeakerLabelingInfo(True, False, "audio_cluster", 1, len(prepared), str(exc))


def _patch_get_plda_optional() -> None:
    """让 pyannote 4.x 的 get_plda 容忍 None（仅本进程生效）。

    4.x 的 SpeakerDiarization.__init__ 无条件调用 get_plda(plda, ...)，
    而 get_plda 只接受 PLDA 实例/str/dict，None 直接 TypeError。但 3.1
    管线（clustering: AgglomerativeClustering）运行时不使用 PLDA，
    本地 vendor config 已注入 ``plda: null`` 显式跳过 community-1
    的 PLDA 下载。此处让 None 透传返回 None。
    注意 speaker_diarization.py 是按名字导入 get_plda 的，两处都要替换。
    """
    import pyannote.audio.pipelines.speaker_diarization as sd
    from pyannote.audio.pipelines.utils import getter

    if getattr(sd.get_plda, "_mtd_plda_optional", False):
        return
    original = getter.get_plda

    def get_plda(plda, *args, **kwargs):  # noqa: ANN002, ANN003
        if plda is None:
            return None
        return original(plda, *args, **kwargs)

    get_plda._mtd_plda_optional = True
    getter.get_plda = get_plda
    sd.get_plda = get_plda


def _label_speakers_pyannote(
    media_path: str | Path,
    prepared: list[SubtitleSegment],
    *,
    work_dir: str | Path,
    target_speakers: int | None,
    max_speakers: int,
    hf_token: str | None,
    model_name: str,
    device: str,
    words: list[tuple[float, float, str]] | None = None,
    audio_path: str | Path | None = None,
) -> tuple[list[SubtitleSegment], SpeakerLabelingInfo]:
    # 依赖可用性与第三方兼容层统一在 _load_pipeline 咽喉处处理。
    if audio_path is not None:
        # 调用方预备好的音频（如 demucs 人声），直接使用，不负责清理。
        audio_file = Path(audio_path)
        owned_audio = False
    else:
        audio_file = _extract_audio_file(media_path, work_dir=work_dir, stem="pyannote_16k")
        owned_audio = True
    try:
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        pipeline = _load_pipeline_with_offline_fallback(model_name, token)
        resolved_device = _resolve_torch_device(device)
        if resolved_device is not None:
            pipeline.to(resolved_device)
        options: dict[str, int] = {}
        if target_speakers and target_speakers > 0:
            # 不直接 num_speakers 强切（会把最相似的两个真人簇焊死），
            # 先放宽上限自由聚类，跑完再按嵌入质心把多余簇合并到目标人数。
            options["max_speakers"] = max(int(target_speakers) + 2, int(max_speakers or 0))
        elif max_speakers and max_speakers > 1:
            # 人声轨上两个真声的嵌入相似度会被拉高，天花板=2 恰好卡在聚类树
            # 把两个真人焊成一簇的位置（实测 774s+53s vs 自由分裂 585s+189s+53s）。
            # 放宽到 4 自由聚类，跑完把占比过小的杂簇（TTS/噪声）收编回主簇。
            options["max_speakers"] = max(4, int(max_speakers))
        # pyannote 4.x 直读文件依赖 torchcodec（Windows 上常缺 FFmpeg DLL 而不可用），
        # 这里改用 torchaudio 解码后以官方支持的内存 dict 形态喂给管线。
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_file))
        diarization = pipeline({"waveform": waveform, "sample_rate": int(sample_rate)}, **options)
        turns = _pyannote_turns(diarization)
        if target_speakers and target_speakers > 0:
            found = len({speaker for _, _, speaker in turns})
            if found < int(target_speakers):
                # 自由聚类分裂不足（音色太接近/素材太短）：强制按目标人数重切。
                diarization = pipeline(
                    {"waveform": waveform, "sample_rate": int(sample_rate)},
                    num_speakers=int(target_speakers),
                )
                turns = _pyannote_turns(diarization)
            else:
                turns = _merge_turn_clusters(
                    turns,
                    getattr(diarization, "speaker_embeddings", None),
                    int(target_speakers),
                )
        else:
            # 自由聚类后的杂簇收编：时长占比过小（<10%）且无连续长 turn 的簇
            # 不是真人对话方（弹幕 TTS、噪声残留全是 1~2s 碎片），并回最相似簇。
            # 但真人短登场（如后半段才出现的第三说话人）占比可能恰好 <10%，
            # 只要簇内存在数秒级连续长 turn（≥4s，TTS 弹幕达不到）就保留。
            major = _real_speaker_clusters(turns)
            if 1 <= len(major) < len({speaker for _, _, speaker in turns}):
                turns = _merge_turn_clusters(
                    turns,
                    getattr(diarization, "speaker_embeddings", None),
                    len(major),
                )
        if not turns:
            return prepared, SpeakerLabelingInfo(True, False, "pyannote", 1, len(prepared), "no speaker turns returned")
        output = _assign_turns_to_segments(prepared, turns, words)
        speaker_count = len({segment.speaker for segment in output})
        return output, SpeakerLabelingInfo(True, True, "pyannote", speaker_count, len(output))
    finally:
        if owned_audio:
            try:
                audio_file.unlink()
            except OSError:
                pass


def _ensure_numpy_compat() -> None:
    """pyannote.audio 3.1.x 在 numpy>=2 下引用已移除的 np.NaN，导入前补齐。"""
    import numpy as np

    if not hasattr(np, "NaN"):
        np.NaN = np.nan


def _stub_speechbrain_k2() -> None:
    """中和 speechbrain 的懒加载代理（仅本进程生效）。

    ``import speechbrain`` 会在 sys.modules 里留下多个 LazyModule 别名
    （如 speechbrain.k2_integration -> speechbrain.integrations.k2_fsa）。
    这些代理的目标模块（k2、spacy、flair 等）是未安装的可选依赖，
    而其 ``__getattr__`` 在懒加载失败时抛 ImportError 而非 AttributeError。
    之后 lightning 经 torch.library 注册 fake op 时会用 inspect.getmodule
    扫描整个 sys.modules 并对每项取 ``__file__``，一旦命中这些代理就会
    被拖崩，进而让 pyannote.audio 导入失败。

    解法分两层：
    1. 为 k2_fsa 预注册空壳（说话人分离不使用 k2）；
    2. 给所有 LazyModule 别名实例直接补 ``__file__`` 等属性，让普通属性
       查找命中实例字典、不再触发懒加载（代理的显式 ``import`` 不受影响）。
    """
    import sys
    import types

    target = "speechbrain.integrations.k2_fsa"
    if target not in sys.modules:
        try:
            import speechbrain  # noqa: F401  # 触发别名注册
            import speechbrain.integrations as integrations  # noqa: F401
        except Exception:
            return
        stub = types.ModuleType(target)
        stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules[target] = stub
        parent = sys.modules.get("speechbrain.integrations")
        if parent is not None and "k2_fsa" not in getattr(parent, "__dict__", {}):
            setattr(parent, "k2_fsa", stub)

    try:
        from speechbrain.utils.importutils import LazyModule
    except Exception:
        return
    for mod in list(sys.modules.values()):
        if isinstance(mod, LazyModule):
            d = mod.__dict__
            d.setdefault("__file__", None)
            d.setdefault("__spec__", None)
            d.setdefault("__path__", [])
            d.setdefault("__loader__", None)


def _patch_torch_load_trusted() -> None:
    """pyannote 3.1.x 的旧式 checkpoint 无法在 torch>=2.6 的
    weights_only=True 新默认下反序列化；本地模型文件来自官方仓库的
    鉴权下载，属可信源，此处恢复允许完整 pickle 的加载行为。"""
    import torch

    if getattr(torch.load, "_mtd_trusted_patched", False):
        return
    original = torch.load

    def load(*args, **kwargs):
        # 调用方（如 lightning）常显式传 weights_only=None，torch 会按新默认 True 处理，
        # 这里对"未指定/None"统一改为 False（可信源完整加载）。
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    load._mtd_trusted_patched = True
    torch.load = load


def _patch_hub_use_auth_token() -> None:
    """pyannote 3.1.x 与新版 huggingface_hub 的兼容层（仅本进程生效）：

    1. use_auth_token= -> token= 透明改写；
    2. 若存在本地映射清单（tools/prefetch_pyannote_cache.py 产物），
       命中的 repo/filename 直接返回本地文件路径，彻底跳过网络与缓存解析。
    必须在任何 pyannote 模块被导入之前调用。
    """

    local_manifest: dict[str, dict[str, str]] = {}
    if HUB_LOCAL_MANIFEST.is_file():
        import json

        try:
            local_manifest = json.loads(HUB_LOCAL_MANIFEST.read_text(encoding="utf-8"))
        except Exception as exc:  # 清单损坏不致命，仅退化到在线/缓存路径
            print(f"[speaker_labeler] 本地模型清单读取失败，忽略: {exc}")

    def _wrap(fn):
        if getattr(fn, "_mtd_token_patched", False):
            return fn

        def wrapper(*args, **kwargs):
            # 参数形态兼容：位置参数 (repo_id, filename) 或纯关键字。
            repo_id = None
            filename = kwargs.pop("filename", None)
            if args:
                repo_id = args[0]
                rest = args[1:]
                if rest and filename is None and isinstance(rest[0], str):
                    filename = rest[0]
                    rest = rest[1:]
                if rest:
                    return fn(repo_id, filename, *rest, **kwargs)
            else:
                repo_id = kwargs.get("repo_id")
            files = local_manifest.get(str(repo_id), {})
            if filename and str(filename) in files and Path(files[str(filename)]).is_file():
                return files[str(filename)]
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return fn(repo_id, filename, *args[2:], **kwargs) if args else fn(**kwargs)

        wrapper._mtd_token_patched = True
        return wrapper

    try:
        import huggingface_hub as hub

        new_hf_hub_download = _wrap(hub.hf_hub_download)
        hub.hf_hub_download = new_hf_hub_download
        if hasattr(hub, "file_download"):
            hub.file_download.hf_hub_download = new_hf_hub_download
    except ImportError:
        pass


# 预取脚本产物：自包含模型目录 + hf_hub_download 本地映射清单。
LOCAL_DIARIZATION_DIR = Path(__file__).resolve().parents[2] / "models" / "pyannote-speaker-diarization-local"
HUB_LOCAL_MANIFEST = Path(__file__).resolve().parents[2] / "models" / "pyannote-hub-local.json"


def _load_pipeline(model_name: str, token: str | None):
    """加载顺序：hf_hub_download 本地劫持(清单存在时) > HF 仓库(token 兼容新旧参数名)。

    全部第三方兼容层（numpy/speechbrain 懒加载/hub 参数/torch 权重加载/PLDA 可选化）
    统一在此咽喉应用，保证任何调用路径行为一致。
    """
    try:
        import speechbrain  # noqa: F401 先行导入以便打桩生效

        _stub_speechbrain_k2()
        _ensure_numpy_compat()
        _patch_hub_use_auth_token()
        _patch_torch_load_trusted()
        from pyannote.audio import Pipeline

        _patch_get_plda_optional()
    except ImportError as exc:
        raise RuntimeError("pyannote.audio is not installed. Install the diarization extra first.") from exc

    try:
        return Pipeline.from_pretrained(model_name, token=token)
    except TypeError:
        return Pipeline.from_pretrained(model_name, use_auth_token=token)


def _load_pipeline_with_offline_fallback(model_name: str, token: str | None):
    """在线加载失败（网络受限或新版 hub 与镜像不兼容）时，切离线缓存重试一次。

    模型已由 tools/prefetch_pyannote_cache.py 预取到本地 HF 缓存后，
    离线模式即可完全脱离网络运行。
    """
    try:
        return _load_pipeline(model_name, token)
    except Exception as online_error:
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from huggingface_hub import constants as hub_constants

            hub_constants.HF_HUB_OFFLINE = True
        except ImportError:
            pass
        try:
            return _load_pipeline(model_name, token)
        except Exception as exc:
            raise RuntimeError(
                f"无法加载说话人分离模型 {model_name}（已尝试离线缓存）。"
                f"请先运行 tools/prefetch_pyannote_cache.py 预取模型。原始错误: {exc}"
            ) from online_error


def _extract_audio(media_path: str | Path, *, work_dir: str | Path) -> tuple[int, np.ndarray]:
    wav_path = _extract_audio_file(media_path, work_dir=work_dir, stem="speaker_labeler_16k")
    try:
        with wave.open(str(wav_path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            frames = handle.readframes(handle.getnframes())
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass
    if width != 2:
        raise RuntimeError(f"unsupported wav sample width: {width}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def _extract_audio_file(media_path: str | Path, *, work_dir: str | Path, stem: str) -> Path:
    tools = detect_ffmpeg()
    if not tools.ffmpeg:
        raise RuntimeError("ffmpeg is not available")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / f"{stem}.wav"
    command = [
        tools.ffmpeg,
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout or "").splitlines()[-8:])
        raise RuntimeError(f"audio extraction failed: {tail}")
    return wav_path


def _segment_features(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[SubtitleSegment],
) -> tuple[np.ndarray, list[int]]:
    features: list[np.ndarray] = []
    indices: list[int] = []
    for index, segment in enumerate(segments):
        start = max(0, int((segment.start + 0.04) * sample_rate))
        end = min(len(audio), int((segment.end - 0.04) * sample_rate))
        if end - start < int(0.28 * sample_rate):
            continue
        feature = _audio_feature(audio[start:end], sample_rate)
        if feature is None:
            continue
        features.append(feature)
        indices.append(index)
    if not features:
        return np.empty((0, 0), dtype=np.float32), []
    return np.vstack(features).astype(np.float32), indices


def _audio_feature(chunk: np.ndarray, sample_rate: int) -> np.ndarray | None:
    chunk = chunk.astype(np.float32, copy=False)
    chunk = chunk - float(np.mean(chunk))
    if float(np.sqrt(np.mean(chunk * chunk))) < 0.0025:
        return None

    frame_size = max(256, int(sample_rate * 0.025))
    hop = max(128, int(sample_rate * 0.010))
    if len(chunk) < frame_size:
        return None
    frame_count = 1 + (len(chunk) - frame_size) // hop
    frames = np.lib.stride_tricks.as_strided(
        chunk,
        shape=(frame_count, frame_size),
        strides=(chunk.strides[0] * hop, chunk.strides[0]),
        writeable=False,
    ).copy()
    window = np.hanning(frame_size).astype(np.float32)
    frames *= window
    energy = np.mean(frames * frames, axis=1)
    if float(np.max(energy)) <= 0:
        return None
    voiced = energy >= np.quantile(energy, 0.45)
    frames = frames[voiced]
    energy = energy[voiced]
    if len(frames) < 3:
        return None

    n_fft = 512
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    spectrum += 1e-10
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    total_power = np.sum(spectrum, axis=1)
    centroid = np.sum(spectrum * freqs[None, :], axis=1) / total_power
    bandwidth = np.sqrt(np.sum(spectrum * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / total_power)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    bands = [
        (80, 250),
        (250, 700),
        (700, 1600),
        (1600, 3400),
        (3400, 7600),
    ]
    band_ratios = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_ratios.append(np.mean(np.sum(spectrum[:, mask], axis=1) / total_power))

    f0_values = [_estimate_pitch(frame, sample_rate) for frame in frames[:80]]
    voiced_f0 = np.array([value for value in f0_values if value > 0], dtype=np.float32)
    if len(voiced_f0):
        log_f0 = float(np.mean(np.log(voiced_f0)))
        f0_spread = float(np.std(np.log(voiced_f0)))
        voiced_ratio = float(len(voiced_f0) / len(f0_values))
    else:
        log_f0 = 0.0
        f0_spread = 0.0
        voiced_ratio = 0.0

    values = [
        float(np.mean(np.log1p(energy))),
        float(np.std(np.log1p(energy))),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(bandwidth)),
        float(np.std(bandwidth)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        log_f0,
        f0_spread,
        voiced_ratio,
        *[float(value) for value in band_ratios],
    ]
    return np.array(values, dtype=np.float32)


def _estimate_pitch(frame: np.ndarray, sample_rate: int) -> float:
    frame = frame.astype(np.float32, copy=False)
    frame = frame - float(np.mean(frame))
    if float(np.sqrt(np.mean(frame * frame))) < 0.003:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    if corr[0] <= 0:
        return 0.0
    min_lag = max(1, int(sample_rate / 450))
    max_lag = min(len(corr) - 1, int(sample_rate / 75))
    if max_lag <= min_lag:
        return 0.0
    region = corr[min_lag:max_lag]
    lag = int(np.argmax(region)) + min_lag
    confidence = float(corr[lag] / corr[0])
    if confidence < 0.25:
        return 0.0
    return float(sample_rate / lag)


def _cluster_features(features: np.ndarray, *, max_speakers: int, target_speakers: int | None = None) -> list[int]:
    if len(features) < 4:
        return [0 for _ in range(len(features))]
    normalized = _standardize(features)
    if target_speakers is not None and target_speakers > 1:
        k = min(int(target_speakers), len(features), 8)
        return _kmeans(normalized, k)
    best_labels = [0 for _ in range(len(features))]
    best_score = -1.0
    upper = min(max(1, int(max_speakers)), len(features), 6)
    for k in range(2, upper + 1):
        labels = _kmeans(normalized, k)
        score = _silhouette_score(normalized, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
    if best_score < 0.12:
        return [0 for _ in range(len(features))]
    return best_labels


def _resolve_torch_device(device: str):
    device = (device or "auto").lower()
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return None
    try:
        import torch

        return torch.device(device)
    except Exception:
        return None


def _pyannote_turns(diarization) -> list[tuple[float, float, str]]:
    # pyannote 4.x 返回 DiarizeOutput（含 speaker_diarization 等字段），
    # 3.x 直接返回 Annotation。优先取 exclusive_speaker_diarization：
    # 官方为转写下游设计的无重叠版本，正好匹配字幕逐行归属单说话人的用法。
    annotation = getattr(diarization, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(diarization, "speaker_diarization", diarization)
    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start = float(getattr(turn, "start", 0.0))
        end = float(getattr(turn, "end", start))
        if end > start:
            turns.append((start, end, str(speaker)))
    turns.sort(key=lambda item: (item[0], item[1], item[2]))
    return turns


def _real_speaker_clusters(
    turns: list[tuple[float, float, str]],
    *,
    min_share: float = 0.10,
    min_long_turn: float = 4.0,
) -> list[str]:
    """判定哪些簇是真人对话方，返回应保留的说话人列表。

    两条判据任一命中即保留：
    - 总时长占比 >= min_share（默认 10%）；
    - 存在单个 >= min_long_turn 秒的连续长 turn——真人短登场（如后半段
      才出现的第三说话人）占比可能不足 10%，但连续说上 4 秒以上的
      只能是真人；弹幕 TTS/噪声残留等幽灵簇全是 1~2s 碎片，达不到。
    """
    total = sum(t_end - t_start for t_start, t_end, _ in turns)
    if total <= 0:
        return []
    durations: dict[str, float] = {}
    longest: dict[str, float] = {}
    for t_start, t_end, speaker in turns:
        durations[speaker] = durations.get(speaker, 0.0) + (t_end - t_start)
        longest[speaker] = max(longest.get(speaker, 0.0), t_end - t_start)
    return [
        speaker
        for speaker, dur in durations.items()
        if dur >= min_share * total or longest.get(speaker, 0.0) >= min_long_turn
    ]


def _merge_turn_clusters(
    turns: list[tuple[float, float, str]],
    embeddings,
    target: int,
) -> list[tuple[float, float, str]]:
    """把自由聚类多出来的说话人簇合并到目标人数。

    直接 ``num_speakers=N`` 强切会沿聚类树硬砍一刀：当两个真人簇最相似时
    会被焊死成一簇，反而把孤立的第三声源（弹幕 TTS、片段混入的人声等）
    留作"第二说话人"。实测同一素材强切 790s/40s，自由分裂+合并 652s/193s。

    合并规则：反复取当前时长最小的簇——
    - 若它与某簇质心余弦相似度 > 0.3（同一人被阈值分裂的情形），并入最相似簇；
    - 否则视为孤立声源，并入最大簇，不污染次要说话人的簇。
    """
    labels = sorted({speaker for _, _, speaker in turns})
    if target <= 0 or len(labels) <= target:
        return turns
    durations = {label: 0.0 for label in labels}
    for start, end, speaker in turns:
        # 累加说话时长(end-start)而非绝对 end:turn 在时间轴上的位置
        # 不代表说话量,开场 100s 与 5s 处一句话的绝对 end 完全不同。
        durations[speaker] += max(0.0, end - start)

    def _row(label: str) -> int:
        tail = label.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else labels.index(label)

    matrix = None
    if embeddings is not None:
        try:
            import numpy as np

            arr = np.asarray(embeddings, dtype=float)
            if arr.ndim == 2 and len(arr) == len(labels):
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                matrix = arr / np.maximum(norms, 1e-9)
        except Exception:
            matrix = None

    mapping = {label: label for label in labels}
    while True:
        # 组时长 = 组内原始簇时长之和
        group_dur = {g: sum(d for lbl, d in durations.items() if mapping[lbl] == g) for g in set(mapping.values())}
        groups = sorted(set(mapping.values()), key=lambda g: -group_dur[g])
        if len(groups) <= target:
            break
        smallest = groups[-1]
        largest = groups[0]
        dest = largest
        if matrix is not None:
            members = {g: [lbl for lbl in labels if mapping[lbl] == g] for g in groups}
            best_sim = 0.3
            for g in groups:
                if g == smallest:
                    continue
                sims = [float(matrix[_row(m)] @ matrix[_row(s)]) for m in members[g] for s in members[smallest]]
                if not sims:
                    continue
                sim = sum(sims) / len(sims)
                if sim > best_sim:
                    best_sim, dest = sim, g
        for label in labels:
            if mapping[label] == smallest:
                mapping[label] = dest
    return [(start, end, mapping[speaker]) for start, end, speaker in turns]


def _items_in_range(
    words: list[tuple[float, float, str]] | None,
    start: float,
    end: float,
) -> list[SubtitleItem] | None:
    """取落在 [start, end) 内的词级时间戳；words 缺失时返回 None。"""
    if words is None:
        return None
    items = [
        SubtitleItem(text=str(text), start=float(ws), end=float(we))
        for ws, we, text in words
        if str(text).strip() and we > start and ws < end
    ]
    return items or None


def _assign_turns_to_segments(
    segments: list[SubtitleSegment],
    turns: list[tuple[float, float, str]],
    words: list[tuple[float, float, str]] | None = None,
) -> list[SubtitleSegment]:
    speaker_names: dict[str, str] = {}
    output: list[SubtitleSegment] = []
    previous_source = ""
    seg_counter = 0

    def _map(speaker: str) -> str:
        if speaker not in speaker_names:
            speaker_names[speaker] = f"S{len(speaker_names) + 1:02d}"
        return speaker_names[speaker]

    def _emit(start: float, end: float, text: str, speaker_raw: str) -> None:
        nonlocal seg_counter, previous_source
        if not text.strip() or end - start < 0.15:
            return
        previous_source = speaker_raw
        seg_counter += 1
        output.append(SubtitleSegment(
            id=f"seg_{seg_counter:04d}",
            start=start,
            end=end,
            speaker=_map(speaker_raw),
            text=text.strip(),
            items=_items_in_range(words, start, end),
        ))

    for segment in segments:
        overlaps: dict[str, float] = {}
        for t_start, t_end, t_speaker in turns:
            ov = max(0.0, min(segment.end, t_end) - max(segment.start, t_start))
            if ov > 0:
                overlaps[t_speaker] = overlaps.get(t_speaker, 0.0) + ov
        if len(overlaps) <= 1 or words is None:
            source = max(overlaps.items(), key=lambda x: x[1])[0] if overlaps else (previous_source or (turns[0][2] if turns else "SPEAKER_00"))
            _emit(segment.start, segment.end, segment.text, source)
            continue
        # 段跨多个说话人：按词级时间戳归属说话人。
        # 每个词归给“覆盖该词时间跨度最长”的 turn，相邻同说话人的词
        # 合并成子段。比按 turn 边界硬切更稳：边界骑跨的词按真实重叠归属，
        # 且 "No! Child!" 这类紧接换人的短插话能按词切开、不再整段归错人。
        seg_words = [
            w for w in words
            if w[1] > segment.start and w[0] < segment.end and str(w[2]).strip()
        ]
        if not seg_words:
            source = max(overlaps.items(), key=lambda x: x[1])[0]
            _emit(segment.start, segment.end, segment.text, source)
            continue
        raw_parts: list[list] = []
        for ws, we, wtext in seg_words:
            best_speaker = ""
            best_overlap = 0.0
            for t_start, t_end, t_speaker in turns:
                ov = max(0.0, min(we, t_end) - max(ws, t_start))
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = t_speaker
            if not best_speaker:
                # 词落在所有 turn 之外的缝隙：就近归属最近的 turn。
                center = (ws + we) / 2.0
                best_speaker = min(
                    turns,
                    key=lambda t: min(abs(center - t[0]), abs(center - t[1])),
                )[2]
            if raw_parts and raw_parts[-1][3] == best_speaker:
                raw_parts[-1][1] = max(raw_parts[-1][1], we)
                raw_parts[-1][2].append(wtext)
            else:
                raw_parts.append([ws, we, [wtext], best_speaker])
        # 下刀闸门：换说话人的词序列只在语言学边界处切开——左侧句末标点、
        # 打断破折号（"There was a-"）或真实停顿（≥0.3s）。否则并回前行
        # （标签取时长占比大者），避免 pyannote 簇碎片化把 "Then act |
        # more like a mother." 这类同一句话从词中间剁开。
        gated: list[list] = []
        for part in raw_parts:
            if gated and gated[-1][3] != part[3]:
                left_text = "".join(gated[-1][2]).strip()
                real_pause = part[0] - gated[-1][1] >= 0.3
                boundary = left_text.endswith("-") or _ends_sentence(left_text)
                if not (boundary or real_pause):
                    prev = gated[-1]
                    speaker = prev[3] if (prev[1] - prev[0]) >= (part[1] - part[0]) else part[3]
                    gated[-1] = [prev[0], part[1], prev[2] + part[2], speaker]
                    continue
            gated.append(part)
        raw_parts = gated

        sub_parts: list[list] = [
            [s, e, "".join(toks).strip(), spk]
            for s, e, toks, spk in raw_parts
        ]
        sub_parts = [part for part in sub_parts if part[2]]

        # Merge sub-segments shorter than 1.0s into adjacent segments
        MIN_DUR = 1.0
        i = 0
        while i < len(sub_parts):
            s, e, text, source = sub_parts[i]
            if (e - s) < MIN_DUR and len(sub_parts) > 1:
                # 只与同说话人邻居合并；跨说话人吞并把短插话（"No! Child!"类）
                # 计入对方名下，是快速对话区人物融合的主要来源，宁可保留短行。
                if i < len(sub_parts) - 1:
                    ns, ne, ntext, nsource = sub_parts[i + 1]
                    if nsource == source:
                        sub_parts[i + 1] = [s, ne, text + " " + ntext, source]
                        sub_parts.pop(i)
                        continue
                elif i > 0:
                    ps, pe, ptext, psource = sub_parts[i - 1]
                    if psource == source:
                        sub_parts[i - 1] = [ps, e, ptext + " " + text, source]
                        sub_parts.pop(i)
                        i -= 1
                        continue
            i += 1

        for s, e, text, source in sub_parts:
            _emit(s, e, text, source)
    return output


def _best_turn_speaker(segment: SubtitleSegment, turns: list[tuple[float, float, str]]) -> str:
    scores: dict[str, float] = {}
    midpoint = (segment.start + segment.end) / 2.0
    nearest: tuple[float, str] | None = None
    for start, end, speaker in turns:
        overlap = max(0.0, min(segment.end, end) - max(segment.start, start))
        if overlap > 0:
            scores[speaker] = scores.get(speaker, 0.0) + overlap
        distance = 0.0 if start <= midpoint <= end else min(abs(midpoint - start), abs(midpoint - end))
        if nearest is None or distance < nearest[0]:
            nearest = (distance, speaker)
    if scores:
        return max(scores.items(), key=lambda item: item[1])[0]
    return nearest[1] if nearest is not None else ""


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / np.where(std < 1e-6, 1.0, std)


def _kmeans(features: np.ndarray, k: int, *, iterations: int = 60) -> list[int]:
    centers = _initial_centers(features, k)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(iterations):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centers[cluster] = np.mean(members, axis=0)
    return [int(label) for label in labels]


def _initial_centers(features: np.ndarray, k: int) -> np.ndarray:
    centers = [features[0]]
    while len(centers) < k:
        distances = np.min(
            np.linalg.norm(features[:, None, :] - np.vstack(centers)[None, :, :], axis=2),
            axis=1,
        )
        centers.append(features[int(np.argmax(distances))])
    return np.vstack(centers).astype(np.float32)


def _silhouette_score(features: np.ndarray, labels: list[int]) -> float:
    labels_array = np.array(labels)
    clusters = sorted(set(labels))
    if len(clusters) <= 1:
        return -1.0
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    scores = []
    for index, label in enumerate(labels_array):
        same = labels_array == label
        same[index] = False
        a = float(np.mean(distances[index, same])) if np.any(same) else 0.0
        b = math.inf
        for other in clusters:
            if other == label:
                continue
            other_mask = labels_array == other
            if np.any(other_mask):
                b = min(b, float(np.mean(distances[index, other_mask])))
        if not math.isfinite(b):
            continue
        scores.append((b - a) / max(a, b, 1e-6))
    return float(np.mean(scores)) if scores else -1.0


def _apply_cluster_labels(
    segments: list[SubtitleSegment],
    usable_indices: list[int],
    labels: list[int],
) -> list[SubtitleSegment]:
    assigned: list[int | None] = [None for _ in segments]
    for index, label in zip(usable_indices, labels):
        assigned[index] = int(label)
    for index, label in enumerate(assigned):
        if label is not None:
            continue
        assigned[index] = _nearest_label(index, assigned)
    assigned = _smooth_short_islands(segments, assigned)
    label_names: dict[int, str] = {}
    next_id = 1
    output: list[SubtitleSegment] = []
    for segment, label in zip(segments, assigned):
        label = int(label or 0)
        if label not in label_names:
            label_names[label] = f"S{next_id:02d}"
            next_id += 1
        output.append(
            SubtitleSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                speaker=label_names[label],
                text=segment.text,
                items=segment.items,
            )
        )
    return output


def _nearest_label(index: int, assigned: list[int | None]) -> int:
    for distance in range(1, len(assigned) + 1):
        left = index - distance
        right = index + distance
        if left >= 0 and assigned[left] is not None:
            return int(assigned[left] or 0)
        if right < len(assigned) and assigned[right] is not None:
            return int(assigned[right] or 0)
    return 0


def _smooth_short_islands(segments: list[SubtitleSegment], labels: list[int | None]) -> list[int | None]:
    smoothed = list(labels)
    for index in range(1, len(smoothed) - 1):
        if smoothed[index - 1] == smoothed[index + 1] != smoothed[index]:
            duration = segments[index].end - segments[index].start
            if duration < 1.3:
                smoothed[index] = smoothed[index - 1]
    return smoothed


def _with_single_speaker(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    return [
        SubtitleSegment(
            id=segment.id,
            start=segment.start,
            end=segment.end,
            speaker="S01",
            text=segment.text,
            items=segment.items,
        )
        for segment in segments
    ]
