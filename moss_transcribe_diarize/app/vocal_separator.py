"""Demucs 人声分离：为说话人分离提供去 BGM 的人声轨。

whisper 仍跑原始音频（人声分离的伪影会重伤 ASR 召回率），
仅 pyannote 改吃分离出的人声——背景音乐会把说话人嵌入拉偏，
导致短插话被并进相邻说话人的 turn（快速对话区"人物融合"的主因）。

分离在 whisper 转录期间并行跑（两者显存合计约 5GB，8GB 卡放得下），
对总耗时几乎零增加。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .ffmpeg import detect_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMUCS_WEIGHTS = PROJECT_ROOT / "models" / "demucs-htdemucs" / "955717e8.safetensors"

# 词间隙响度 / 语音响度 超过该比例视为存在持续背景音（音乐/吟唱）。
# 纯人语音轨的间隙接近底噪（<0.05），持续 BGM 通常 >0.2。
BACKGROUND_RATIO_THRESHOLD = 0.12


def vocal_separation_available() -> bool:
    """人声分离可用：权重已下载、demucs 已安装且 CUDA 可用。

    CPU 跑 demucs 慢一个数量级，不值得自动启用。
    """
    if not DEMUCS_WEIGHTS.is_file():
        return False
    try:
        import torch  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def separate_vocals(
    media_path: str | Path,
    work_dir: str | Path,
    *,
    device: str = "auto",
) -> Path | None:
    """把媒体的人声分离成 16k 单声道 wav，返回路径；失败返回 None。"""
    if not vocal_separation_available():
        return None
    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.hf import load_safetensors_model

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        out_dir = Path(work_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "vocals_16k.wav"
        tmp44k = out_dir / "vocals_src_44k.wav"

        ffmpeg = detect_ffmpeg().ffmpeg
        subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(media_path), "-vn", "-ac", "2", "-ar", "44100", str(tmp44k), "-y"],
            check=True,
            capture_output=True,
        )
        try:
            model = load_safetensors_model(str(DEMUCS_WEIGHTS)).eval().to(device)
            mix, sr = torchaudio.load(str(tmp44k))  # (2, T) 44100
            with torch.no_grad():
                estimates = apply_model(
                    model, mix.unsqueeze(0), device=device, split=True, overlap=0.25, progress=False
                )[0]  # (sources, 2, T)
            vocals = estimates[model.sources.index("vocals")]
            mono = vocals.mean(dim=0, keepdim=True)
            res = torchaudio.functional.resample(mono, sr, 16000)
            torchaudio.save(str(out), res.cpu(), 16000, encoding="PCM_S", bits_per_sample=16)
        finally:
            tmp44k.unlink(missing_ok=True)
        return out if out.is_file() else None
    except Exception:
        return None


def has_background_audio(
    words: list[tuple[float, float, str]] | None,
    media_path: str | Path,
    work_dir: str | Path,
) -> bool:
    """用 whisper 词表判断有无持续背景音：语音间隙的响度是否显著。

    有 BGM 时词间隙仍是响的（音乐在响）；纯对白音轨间隙接近底噪。
    无词时间表时保守返回 True（宁可白跑一次分离）。
    """
    try:
        import numpy as np
        import torchaudio
    except Exception:
        return True
    if not words or len(words) < 8:
        return True

    spans = sorted((float(w[0]), float(w[1])) for w in words if str(w[2]).strip())
    gaps = []
    for i in range(1, len(spans)):
        gap = spans[i][0] - spans[i - 1][1]
        if gap >= 0.5:
            gaps.append((spans[i - 1][1], spans[i][0]))
    if len(gaps) < 3:
        # 间隙样本太少（连续说话），无从判断，保守认为有背景音。
        return True

    audio_path: Path | None = None
    try:
        try:
            ffmpeg = detect_ffmpeg().ffmpeg
            out_dir = Path(work_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            audio_path = out_dir / "bgdetect_16k.wav"
            subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(media_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path), "-y"],
                check=True,
                capture_output=True,
            )
            waveform, sr = torchaudio.load(str(audio_path))
        except Exception:
            return True
        data = waveform.mean(dim=0).to("cpu").numpy()
        total = len(data)

        def rms(start: float, end: float) -> float:
            a, b = int(start * sr), min(int(end * sr), total)
            if b - a < int(0.2 * sr):
                return 0.0
            chunk = data[a:b]
            return float(np.sqrt(np.mean(chunk * chunk)))

        speech_rms = sorted(rms(s, e) for s, e in spans if e - s >= 0.2)
        gap_rms = sorted(rms(s, e) for s, e in gaps)
        if not speech_rms or not gap_rms:
            return True
        speech_mid = speech_rms[len(speech_rms) // 2]
        gap_mid = gap_rms[len(gap_rms) // 2]
        if speech_mid <= 1e-6:
            return True
        return (gap_mid / speech_mid) > BACKGROUND_RATIO_THRESHOLD
    except Exception:
        return True
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
