"""Full pipeline test: demucs -> whisper (with gap recovery) -> regroup -> diarization (with speaker splitting)."""
import sys, time, json, os, subprocess, tempfile
sys.path.insert(0, ".")
from pathlib import Path

run_dir = Path("runs/ce6665fa0c2f")
input_path = run_dir / "input.mp4"
vocals_path = run_dir / "vocals_16k.wav"

# Step 1: Vocal separation (demucs) - skip if vocals already exist
if vocals_path.exists() and vocals_path.stat().st_size > 100000:
    print("=" * 60)
    print("Step 1: Vocal separation (demucs) - SKIPPED (vocals_16k.wav exists)")
    print("=" * 60)
else:
    print("=" * 60)
    print("Step 1: Vocal separation (demucs)")
    print("=" * 60)
    import torch, torchaudio
    from moss_transcribe_diarize.app.ffmpeg import detect_ffmpeg
    ff = detect_ffmpeg()
    if not ff.ffmpeg or "TRAE" in (ff.ffmpeg or "").upper():
        ff_path = Path("tools/ffmpeg/ffmpeg.exe")
        if ff_path.exists():
            ff = type(ff)(ffmpeg=str(ff_path), ffprobe=str(ff_path.parent / "ffprobe.exe"))
    tmp44k = run_dir / "tmp_44k.wav"
    t0 = time.time()
    subprocess.run(
        [ff.ffmpeg, "-v", "error", "-i", str(input_path), "-vn", "-ac", "2", "-ar", "44100", str(tmp44k), "-y"],
        check=True,
    )
    print(f"  ffmpeg extract: {time.time()-t0:.0f}s -> {tmp44k}")

    from demucs.hf import load_safetensors_model
    from demucs.apply import apply_model
    WEIGHTS = Path("models/demucs-htdemucs/955717e8.safetensors")
    model = load_safetensors_model(str(WEIGHTS)).eval().to("cuda")
    mix, sr = torchaudio.load(str(tmp44k))
    print(f"  audio {mix.shape} sr={sr}, separating ...")
    t0 = time.time()
    with torch.no_grad():
        estimates = apply_model(model, mix.unsqueeze(0), device="cuda", split=True, overlap=0.25, progress=False)[0]
    print(f"  demucs: {time.time()-t0:.0f}s, sources={model.sources}")
    vocals_idx = model.sources.index("vocals")
    vocals = estimates[vocals_idx]
    mono = vocals.mean(dim=0, keepdim=True)
    res = torchaudio.functional.resample(mono, sr, 16000)
    torchaudio.save(str(vocals_path), res.cpu(), 16000, encoding="PCM_S", bits_per_sample=16)
    print(f"  saved: {vocals_path} ({vocals_path.stat().st_size/1e6:.0f}MB)")
    orig = mix.mean(dim=0)
    v = vocals.mean(dim=0)
    print(f"  vocal energy ratio: {float(v.pow(2).mean())/float(orig.pow(2).mean()):.2f}")
    tmp44k.unlink(missing_ok=True)

# Step 2: Transcription (with gap recovery)
print("\n" + "=" * 60)
print("Step 2: Transcription (WhisperRunner with gap recovery)")
print("=" * 60)
from moss_transcribe_diarize.app.whisper_runner import WhisperRunner
runner = WhisperRunner(
    model_path="models/faster-whisper-large-v3-turbo",
    device="auto",
    language="en",
    vad_filter=True,
    condition_on_previous_text=True,
)
def status_cb(stage, progress, seg_count):
    if seg_count is not None:
        print(f"  [{stage}] progress={progress:.2f} segments={seg_count}")
    elif progress is not None:
        print(f"  [{stage}] progress={progress:.2f}")

t0 = time.time()
result = runner.transcribe(str(vocals_path), status_callback=status_cb)
print(f"\nTranscription done in {time.time()-t0:.1f}s")
print(f"Text length: {len(result.text)} chars")
print(f"Words: {len(result.words)} word-level timestamps")

# Step 3: Regroup sentences from words
print("\n" + "=" * 60)
print("Step 3: Regroup sentences from words")
print("=" * 60)
from moss_transcribe_diarize.subtitle.postprocess import regroup_sentences_from_words
segments = regroup_sentences_from_words(result.words)
print(f"Segments: {len(segments)}")

# Step 4: Speaker labeling (with word-level splitting)
print("\n" + "=" * 60)
print("Step 4: Speaker labeling (pyannote + turn-boundary splitting)")
print("=" * 60)
from moss_transcribe_diarize.app.speaker_labeler import label_speakers
t0 = time.time()
segments, spk_info = label_speakers(
    str(input_path),
    segments,
    work_dir=run_dir,
    max_speakers=4,
    words=result.words,
)
print(f"Diarization done in {time.time()-t0:.1f}s")
print(f"Method: {spk_info.method}, speakers: {spk_info.speakers}, segments: {spk_info.segments}")
print(f"Applied: {spk_info.applied}, reason: {spk_info.reason}")

# Step 5: Write output files and show results
print("\n" + "=" * 60)
print("Step 5: Results")
print("=" * 60)

seg_data = [
    {"id": s.id, "start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker, "text": s.text}
    for s in segments
]
with open(run_dir / "segments.json", "w", encoding="utf-8") as f:
    json.dump(seg_data, f, ensure_ascii=False, indent=2)

def write_srt(segments, path):
    lines = []
    for i, seg in enumerate(segments, 1):
        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lines.append(str(i))
        lines.append(f"{fmt(seg.start)} --> {fmt(seg.end)}")
        lines.append(f"{seg.speaker}: {seg.text}" if seg.speaker else seg.text)
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

write_srt(segments, run_dir / "subtitle.srt")

from collections import Counter
spk_dist = Counter(s.speaker for s in segments)
print(f"\nTotal segments: {len(segments)}")
print(f"Speaker distribution: {dict(spk_dist)}")

gaps = []
for i in range(1, len(segments)):
    gap = segments[i].start - segments[i-1].end
    if gap > 3.0:
        gaps.append((segments[i-1].end, segments[i].start, gap))
if gaps:
    print(f"\nGaps > 3s ({len(gaps)}):")
    for g_start, g_end, g_dur in gaps:
        print(f"  {g_start:.1f}s - {g_end:.1f}s ({g_dur:.1f}s)")
else:
    print("\nNo gaps > 3s - gap recovery working!")

print("\n--- First 40 segments ---")
for s in segments[:40]:
    print(f"{s.id}: {s.start:6.1f}-{s.end:6.1f} {s.speaker}: {s.text[:65]}")

if gaps:
    print("\n--- Segments around first gap ---")
    g_start, g_end, _ = gaps[0]
    for s in segments:
        if g_start - 10 < s.start < g_end + 10:
            print(f"{s.id}: {s.start:6.1f}-{s.end:6.1f} {s.speaker}: {s.text[:65]}")

print(f"\nOutput: {run_dir}/subtitle.srt")
print(f"Output: {run_dir}/segments.json")
