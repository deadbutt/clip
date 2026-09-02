from __future__ import annotations

import argparse
import logging
from pathlib import Path

from moss_transcribe_diarize.defaults import DEFAULT_PROMPT

from .cli import DEFAULT_MODEL
from .server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local DieShang Workbench web app.")
    parser.add_argument("--backend", choices=["whisper"], default="whisper")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--translator-base-url", default=None, help="OpenAI-compatible chat API for subtitle translation.")
    parser.add_argument("--translator-model", default=None, help="Translation model name or local OPUS-MT CTranslate2 directory.")
    parser.add_argument("--translator-api-key", default=None)
    parser.add_argument("--translator-timeout", type=float, default=None)
    parser.add_argument("--translator-provider", choices=["openai", "ollama", "opus-mt"], default="openai")
    parser.add_argument("--translator-tokenizer-dir", default="models/opus-mt-en-zh")
    parser.add_argument("--translator-device", default="auto")
    parser.add_argument("--translator-compute-type", default="auto")
    parser.add_argument("--translator-protected-terms", default="")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--language", default=None, help="Optional Whisper language code, e.g. en, zh, ja. Skips auto-detection when set.")
    parser.add_argument("--beam-size", type=int, default=5, help="Whisper beam size. 3 is a good long-video balance; 1 is only for rough drafts.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-len", type=int, default=131072)
    parser.add_argument("--decoding", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--speaker-count", type=int, default=None, help="Optional target speaker count, clamped to 1-2.")
    parser.add_argument("--diarization-backend", choices=["auto", "pyannote", "cluster", "none"], default="none")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for pyannote gated models. Defaults to HF_TOKEN env var.")
    parser.add_argument("--pyannote-model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--diarization-device", default="auto")
    return parser.parse_args()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install uvicorn to run mtd-subtitle-web.") from exc

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app(
        model_path=Path(args.model).expanduser(),
        runs_dir=Path(args.runs_dir).expanduser(),
        device=args.device,
        dtype=args.dtype,
        language=args.language,
        whisper_beam_size=args.beam_size,
        prompt=args.prompt,
        max_length=args.max_len,
        max_new_tokens=args.max_new_tokens,
        decoding=args.decoding,
        temperature=args.temperature,
        translator_base_url=args.translator_base_url,
        translator_model=args.translator_model,
        translator_api_key=args.translator_api_key,
        translator_timeout=args.translator_timeout,
        translator_provider=args.translator_provider,
        translator_tokenizer_dir=args.translator_tokenizer_dir,
        translator_device=args.translator_device,
        translator_compute_type=args.translator_compute_type,
        translator_protected_terms=tuple(term.strip() for term in args.translator_protected_terms.split(",") if term.strip()),
        speaker_count=args.speaker_count,
        diarization_backend=args.diarization_backend,
        hf_token=args.hf_token,
        pyannote_model=args.pyannote_model,
        diarization_device=args.diarization_device,
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
