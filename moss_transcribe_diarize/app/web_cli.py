from __future__ import annotations

import argparse
from pathlib import Path

from moss_transcribe_diarize.defaults import DEFAULT_PROMPT

from .cli import DEFAULT_MODEL
from .server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local DieShang Workbench web app.")
    parser.add_argument("--backend", choices=["whisper", "hf", "vllm"], default="whisper")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--vllm-base-url", default=None, help="OpenAI-compatible vLLM base URL, e.g. http://127.0.0.1:8000/v1.")
    parser.add_argument("--vllm-model", default=None, help="vLLM served model name. Defaults to --model.")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-timeout", type=float, default=600.0)
    parser.add_argument("--translator-base-url", default=None, help="OpenAI-compatible chat API for subtitle translation.")
    parser.add_argument("--translator-model", default=None, help="Translation model name. Defaults to --vllm-model or local.")
    parser.add_argument("--translator-api-key", default=None)
    parser.add_argument("--translator-timeout", type=float, default=None)
    parser.add_argument("--translator-provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--translator-protected-terms", default="")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
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
    app = create_app(
        model_path=Path(args.model).expanduser(),
        runs_dir=Path(args.runs_dir).expanduser(),
        device=args.device,
        dtype=args.dtype,
        prompt=args.prompt,
        max_length=args.max_len,
        max_new_tokens=args.max_new_tokens,
        decoding=args.decoding,
        temperature=args.temperature,
        backend=args.backend,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        vllm_api_key=args.vllm_api_key,
        vllm_timeout=args.vllm_timeout,
        translator_base_url=args.translator_base_url,
        translator_model=args.translator_model,
        translator_api_key=args.translator_api_key,
        translator_timeout=args.translator_timeout,
        translator_provider=args.translator_provider,
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
