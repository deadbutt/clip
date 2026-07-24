from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .cli import DEFAULT_MODEL


_VLLM_INSTALL_HINT = (
    "未找到 vLLM CLI，请先按 README 安装 vLLM nightly：\n"
    "  uv pip install -U vllm --torch-backend=auto "
    "--extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129  (CUDA 12)\n"
    "  uv pip install -U vllm --torch-backend=auto "
    "--extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu130  (CUDA 13)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve MOSS-Transcribe-Diarize with vLLM (optimized for long audio)."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model path or repo id. Defaults to the local pretrained dir.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=131072, help="Max sequence length (prompt + output). Matches the model context by default.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Always enabled for this model.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to `vllm serve` after '--'.")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "vllm", "serve", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    extra = [a for a in args.extra if a != "--"]
    if extra:
        cmd.extend(extra)
    return cmd


def main() -> None:
    args = parse_args()
    if shutil.which("vllm") is None:
        print(_VLLM_INSTALL_HINT, file=sys.stderr)
        raise SystemExit(1)
    command = build_command(args)
    print("启动 vLLM 服务：", " ".join(command), flush=True)
    print(
        "服务就绪后，在另一个终端运行 web 应用：\n"
        f"  mtd-subtitle-web --backend vllm --vllm-base-url http://{args.host}:{args.port}/v1",
        flush=True,
    )
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
