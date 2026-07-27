#!/usr/bin/env bash
# Install a reusable vLLM environment in WSL2. This intentionally does not
# download or copy the old MOSS audio model.
# Windows example:
#   wsl -d Ubuntu -e bash /mnt/d/MOSS-Transcribe-Diarize/setup_wsl_vllm.sh
set -euo pipefail

VENV_DIR="${VLLM_VENV_DIR:-$HOME/.moss-vllm}"
VLLM_INDEX="${VLLM_INDEX:-https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129}"

echo "[1/3] Installing uv if needed..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv install failed. Check your network and retry."; exit 1; }

echo "[2/3] Creating venv: $VENV_DIR"
uv venv --python 3.12 --clear "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/3] Installing vLLM..."
uv pip install -U vllm \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url "$VLLM_INDEX"

echo ""
echo "Done. Start a text model with:"
echo "  VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ SERVED_MODEL_NAME=qwen2.5-3b-awq bash serve_wsl.sh"
