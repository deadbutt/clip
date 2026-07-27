#!/usr/bin/env bash
# Start a general-purpose vLLM server in WSL2.
# Windows example:
#   wsl -d Ubuntu -e bash /mnt/d/MOSS-Transcribe-Diarize/serve_wsl.sh Qwen/Qwen2.5-3B-Instruct-AWQ
set -euo pipefail

VENV_DIR="${VLLM_VENV_DIR:-$HOME/.moss-vllm}"
MODEL="${1:-${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-3b-awq}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

if [ ! -x "$VENV_DIR/bin/vllm" ]; then
  echo "vLLM is not installed. Run setup_wsl_vllm.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# WSL2 stability defaults. Keep these conservative for an 8 GB GPU.
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
# FlashInfer's sampler can JIT-build CUDA code at startup. Most WSL runtime
# installs do not include nvcc, so force the native sampler unless explicitly
# debugging vLLM itself.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Starting vLLM: model=$MODEL served-name=$SERVED_MODEL_NAME max-model-len=$MAX_MODEL_LEN gpu-mem-util=$GPU_MEM_UTIL"
echo "FlashInfer sampler: VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "OpenAI-compatible API: http://127.0.0.1:8000/v1"

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --no-enable-flashinfer-autotune \
  --enforce-eager
