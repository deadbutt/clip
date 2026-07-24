#!/usr/bin/env bash
# 在 WSL2 里用 vLLM 启动 MOSS-Transcribe-Diarize 服务。
# Windows 调用：  wsl -d Ubuntu -e bash /mnt/d/MOSS-Transcribe-Diarize/serve_wsl.sh
set -euo pipefail

VENV_DIR="$HOME/.moss-vllm"
MODEL_DIR="$HOME/.moss-model"

if [ ! -x "$VENV_DIR/bin/vllm" ]; then
  echo "vLLM 未安装。请先运行: setup_wsl_vllm.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
  if [ -d "$MODEL_DIR" ]; then
    MODEL="$MODEL_DIR"
  else
    MODEL="OpenMOSS-Team/MOSS-Transcribe-Diarize"
  fi
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

echo "启动 vLLM: model=$MODEL  max-model-len=$MAX_MODEL_LEN  gpu-mem-util=$GPU_MEM_UTIL"
echo "（Windows 侧用 http://127.0.0.1:8000/v1 连接，WSL2 会自动转发 localhost）"
echo "（对外模型名统一为 moss，客户端 --vllm-model moss）"
exec vllm serve "$MODEL" \
  --served-model-name moss \
  --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL"
