#!/usr/bin/env bash
# 一次性安装：在 WSL2 Ubuntu 里装好 vLLM（含 MOSS 模型注册的 nightly）。
# Windows 调用：  wsl -d Ubuntu -e bash /mnt/d/MOSS-Transcribe-Diarize/setup_wsl_vllm.sh
set -euo pipefail

VENV_DIR="$HOME/.moss-vllm"
MODEL_DIR="$HOME/.moss-model"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN_MODEL="$SCRIPT_DIR/pretrained/moss-transcribe-diarize"
VLLM_INDEX="https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129"

echo "[1/4] 安装 uv..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv 安装失败，请检查网络后重试。"; exit 1; }

echo "[2/4] 创建 venv: $VENV_DIR"
uv venv --python 3.12 --clear "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/4] 安装 vLLM（含 MOSS 模型注册 nightly，CUDA 12 / cu129，约 3-6 GB 下载）..."
# 国内用清华 PyPI 镜像做主索引加速；vllm nightly 走 wheels.vllm.ai 额外索引。
# Linux x86_64 的 torch 默认即 CUDA 12 构建，故不再用 --torch-backend=auto（它会走 download.pytorch.org，国内偏慢）。
uv pip install -U vllm \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url "$VLLM_INDEX"

echo "[4/4] 准备模型..."
if [ -d "$MODEL_DIR" ]; then
  echo "  模型已存在于 $MODEL_DIR，跳过。"
elif [ -d "$WIN_MODEL" ]; then
  echo "  从 Windows 挂载复制模型到 WSL 原生文件系统（加载更快）..."
  cp -r "$WIN_MODEL" "$MODEL_DIR"
else
  echo "  未找到本地模型 $WIN_MODEL；启动时将改用 HF 仓库 OpenMOSS-Team/MOSS-Transcribe-Diarize 自动下载。"
fi

echo ""
echo "✓ 安装完成。启动服务请运行: serve_wsl.sh"
