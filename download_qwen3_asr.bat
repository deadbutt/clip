@echo off
setlocal

cd /d "%~dp0"

set PYTHON=.venv\Scripts\python.exe
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
set HF_HUB_ETAG_TIMEOUT=300
set HF_HUB_DOWNLOAD_TIMEOUT=1800

if not exist "%PYTHON%" (
  echo [ERROR] Missing %PYTHON%
  echo         Please run: uv venv --python 3.12 .venv ^&^& uv pip install -e ".[torch-runtime]"
  pause
  exit /b 1
)

if not exist "models" mkdir models

echo Downloading Qwen3-ASR-1.7B model to models\Qwen3-ASR-1.7B ...
echo This is a one-time download (~3.5GB).
echo.

"%PYTHON%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-ASR-1.7B', local_dir='models/Qwen3-ASR-1.7B', resume_download=True)"

if errorlevel 1 (
  echo.
  echo [ERROR] Download failed. If HuggingFace is slow or blocked, try again with a proxy/VPN and keep this window open until it completes.
  pause
  exit /b 1
)

echo.
echo Downloading Qwen3-ForcedAligner-0.6B model to models\Qwen3-ForcedAligner-0.6B ...
echo This is a one-time download (~1.2GB).
echo.

"%PYTHON%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-ForcedAligner-0.6B', local_dir='models/Qwen3-ForcedAligner-0.6B', resume_download=True)"

if errorlevel 1 (
  echo.
  echo [ERROR] Download failed. If HuggingFace is slow or blocked, try again with a proxy/VPN and keep this window open until it completes.
  pause
  exit /b 1
)

echo.
echo Done. Both models are in models\ and can be loaded locally.
pause
endlocal
