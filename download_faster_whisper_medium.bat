@echo off
setlocal

cd /d "%~dp0"

set PYTHON=.venv\Scripts\python.exe
set HF_HUB_ETAG_TIMEOUT=300
set HF_HUB_DOWNLOAD_TIMEOUT=1800

if not exist "%PYTHON%" (
  echo [ERROR] Missing %PYTHON%
  echo         Please run: uv venv --python 3.12 .venv ^&^& uv pip install -e ".[torch-runtime]"
  pause
  exit /b 1
)

if not exist "models" mkdir models

echo Downloading faster-whisper medium model to models\faster-whisper-medium ...
echo This is a one-time download. start_ollama.bat will use it automatically after it completes.
echo.

"%PYTHON%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-medium', local_dir='models/faster-whisper-medium', local_dir_use_symlinks=False, resume_download=True)"

if errorlevel 1 (
  echo.
  echo [ERROR] Download failed. If HuggingFace is slow or blocked, try again with a proxy/VPN and keep this window open until it completes.
  pause
  exit /b 1
)

echo.
echo Done. You can now restart start_ollama.bat.
pause
endlocal
