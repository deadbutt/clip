@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\mtd-subtitle-web.exe" (
  echo [ERROR] Virtual environment is not ready.
  echo Please run:
  echo   uv venv --python 3.12 .venv
  echo   uv pip install -e ".[torch-runtime]"
  pause
  exit /b 1
)

echo Starting DieShang Workbench...
echo The browser will open after the service is ready.
echo Press Ctrl+C here to stop the server.
echo.

set WHISPER_DEVICE=cuda
set WHISPER_DTYPE=float16
set WHISPER_LANGUAGE=en
set WHISPER_MODEL=large-v3-turbo
if exist "models\faster-whisper-large-v3-turbo\config.json" set WHISPER_MODEL=models\faster-whisper-large-v3-turbo
set WHISPER_BEAM_SIZE=5
set HF_HUB_ETAG_TIMEOUT=300
set HF_HUB_DOWNLOAD_TIMEOUT=1800
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
set TRANSLATOR_ARGS=--translator-provider openai
if exist "models\opus-mt-en-zh-ct2-int8\model.bin" if exist "models\opus-mt-en-zh\source.spm" (
  echo Local OPUS-MT translator enabled.
  set TRANSLATOR_ARGS=--translator-provider opus-mt --translator-model models\opus-mt-en-zh-ct2-int8 --translator-tokenizer-dir models\opus-mt-en-zh --translator-device auto --translator-compute-type auto
)

start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='http://127.0.0.1:7860'; $ready=$url + '/api/runtime'; for ($i=0; $i -lt 90; $i++) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri $ready -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch { Start-Sleep -Milliseconds 700 } }; Start-Process $url"

".venv\Scripts\mtd-subtitle-web.exe" ^
  --backend whisper ^
  --model %WHISPER_MODEL% ^
  --host 127.0.0.1 ^
  --port 7860 ^
  --device %WHISPER_DEVICE% ^
  --dtype %WHISPER_DTYPE% ^
  --language %WHISPER_LANGUAGE% ^
  --beam-size %WHISPER_BEAM_SIZE% ^
  --max-new-tokens 8192 ^
  %TRANSLATOR_ARGS%

echo.
echo Server stopped.
pause
endlocal
