@echo off
setlocal

cd /d "%~dp0"

REM ============================================================
REM  DieShang Workbench + Ollama text model
REM  Whisper transcription runs locally.
REM  Ollama is only used for subtitle translation and AI clip ranking.
REM ============================================================

set WEB_HOST=127.0.0.1
set WEB_PORT=7860
set PYTHON=.venv\Scripts\python.exe
set WHISPER_DEVICE=cuda
set WHISPER_DTYPE=float16
set WHISPER_LANGUAGE=en
set WHISPER_MODEL=medium
if exist "models\faster-whisper-medium\config.json" set WHISPER_MODEL=models\faster-whisper-medium
set WHISPER_BEAM_SIZE=3
set HF_HUB_ETAG_TIMEOUT=300
set HF_HUB_DOWNLOAD_TIMEOUT=1800
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=qwen:latest
set PROTECTED_TERMS=Twitter,Twitch,OBS,Vidal,Nero

echo [1/3] Checking Windows virtual environment...
if not exist "%PYTHON%" (
  echo [ERROR] Missing %PYTHON%
  echo         Please run: uv venv --python 3.12 .venv ^&^& uv pip install -e ".[torch-runtime]"
  pause
  exit /b 1
)

echo [2/3] Checking Ollama...
where ollama.exe >nul 2>&1 || ( echo [ERROR] ollama.exe not found. & pause & exit /b 1 )
ollama list >nul 2>&1 || (
  echo [ERROR] Ollama is not responding. Please start Ollama first.
  pause
  exit /b 1
)

echo [3/3] Starting DieShang Workbench with Ollama model %OLLAMA_MODEL%...
echo       Translation API: %OLLAMA_BASE_URL%/api/chat
echo       The browser will open after the service is ready.
echo.

start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='http://127.0.0.1:%WEB_PORT%'; $ready=$url + '/api/runtime'; for ($i=0; $i -lt 90; $i++) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri $ready -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch { Start-Sleep -Milliseconds 700 } }; Start-Process $url"

"%PYTHON%" -m moss_transcribe_diarize.app.web_cli ^
  --backend whisper ^
  --model %WHISPER_MODEL% ^
  --host %WEB_HOST% ^
  --port %WEB_PORT% ^
  --device %WHISPER_DEVICE% ^
  --dtype %WHISPER_DTYPE% ^
  --language %WHISPER_LANGUAGE% ^
  --beam-size %WHISPER_BEAM_SIZE% ^
  --max-new-tokens 8192 ^
  --translator-provider ollama ^
  --translator-base-url %OLLAMA_BASE_URL% ^
  --translator-model %OLLAMA_MODEL% ^
  --translator-timeout 900 ^
  --translator-protected-terms "%PROTECTED_TERMS%"

echo.
echo Server stopped.
pause
endlocal
