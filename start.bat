@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\mtd-subtitle-web.exe" (
  echo [ERROR] Virtual environment is not ready.
  echo Please run:
  echo   uv venv --python 3.12 .venv
  echo   uv pip install -e ".[torch-runtime]" --torch-backend=auto
  pause
  exit /b 1
)

if not exist "pretrained\moss-transcribe-diarize" (
  echo [ERROR] Model directory was not found:
  echo   pretrained\moss-transcribe-diarize
  echo Please download the model first.
  pause
  exit /b 1
)

echo Starting MOSS Subtitle Studio...
echo The browser will open after the service is ready.
echo Press Ctrl+C here to stop the server.
echo.

start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='http://127.0.0.1:7860'; $ready=$url + '/api/runtime'; for ($i=0; $i -lt 90; $i++) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri $ready -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch { Start-Sleep -Milliseconds 700 } }; Start-Process $url"

".venv\Scripts\mtd-subtitle-web.exe" ^
  --model "pretrained\moss-transcribe-diarize" ^
  --host 127.0.0.1 ^
  --port 7860 ^
  --device cuda:0 ^
  --dtype bf16 ^
  --max-new-tokens 2048

echo.
echo Server stopped.
pause
