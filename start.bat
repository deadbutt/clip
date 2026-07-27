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

start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='http://127.0.0.1:7860'; $ready=$url + '/api/runtime'; for ($i=0; $i -lt 90; $i++) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri $ready -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch { Start-Sleep -Milliseconds 700 } }; Start-Process $url"

".venv\Scripts\mtd-subtitle-web.exe" ^
  --backend whisper ^
  --model small ^
  --host 127.0.0.1 ^
  --port 7860 ^
  --dtype auto ^
  --max-new-tokens 8192

echo.
echo Server stopped.
pause
endlocal
