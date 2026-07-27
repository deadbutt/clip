@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM  DieShang Workbench + optional WSL vLLM text server
REM  Whisper transcription runs on Windows.
REM  vLLM runs in WSL2 and is only for downstream text models.
REM ============================================================

set WSL_DISTRO=Ubuntu
set VLLM_HOST=127.0.0.1
set VLLM_PORT=8000
set WEB_HOST=127.0.0.1
set WEB_PORT=7860
set PYTHON=.venv\Scripts\python.exe
set VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
set SERVED_MODEL_NAME=qwen2.5-3b-awq
set MAX_MODEL_LEN=32768
set GPU_MEM_UTIL=0.70

echo [1/5] Checking Windows virtual environment...
if not exist "%PYTHON%" (
  echo [ERROR] Missing %PYTHON%
  echo         Please run: uv venv --python 3.12 .venv ^&^& uv pip install -e ".[torch-runtime]"
  pause
  exit /b 1
)

echo [2/5] Checking WSL2...
where wsl.exe >nul 2>&1 || ( echo [ERROR] wsl.exe not found. & pause & exit /b 1 )
wsl.exe -d %WSL_DISTRO% -e true >nul 2>&1 || ( echo [ERROR] WSL distro %WSL_DISTRO% is unavailable. & pause & exit /b 1 )

echo [3/5] Calculating WSL path...
set "BATDIR=%~dp0"
set "BATDIR=%BATDIR:~0,-1%"
for /f "usebackq delims=" %%i in (`wsl.exe -d %WSL_DISTRO% -e wslpath -u "%BATDIR%"`) do set "WSLDIR=%%i"
echo       Repo WSL path: %WSLDIR%

echo [4/5] Checking WSL vLLM...
wsl.exe -d %WSL_DISTRO% -e bash -lc "test -x ~/.moss-vllm/bin/vllm" >nul 2>&1
if errorlevel 1 (
  echo       Not installed, running setup...
  wsl.exe -d %WSL_DISTRO% -e bash -lc "%WSLDIR%/setup_wsl_vllm.sh"
  if errorlevel 1 ( echo [ERROR] vLLM install failed. & pause & exit /b 1 )
) else (
  echo       Installed, skipping setup.
)

echo [5/5] Starting vLLM in WSL and DieShang Workbench in Windows...
start "whisper-vllm (WSL)" wsl.exe -d %WSL_DISTRO% -e bash -lc "MAX_MODEL_LEN=%MAX_MODEL_LEN% GPU_MEM_UTIL=%GPU_MEM_UTIL% VLLM_MODEL=%VLLM_MODEL% SERVED_MODEL_NAME=%SERVED_MODEL_NAME% %WSLDIR%/serve_wsl.sh"
start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='http://127.0.0.1:%WEB_PORT%'; $ready=$url + '/api/runtime'; for ($i=0; $i -lt 90; $i++) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri $ready -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch { Start-Sleep -Milliseconds 700 } }; Start-Process $url"

"%PYTHON%" -m moss_transcribe_diarize.app.web_cli --backend whisper --model small --host %WEB_HOST% --port %WEB_PORT% --translator-base-url http://%VLLM_HOST%:%VLLM_PORT%/v1 --translator-model %SERVED_MODEL_NAME%

echo.
echo Web app exited. The optional vLLM window may still be running.
pause
endlocal
