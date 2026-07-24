@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ============================================================
REM  MOSS-Transcribe-Diarize 一键启动 -- vLLM 后端（WSL2）
REM  vLLM 跑在 WSL2 Ubuntu 里（Linux-only），Web 应用跑在 Windows。
REM  WSL2 自动转发 localhost，Windows 直连 http://127.0.0.1:8000/v1。
REM  首次运行会自动在 WSL 里安装 vLLM（约 5-15 分钟），之后直接启动。
REM ============================================================

REM ---------- 可按需修改的配置 ----------
set WSL_DISTRO=Ubuntu
set VLLM_HOST=127.0.0.1
set VLLM_PORT=8000
set WEB_HOST=127.0.0.1
set WEB_PORT=7860
set PYTHON=.venv\Scripts\python.exe
set SERVED_MODEL=moss
REM  8GB 显存建议 32768（约 30 分钟单次上限）；12GB+ 可 65536，24GB+ 可 131072
set MAX_MODEL_LEN=32768
set GPU_MEM_UTIL=0.85
REM --------------------------------------

echo [1/6] 检查 Windows 虚拟环境...
if not exist "%PYTHON%" (
  echo [ERROR] 未找到 %PYTHON%
  echo         请先：python -m venv .venv ^&^& .venv\Scripts\pip install -e ".[torch-runtime]"
  pause
  exit /b 1
)

echo [2/6] 检查 WSL2...
where wsl.exe >nul 2>&1 || ( echo [ERROR] 未找到 wsl.exe，请先启用 WSL2 并安装 Ubuntu。 & pause & exit /b 1 )
wsl.exe -d %WSL_DISTRO% -e true >nul 2>&1 || ( echo [ERROR] WSL 发行版 %WSL_DISTRO% 不可用，请用 wsl --install -d Ubuntu 安装。 & pause & exit /b 1 )

echo [3/6] 计算 WSL 路径...
set "BATDIR=%~dp0"
set "BATDIR=%BATDIR:~0,-1%"
for /f "usebackq delims=" %%i in (`wsl.exe -d %WSL_DISTRO% -e wslpath -u "%BATDIR%"`) do set "WSLDIR=%%i"
echo       仓库 WSL 路径: %WSLDIR%

echo [4/6] 检查 WSL 内 vLLM 是否已安装...
wsl.exe -d %WSL_DISTRO% -e bash -lc "test -x ~/.moss-vllm/bin/vllm" >nul 2>&1
if errorlevel 1 (
  echo       未安装，开始一次性安装（约 5-15 分钟，下载 3-6 GB）...
  wsl.exe -d %WSL_DISTRO% -e bash -lc "%WSLDIR%/setup_wsl_vllm.sh"
  if errorlevel 1 ( echo [ERROR] vLLM 安装失败，请查看上方 WSL 输出。 & pause & exit /b 1 )
) else (
  echo       已安装，跳过。
)

echo [5/6] 启动 vLLM 服务（WSL 新窗口）...
start "mtd-vllm (WSL)" wsl.exe -d %WSL_DISTRO% -e bash -lc "MAX_MODEL_LEN=%MAX_MODEL_LEN% GPU_MEM_UTIL=%GPU_MEM_UTIL% %WSLDIR%/serve_wsl.sh"
echo       vLLM 窗口已打开，正在加载模型（首次可能需 1-3 分钟）...

echo [6/6] 等待 vLLM 就绪...
set TRIES=0
set MAX_TRIES=90
:wait_vllm
powershell -NoProfile -Command "try{(Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 'http://%VLLM_HOST%:%VLLM_PORT%/health').StatusCode|Out-Null;exit 0}catch{exit 1}"
if not errorlevel 1 goto :vllm_ready
set /a TRIES+=1
if !TRIES! GEQ %MAX_TRIES% goto :vllm_timeout
timeout /t 2 /nobreak >nul
goto :wait_vllm

:vllm_ready
echo       vLLM 已就绪。启动 Web 应用...
goto :start_web

:vllm_timeout
echo [WARN] %MAX_TRIES% 次内未检测到 vLLM 健康（可能仍在加载）。
echo        仍启动 Web 应用；请等 vLLM 窗口显示就绪后再提交任务，否则会连接失败。

:start_web
echo       浏览器将自动打开：http://%WEB_HOST%:%WEB_PORT%
echo.
REM  后台轮询 Web 的 /api/runtime，就绪后自动打开浏览器
start "" /min powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='http://%WEB_HOST%:%WEB_PORT%'; $r=$u+'/api/runtime'; for($i=0;$i -lt 90;$i++){try{$x=Invoke-WebRequest -UseBasicParsing -Uri $r -TimeoutSec 1; if($x.StatusCode -eq 200){Start-Process $u; exit 0}}catch{Start-Sleep -Milliseconds 700}}; Start-Process $u"

"%PYTHON%" -m moss_transcribe_diarize.app.web_cli --backend vllm --vllm-base-url http://%VLLM_HOST%:%VLLM_PORT%/v1 --vllm-model %SERVED_MODEL% --host %WEB_HOST% --port %WEB_PORT%

echo.
echo Web 应用已退出。vLLM 服务仍在 WSL 窗口运行，如需关闭请关闭那个窗口。
pause
endlocal
