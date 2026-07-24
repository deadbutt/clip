@echo off
REM ============================================================
REM  Compact the WSL2 Ubuntu vhdx to reclaim C: drive space.
REM  Double-click this file and approve the UAC prompt.
REM  (ASCII-only on purpose to avoid codepage issues.)
REM ============================================================

REM --- self-elevate to administrator ---
net session >nul 2>&1
if errorlevel 1 (
  echo Requesting administrator privileges...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

set VHDX=C:\Users\30677\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx

if not exist "%VHDX%" (
  echo [ERROR] vhdx not found: %VHDX%
  pause
  exit /b 1
)

echo Before:
dir "%VHDX%" | findstr "ext4.vhdx"

echo.
echo Shutting down WSL...
wsl --shutdown
timeout /t 3 /nobreak >nul

echo Reverting sparse flag (diskpart requires non-sparse)...
wsl --manage Ubuntu --set-sparse false --allow-unsafe >nul 2>&1

echo Ensuring file is not NTFS-compressed...
compact /u "%VHDX%" >nul 2>&1

echo Compacting vhdx (this may take a minute)...
set DPSCRIPT=%TEMP%\compact_vhdx.txt
> "%DPSCRIPT%" echo select vdisk file="%VHDX%"
>> "%DPSCRIPT%" echo attach vdisk readonly
>> "%DPSCRIPT%" echo compact vdisk
>> "%DPSCRIPT%" echo detach vdisk
>> "%DPSCRIPT%" echo exit
diskpart /s "%DPSCRIPT%"
del "%DPSCRIPT%"

echo.
echo After:
dir "%VHDX%" | findstr "ext4.vhdx"

echo.
echo Done. You can close this window.
pause
