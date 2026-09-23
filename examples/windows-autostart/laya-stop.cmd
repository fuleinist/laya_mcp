@echo off
rem Stop the resident Laya engine and free its VRAM.
rem   laya-stop.cmd                          -> stops whatever listens on 8131
rem   set LAYA_PORT=8132 & laya-stop.cmd      -> a different port
setlocal
if "%LAYA_PORT%"=="" (set "PORT=8131") else (set "PORT=%LAYA_PORT%")

set "PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do set "PID=%%P"

if "%PID%"=="" (
    echo No process is listening on port %PORT%.
    exit /b 0
)
echo Stopping PID %PID% on port %PORT% ...
taskkill /PID %PID% /F >nul 2>&1
if errorlevel 1 (echo Failed to stop PID %PID%. & exit /b 1)
echo Stopped. VRAM released.
endlocal