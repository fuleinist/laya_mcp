@echo off
rem ---------------------------------------------------------------------------
rem  Optional: keep a Laya engine resident so non-MCP callers (curl, cron
rem  scripts, the Decision Studio UI) get a warm endpoint.
rem
rem  NOT required for laya-mcp: the MCP server spawns its own `laya daemon`
rem  child, because an MCP stdio server cannot attach to a foreign process.
rem
rem  EDIT THE THREE PATHS BELOW, then place the companion .vbs in your Startup
rem  folder:
rem    %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
rem
rem  Behaviour: idempotent (exits if the port is already served), so running it
rem  by hand or on every login never starts a second engine.
rem  Logs:  logs\launcher.log        - this script's breadcrumbs (never locked)
rem         logs\serve-autostart.log - the engine's stdout, locked while it runs
rem  Test on a spare port with:  set LAYA_PORT=8132
rem ---------------------------------------------------------------------------
setlocal

set "LAYA_HOME=G:\dev\AI\laya"
set "EXE=%LAYA_HOME%\layabin\laya.exe"
set "MODEL=%LAYA_HOME%\laya_multilingual_q8_0.gguf"
set "LOGDIR=%LAYA_HOME%\logs"
set "SERVERLOG=%LOGDIR%\serve-autostart.log"
set "LAUNCHLOG=%LOGDIR%\launcher.log"

if "%LAYA_PORT%"=="" (set "PORT=8131") else (set "PORT=%LAYA_PORT%")

if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1

rem --- idempotency: is anything already listening on the port? ---
rem Breadcrumbs go to LAUNCHLOG, never SERVERLOG: the running engine holds
rem SERVERLOG open for its whole life, so appending there fails on the second
rem run ("being used by another process") - which is exactly the path that runs
rem on every login after the first.
netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [%DATE% %TIME%] port %PORT% already served - exiting without starting a second engine>> "%LAUNCHLOG%"
    exit /b 0
)

if not exist "%EXE%" (
    echo [%DATE% %TIME%] ERROR binary missing: %EXE%>> "%LAUNCHLOG%"
    exit /b 2
)
if not exist "%MODEL%" (
    echo [%DATE% %TIME%] ERROR model missing: %MODEL%>> "%LAUNCHLOG%"
    exit /b 3
)

echo [%DATE% %TIME%] starting engine on port %PORT%>> "%LAUNCHLOG%"

rem Foreground on purpose: this is a long-lived server and its stdout/stderr are
rem the only diagnostics we get. The console is already hidden by the .vbs parent.
"%EXE%" serve "%MODEL%" --port %PORT% --device auto --cuda-graph >> "%SERVERLOG%" 2>&1

echo [%DATE% %TIME%] engine exited on port %PORT% with code %ERRORLEVEL%>> "%LAUNCHLOG%"
endlocal
exit /b %ERRORLEVEL%