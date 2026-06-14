@echo off
REM Audicop launcher (Windows).
REM One-step setup: auto-installs `uv` if missing, syncs deps, starts the server.
REM Auto-detects an NVIDIA GPU and pulls in the CUDA libs when present.
REM No Docker, no Node -- a single local uvicorn process serves API + frontend.
setlocal ENABLEDELAYEDEXPANSION

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "REPO_ROOT=%CD%"
if "%PORT%"=="" set "PORT=8000"

call :resolve_uv
if "%UV_CMD%"=="" (
    echo ==^> uv no encontrado. Instalando con el script oficial de Astral...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo.
        echo *** ERROR: la instalacion automatica de uv fallo. ***
        echo Instalalo manualmente: https://docs.astral.sh/uv/getting-started/installation/
        echo.
        pause
        popd
        exit /b 1
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    call :resolve_uv
    if "%UV_CMD%"=="" (
        echo Error: uv se instalo pero no esta en PATH. Abre una nueva terminal y reintenta.
        popd
        exit /b 1
    )
)

echo ==^> uv:
%UV_CMD% --version

set "SYNC_EXTRAS="
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo ==^> GPU NVIDIA detectada -- instalando soporte CUDA ^(cuBLAS + cuDNN^)
    set "SYNC_EXTRAS=--extra cuda"
) else (
    echo ==^> Sin GPU NVIDIA -- instalacion CPU-only
)

echo ==^> Sincronizando dependencias ^(la primera vez tarda 5-10 min^)...
%UV_CMD% sync %SYNC_EXTRAS%
if errorlevel 1 (
    echo.
    echo *** ERROR: no se pudieron instalar las dependencias. ***
    echo Mira el log de arriba; si el problema persiste, abre un issue.
    echo.
    pause
    popd
    exit /b 1
)

echo ==^> Lanzando Audicop en http://localhost:%PORT%
REM Open the browser shortly after the server comes up.
start "" cmd /c "timeout /t 3 >nul & start "" http://localhost:%PORT%"
%UV_CMD% run uvicorn app.main:app --host 127.0.0.1 --port %PORT%
set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%

REM ---------------------------------------------------------------------------
:resolve_uv
set "UV_CMD="
where uv >nul 2>nul
if not errorlevel 1 (
    set "UV_CMD=uv"
    exit /b 0
)
where python >nul 2>nul
if not errorlevel 1 (
    python -m uv --version >nul 2>nul
    if not errorlevel 1 (
        set "UV_CMD=python -m uv"
        exit /b 0
    )
)
exit /b 0
