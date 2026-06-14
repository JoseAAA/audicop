# Audicop launcher (Windows, PowerShell). Run from a terminal:
#   .\scripts\start.ps1
# If PowerShell blocks it ("execution of scripts is disabled"), run once:
#   powershell -ExecutionPolicy Bypass -File scripts\start.ps1
#
# One-step setup: auto-installs `uv` if missing, syncs deps, starts the server.
# Auto-detects an NVIDIA GPU and pulls in the CUDA libs when present.
# No Docker, no Node -- a single local uvicorn process serves API + frontend.

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Port = if ($env:PORT) { $env:PORT } else { "8000" }
$Url = "http://localhost:$Port"

# Resolve how to call uv: prefer `uv`, else `python -m uv`.
$UvExe = $null
$UvArgs = @()
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $UvExe = "uv"
} else {
    foreach ($py in @("python", "python3")) {
        if (Get-Command $py -ErrorAction SilentlyContinue) {
            & $py -m uv --version *> $null 2>&1
            if ($LASTEXITCODE -eq 0) { $UvExe = $py; $UvArgs = @("-m", "uv"); break }
        }
    }
}

if (-not $UvExe) {
    Write-Host "==> uv no encontrado. Instalando con el script oficial de Astral..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (Get-Command uv -ErrorAction SilentlyContinue) { $UvExe = "uv" }
    if (-not $UvExe) {
        Write-Error "uv se instalo pero no esta en PATH. Abre una nueva terminal y reintenta."
        exit 1
    }
}

Write-Host "==> uv: $(& $UvExe @UvArgs --version)"

$SyncArgs = @("sync")
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "==> GPU NVIDIA detectada -- instalando soporte CUDA (cuBLAS + cuDNN)"
    $SyncArgs += @("--extra", "cuda")
} else {
    Write-Host "==> Sin GPU NVIDIA -- instalacion CPU-only"
}

Write-Host "==> Sincronizando dependencias (la primera vez tarda; luego es instantaneo)..."
& $UvExe @UvArgs @SyncArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "No se pudieron instalar las dependencias. Revisa el log de arriba."
    exit 1
}

# Open the browser shortly after the server comes up.
Start-Job -ArgumentList $Url -ScriptBlock {
    param($u) Start-Sleep -Seconds 3; Start-Process $u
} | Out-Null

Write-Host "==> Lanzando Audicop en $Url"
& $UvExe @UvArgs run uvicorn app.main:app --host 127.0.0.1 --port $Port
