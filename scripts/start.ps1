# Audicop launcher (Windows, PowerShell). Run from a terminal:
#   .\scripts\start.ps1
# If PowerShell blocks .ps1 files (corporate "AllSigned"/RemoteSigned policy,
# or files marked as downloaded via OneDrive), just double-click
# `scripts\start.cmd` instead -- it runs this same launcher in a way the
# execution policy does not restrict.
#
# One-step setup: auto-installs `uv` if missing, syncs deps, starts the server.
# Auto-detects an NVIDIA GPU and pulls in the CUDA libs when present.
# No Docker, no Node -- a single local uvicorn process serves API + frontend.

$ErrorActionPreference = "Stop"

# Resolve the repo root robustly. $PSScriptRoot is set when run as a file
# (`-File scripts\start.ps1`); when start.cmd runs us inline to dodge the
# execution policy, it passes the scripts dir via $env:AUDICOP_SCRIPT_DIR.
$ScriptDir =
    if ($PSScriptRoot) { $PSScriptRoot }
    elseif ($env:AUDICOP_SCRIPT_DIR) { ($env:AUDICOP_SCRIPT_DIR).TrimEnd('\') }
    else { Join-Path (Get-Location) "scripts" }
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot
$Port = if ($env:PORT) { $env:PORT } else { "8000" }
$Url = "http://localhost:$Port"

# Warn if running from cloud-synced storage (OneDrive, Dropbox, Google Drive…).
# It's the usual culprit behind setup failures: scripts get blocked (the
# "downloaded from internet" mark), and the sync client churns/locks the .venv
# while uv creates it. A plain local path avoids all of it.
$cloudHit = $null
$rootLower = $RepoRoot.ToLower()
foreach ($m in @("onedrive", "dropbox", "google drive", "googledrive", "\my drive\",
                 "nextcloud", "icloud", "creative cloud", "pcloud", "box sync")) {
    if ($rootLower.Contains($m)) { $cloudHit = $m.Trim('\'); break }
}
foreach ($v in @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)) {
    if (-not $cloudHit -and $v -and $rootLower.StartsWith($v.ToLower())) { $cloudHit = "OneDrive" }
}
if ($cloudHit) {
    Write-Host ""
    Write-Host "ADVERTENCIA: Audicop esta en una carpeta sincronizada a la nube ($cloudHit)." -ForegroundColor Yellow
    Write-Host "  Esto suele causar fallos: PowerShell bloquea el script y la creacion del" -ForegroundColor Yellow
    Write-Host "  entorno (.venv) falla o va lenta por la sincronizacion." -ForegroundColor Yellow
    Write-Host "  Recomendado: clona el proyecto en una ruta local, p.ej. C:\dev\audicop." -ForegroundColor Yellow
    Write-Host ""
}

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

# Be patient on slow / corporate networks: some wheels (av, ctranslate2, the
# CUDA libs) are tens to hundreds of MB and uv's default 30 s per-download
# timeout is easy to exceed behind a proxy.
if (-not $env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT = "300" }

# Detect the GPU once: it decides both the CUDA extra and which llama.cpp wheel.
$HasGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)

# `--inexact` so the launcher-managed llama-cpp-python (installed just below,
# deliberately NOT in the lock) survives the sync instead of being pruned.
$SyncArgs = @("sync", "--inexact")
if ($HasGpu) {
    Write-Host "==> GPU NVIDIA detectada -- soporte CUDA (cuBLAS + cuDNN + runtime)"
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

# Local, on-device AI (llama.cpp). No universal wheel exists -- CPU and CUDA are
# different builds on a separate index -- so install the one matching this
# machine. Idempotent: uv audits and skips once present. Non-fatal on failure:
# only the local AI chat needs it; the rest of Audicop runs regardless.
if ($HasGpu) {
    $LlamaIndex = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
    Write-Host "==> Modo local (IA privada): instalando llama-cpp-python (CUDA)"
} else {
    $LlamaIndex = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
    Write-Host "==> Modo local (IA privada): instalando llama-cpp-python (CPU)"
}
# Pinned to the tested release: reproducible installs and no surprise upgrades
# from the wheel index. Bump deliberately (test CPU + CUDA) when upgrading.
& $UvExe @UvArgs pip install --no-build --extra-index-url $LlamaIndex "llama-cpp-python==0.3.32"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ADVERTENCIA: no se pudo instalar el modo local (llama-cpp-python)." -ForegroundColor Yellow
    Write-Host "  El resto de Audicop funciona; el chat IA local no estara disponible." -ForegroundColor Yellow
}

# Open the browser shortly after the server comes up.
Start-Job -ArgumentList $Url -ScriptBlock {
    param($u) Start-Sleep -Seconds 3; Start-Process $u
} | Out-Null

Write-Host "==> Lanzando Audicop en $Url"
# --no-sync: we already synced above; this also stops `uv run` from pruning the
# launcher-installed llama-cpp-python (it's not in the lock).
& $UvExe @UvArgs run --no-sync uvicorn app.main:app --host 127.0.0.1 --port $Port
