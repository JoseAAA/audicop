#!/usr/bin/env bash
# Audicop launcher (Linux / macOS).
# One-step setup: auto-installs `uv` if missing, syncs deps, starts the server.
# Auto-detects an NVIDIA GPU and pulls in the CUDA libs when present.
# No Docker, no Node — a single local uvicorn process serves API + frontend.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# Warn if running from cloud-synced storage (OneDrive, Dropbox, Google Drive,
# iCloud…). It tends to break setup: the sync client churns/locks the .venv
# while uv creates it. A plain local path avoids it.
case "$(printf '%s' "${REPO_ROOT}" | tr '[:upper:]' '[:lower:]')" in
    *onedrive*|*dropbox*|*"google drive"*|*googledrive*|*"/my drive/"*|*nextcloud*|\
    *"/library/mobile documents"*|*pcloud*)
        echo ""
        echo "ADVERTENCIA: Audicop esta en una carpeta sincronizada a la nube."
        echo "  Puede fallar o ir lento al crear el entorno (.venv) por la sincronizacion."
        echo "  Recomendado: clona el proyecto en una ruta local, p.ej. ~/audicop."
        echo ""
        ;;
esac

PORT="${PORT:-8000}"
URL="http://localhost:${PORT}"

resolve_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV=(uv)
        return 0
    fi
    for py in python3 python; do
        if command -v "$py" >/dev/null 2>&1 && "$py" -m uv --version >/dev/null 2>&1; then
            UV=("$py" -m uv)
            return 0
        fi
    done
    return 1
}

if ! resolve_uv; then
    echo "==> uv no encontrado. Instalando con el script oficial de Astral…"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Error: necesito 'curl' o 'wget' para instalar uv automáticamente." >&2
        echo "Instala uv manualmente: https://docs.astral.sh/uv/getting-started/installation/" >&2
        exit 1
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! resolve_uv; then
        echo "Error: uv se instaló pero no está en PATH. Abre una nueva terminal y reintenta." >&2
        exit 1
    fi
fi

echo "==> uv: $("${UV[@]}" --version)"

# Be patient on slow / corporate networks: some wheels (av, ctranslate2, the
# CUDA libs) are tens to hundreds of MB and uv's default 30 s per-download
# timeout is easy to exceed behind a proxy.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

SYNC_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> GPU NVIDIA detectada — instalando soporte CUDA (cuBLAS + cuDNN)"
    SYNC_ARGS+=(--extra cuda)
else
    echo "==> Sin GPU NVIDIA — instalación CPU-only"
fi

echo "==> Sincronizando dependencias (la primera vez tarda; luego es instantáneo)…"
"${UV[@]}" sync "${SYNC_ARGS[@]}"

# Open the browser shortly after the server comes up.
( sleep 2
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "${URL}"
  elif command -v open >/dev/null 2>&1; then open "${URL}"
  fi ) >/dev/null 2>&1 &

echo "==> Lanzando Audicop en ${URL}"
exec "${UV[@]}" run uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
