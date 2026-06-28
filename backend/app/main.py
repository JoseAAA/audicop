"""FastAPI entry point: mounts the API and serves the static frontend.

Run with: ``uvicorn app.main:app`` (the launchers do this for you).
The API lives under ``/api``; everything else is the static frontend in
``frontend/`` at the repo root.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import chat, hardware, record, transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# repo_root/backend/app/main.py -> parents[2] == repo root
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

# Hostnames the local UI can legitimately come from. Used by the CSRF guard.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

app = FastAPI(title="Audicop", version=__version__)


@app.middleware("http")
async def csrf_origin_guard(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject cross-site state-changing requests (CSRF defense in depth).

    Audicop binds to localhost and has no auth, so a malicious web page the
    user happens to visit could otherwise POST to these endpoints (e.g. start
    a recording of the mic). We block any non-safe request whose ``Origin`` is
    a remote site. Same-origin requests (the local UI) send a localhost
    Origin; non-browser tools send none — both are allowed.
    """
    if request.method not in _SAFE_METHODS:
        origin = request.headers.get("origin")
        if origin and urlparse(origin).hostname not in _LOCAL_HOSTS:
            return JSONResponse(
                status_code=403, content={"detail": "Origen no permitido (posible CSRF)."}
            )
    return await call_next(request)


app.include_router(hardware.router)
app.include_router(transcribe.router)
app.include_router(chat.router)
app.include_router(record.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


# Mount the SPA last so /api/* routes take precedence. html=True serves
# index.html at "/".
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
