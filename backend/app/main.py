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
from app.api import chat, hardware, meetings, record, transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# repo_root/backend/app/main.py -> parents[2] == repo root
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

# Hostnames the local UI can legitimately come from. Used by the CSRF guard
# and the Host guard. "testserver" is Starlette's TestClient default.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", "testserver"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

app = FastAPI(title="Audicop", version=__version__)


@app.middleware("http")
async def host_guard(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject requests whose Host header is not a localhost name.

    Defense against DNS rebinding: a malicious page at ``evil.com`` can point
    that name at ``127.0.0.1`` and then read same-origin responses from this
    server — including ``/api/transcript`` (private meeting content). Those
    requests arrive with ``Host: evil.com``, so validating the Host header
    blocks them for **every** method, GETs included (the Origin-based CSRF
    guard below only covers state-changing requests).
    """
    raw = (request.headers.get("host") or "").strip().lower()
    if raw.startswith("["):  # bracketed IPv6, e.g. "[::1]:8000" or "[::1]"
        host = raw.split("]", 1)[0].lstrip("[")
    elif raw.count(":") == 1:  # "name:port"
        host = raw.rsplit(":", 1)[0]
    else:  # bare name, or unbracketed IPv6 like "::1"
        host = raw
    if host not in _LOCAL_HOSTS:
        return JSONResponse(
            status_code=403, content={"detail": "Host no permitido (posible DNS rebinding)."}
        )
    return await call_next(request)


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


@app.middleware("http")
async def revalidate_frontend(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Make browsers revalidate the static frontend on every load.

    Starlette's StaticFiles sets no explicit ``Cache-Control``, so a browser
    can keep serving a stale ``app.js`` / ``index.html`` from heuristic cache
    after an update — leaving an outdated UI running against a newer backend.
    ``no-cache`` keeps the fast ETag path (304 when unchanged) while forcing a
    revalidation, so a plain refresh always picks up new frontend code. API
    responses (including SSE streams) are left untouched.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.include_router(hardware.router)
app.include_router(transcribe.router)
app.include_router(chat.router)
app.include_router(record.router)
app.include_router(meetings.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


# Mount the SPA last so /api/* routes take precedence. html=True serves
# index.html at "/".
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
