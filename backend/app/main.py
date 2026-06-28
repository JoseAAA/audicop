"""FastAPI entry point: mounts the API and serves the static frontend.

Run with: ``uvicorn app.main:app`` (the launchers do this for you).
The API lives under ``/api``; everything else is the static frontend in
``frontend/`` at the repo root.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import chat, hardware, record, transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# repo_root/backend/app/main.py -> parents[2] == repo root
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="Audicop", version=__version__)

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
