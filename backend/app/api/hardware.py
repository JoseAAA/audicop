"""Hardware + recommendation endpoint."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from app import prompts
from app.adapters import llm
from app.adapters.hardware import detect_hardware
from app.core import config
from app.services.recommender import recommend

router = APIRouter(prefix="/api", tags=["hardware"])


@router.get("/hardware")
def get_hardware() -> dict[str, object]:
    """Detect hardware, recommend a model, and estimate speed for the UI banner."""
    hw = detect_hardware()
    choice = recommend(hw)
    factor = config.REALTIME_FACTORS.get((choice.model_size, choice.device), 1.0)
    minutes_per_hour = max(1, round(60 / factor)) if factor > 0 else 60
    return {
        "hardware": asdict(hw),
        "recommendation": asdict(choice),
        "capacity": {
            "realtime_factor": factor,
            "minutes_per_hour": minutes_per_hour,
            "max_duration_hours": config.MAX_DURATION_HOURS,
            "max_upload_mb": config.MAX_UPLOAD_MB,
        },
        "options": {
            "languages": list(config.DEFAULT_LANGUAGES),
            "tasks": list(config.DEFAULT_TASKS),
            "model_sizes": list(config.VALID_MODEL_SIZES),
            "compute_types": list(config.VALID_COMPUTE_TYPES),
            "supported_extensions": list(config.SUPPORTED_EXTENSIONS),
        },
        "ai": {
            "provider_labels": dict(llm.PROVIDER_LABELS),
            "models_by_provider": {k: list(v) for k, v in llm.MODELS_BY_PROVIDER.items()},
            "api_key_help": dict(llm.API_KEY_HELP),
            "default_provider": "gemini",
            "quick_actions": [
                {"label": a.label, "prompt": a.prompt} for a in prompts.QUICK_ACTIONS
            ],
        },
    }
