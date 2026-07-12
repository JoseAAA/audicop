"""Hardware + recommendation endpoint."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from app import prompts
from app.adapters import local_llm
from app.adapters.hardware import detect_hardware
from app.core import config
from app.services.recommender import recommend, recommend_llm

router = APIRouter(prefix="/api", tags=["hardware"])


@router.get("/hardware")
def get_hardware() -> dict[str, object]:
    """Detect hardware, recommend a model, and estimate speed for the UI banner."""
    hw = detect_hardware()
    choice = recommend(hw)
    # If the local model is already loaded, report IT: measuring free memory
    # now would count what the loaded model itself consumes and wrongly show
    # "not available" for a model that is running fine.
    active = local_llm.get_active()
    active_spec = (
        next((m for m in config.LLM_MODELS.values() if m.filename == active.filename), None)
        if active is not None
        else None
    )
    if active is not None and active_spec is not None:
        local_ai: dict[str, object] = {
            "available": True,
            "label": active_spec.label,
            "model_key": active_spec.key,
            "device": active.device,
            "download_size_mb": active_spec.download_size_mb,
            "cached": True,
            "rationale": "El modelo ya está cargado en memoria y listo.",
        }
    else:
        llm_choice = recommend_llm(hw, gpu_offload=local_llm.supports_gpu_offload())
        local_ai = {
            "available": llm_choice.available,
            "label": llm_choice.label,
            "model_key": llm_choice.model_key,
            "device": llm_choice.device,
            "download_size_mb": llm_choice.download_size_mb,
            "cached": (
                local_llm.is_model_cached(llm_choice.filename) if llm_choice.available else False
            ),
            "rationale": llm_choice.rationale,
        }
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
        # 100% on-device analysis: the local model + the quick actions. No
        # providers, no API keys.
        "ai": {
            "local": local_ai,
            "quick_actions": [
                {"label": a.label, "prompt": a.prompt} for a in prompts.QUICK_ACTIONS
            ],
        },
    }
