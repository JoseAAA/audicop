"""Prompt engineering for the AI analysis/chat over a transcript.

All user-facing prompt text is Spanish (the assistant answers in the
user's language). The system prompt frames the model as an audio-analysis
assistant working from a timestamped transcript, so it can cite the exact
moment (``[MM:SS]``) where something was said.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT: str = (
    "Eres un asistente experto en análisis de audio y vídeo. Trabajas a partir "
    "de una transcripción con marcas de tiempo en formato [MM:SS] o [HH:MM:SS]. "
    "Reglas:\n"
    "- Responde en el mismo idioma en que te escriba la persona.\n"
    "- Cuando menciones algo concreto, cita el momento entre corchetes, p. ej. "
    "[12:34], para que la persona pueda ir a ese punto del audio.\n"
    "- Básate SOLO en la transcripción. Si algo no está, dilo claramente en "
    "lugar de inventarlo.\n"
    "- Sé conciso y directo. Usa listas y encabezados cuando ayuden a escanear.\n"
    "- Si la transcripción parece incompleta o cortada, avísalo."
)


@dataclass(frozen=True, slots=True)
class QuickAction:
    """A one-click analysis preset.

    Attributes:
        label: Button text shown to the user (Spanish).
        prompt: The instruction sent to the model.
    """

    label: str
    prompt: str


QUICK_ACTIONS: tuple[QuickAction, ...] = (
    QuickAction(
        label="📝 Resumen",
        prompt=(
            "Haz un resumen ejecutivo de la transcripción en 5-8 frases. "
            "Empieza por el tema principal y termina con la conclusión o el "
            "siguiente paso si lo hay."
        ),
    ),
    QuickAction(
        label="🔑 Puntos clave",
        prompt=(
            "Lista los puntos clave de la transcripción como viñetas. Junto a "
            "cada punto, cita el momento [MM:SS] donde se trató."
        ),
    ),
    QuickAction(
        label="✅ Tareas y acuerdos",
        prompt=(
            "Extrae todas las tareas, decisiones y acuerdos mencionados. Para "
            "cada uno indica, si se sabe, quién es responsable y el momento "
            "[MM:SS] en que se dijo. Si no hay ninguno, dilo."
        ),
    ),
    QuickAction(
        label="🗒️ Acta de reunión",
        prompt=(
            "Redacta un acta breve: asistentes o voces detectadas (si se "
            "deducen), temas tratados con sus momentos [MM:SS], decisiones y "
            "tareas pendientes."
        ),
    ),
)


def build_context(transcript_timestamped: str, *, language: str, duration_seconds: float) -> str:
    """Assemble the context block prepended to the user's first message.

    Args:
        transcript_timestamped: The transcript rendered as ``[MM:SS] text``.
        language: Detected (or forced) language code of the audio.
        duration_seconds: Audio duration in seconds.

    Returns:
        A Spanish context block embedding the transcript and its metadata.
    """
    minutes = duration_seconds / 60.0
    return (
        "A continuación tienes la transcripción del audio con marcas de tiempo. "
        f"Idioma detectado: {language}. Duración: {minutes:.1f} minutos.\n\n"
        "=== TRANSCRIPCIÓN ===\n"
        f"{transcript_timestamped}\n"
        "=== FIN DE LA TRANSCRIPCIÓN ===\n"
    )
