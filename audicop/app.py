"""Streamlit entry point for Audicop.

Kept intentionally thin: it wires the modules together (hardware →
recommender → uploader → audio → transcriber → UI). Real logic lives in
the dedicated modules.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import streamlit as st

from audicop import config, formatting
from audicop.audio import AudioConversionError, cleanup, get_duration_seconds, to_wav_16k
from audicop.hardware import HardwareInfo, detect_hardware
from audicop.recommender import ModelChoice, recommend
from audicop.transcriber import (
    Transcriber,
    TranscriptionError,
    TranscriptSegment,
    is_model_cached,
)
from audicop.ui import (
    MediaInput,
    TranscriptionSettings,
    render_ai_panel,
    render_header,
    render_input_summary,
    render_privacy_footer,
    render_results,
    render_sidebar,
    render_status_banner,
    render_uploader,
)


def _format_seconds(seconds: float) -> str:
    """Format a duration as ``H h M m`` / ``M m S s`` / ``S s``."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} m {sec:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} m"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("audicop.app")


# ---------------------------------------------------------------------------
# Cached helpers — rerun-safe
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_hardware() -> HardwareInfo:
    return detect_hardware()


@st.cache_data(show_spinner=False)
def _cached_recommendation(_hw: HardwareInfo) -> ModelChoice:
    # The leading underscore tells Streamlit not to hash this argument
    # (HardwareInfo is hashable, but we want stability across reruns anyway).
    return recommend(_hw)


@st.cache_resource(show_spinner="Cargando modelo (puede tardar la primera vez)…")
def _get_transcriber(model_size: str, compute_type: str, device: str) -> Transcriber:
    transcriber = Transcriber(
        model_size=model_size,
        compute_type=compute_type,
        device=device,
    )
    transcriber.load()
    return transcriber


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
def _run_transcription(
    *,
    media: MediaInput,
    settings: TranscriptionSettings,
) -> None:
    """End-to-end pipeline for a single media input (uploaded or local)."""
    wav_path: Path | None = None
    try:
        if not is_model_cached(settings.model_size):
            size_mb = config.MODEL_DOWNLOAD_SIZES_MB.get(settings.model_size)
            size_text = f" (~{size_mb} MB)" if size_mb else ""
            st.warning(
                f"📥 Primera vez con `{settings.model_size}`{size_text}. "
                f"Voy a descargar el modelo desde HuggingFace; sólo pasa una vez. "
                f"Las siguientes ejecuciones serán instantáneas."
            )

        with st.status("Preparando audio…", expanded=False) as status:
            status.update(label="Extrayendo audio a 16 kHz mono (vídeos: sólo audio)…")
            wav_path = to_wav_16k(media.path)
            duration = get_duration_seconds(wav_path) or 0.0

            if duration >= config.MAX_DURATION_HOURS * 3600:
                hours = duration / 3600
                st.warning(
                    f"⚠️ Archivo de {hours:.1f} h — excede el límite probado de "
                    f"{config.MAX_DURATION_HOURS:.0f} h. Funcionará igual (Whisper "
                    f"trabaja con ventanas de 30 s) pero puede tardar bastante."
                )
            elif duration >= config.LONG_FILE_THRESHOLD_S:
                est = config.estimate_processing_seconds(
                    duration, settings.model_size, settings.device
                )
                st.info(
                    f"📏 Audio de {duration / 60:.0f} min. Estimado: "
                    f"~{est / 60:.0f} min en `{settings.model_size}`/`{settings.device}`."
                )

            status.update(label="Cargando modelo…")
            transcriber = _get_transcriber(
                settings.model_size, settings.compute_type, settings.device
            )
            status.update(label="Transcribiendo…", state="running")

            segments_iter, info = transcriber.transcribe(
                wav_path,
                language=settings.language,
                task=settings.task,
                vad_filter=settings.vad_filter,
            )
            effective_duration = duration or info.duration or 0.0

            progress_bar = st.progress(0.0, text="0%")
            live_text = st.empty()
            segments: list[TranscriptSegment] = []
            start_ts = time.monotonic()

            for seg in segments_iter:
                segments.append(seg)
                live_text.markdown("\n\n".join(s.text for s in segments[-30:] if s.text))
                if effective_duration > 0:
                    pct = min(seg.end / effective_duration, 1.0)
                    elapsed = time.monotonic() - start_ts
                    label = f"{int(pct * 100)}% · transcurrido {_format_seconds(elapsed)}"
                    if pct >= 0.02 and elapsed > 1.0:
                        eta = max(0.0, elapsed / pct - elapsed)
                        label += f" · queda ~{_format_seconds(eta)}"
                    progress_bar.progress(pct, text=label)

            total_elapsed = time.monotonic() - start_ts
            progress_bar.progress(
                1.0, text=f"100% · completado en {_format_seconds(total_elapsed)}"
            )
            live_text.empty()
            status.update(label="Listo.", state="complete")

        if not formatting.to_plain_text(segments):
            st.warning("No se detectó voz en el archivo.")
            return

        # Persist so the result (and the AI chat over it) survive Streamlit
        # reruns. Starting a new transcription resets the chat context.
        st.session_state["result_segments"] = segments
        st.session_state["result_meta"] = {
            "base": Path(media.name).stem,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": effective_duration,
        }
        st.session_state["chat_history"] = []

    except FileNotFoundError as exc:
        logger.exception("Archivo no encontrado")
        st.error(f"No se encontró el archivo: {exc}")
    except AudioConversionError as exc:
        logger.exception("Error de conversión")
        st.error(f"No se pudo preparar el audio. {exc}")
    except TranscriptionError as exc:
        logger.exception("Error de transcripción")
        st.error(f"Falló la transcripción. {exc}")
    except Exception as exc:  # last-resort guard so the UI never goes blank
        logger.exception("Error inesperado")
        st.error(f"Ha ocurrido un error inesperado: {exc}")
    finally:
        if wav_path is not None:
            cleanup(wav_path.parent)
        # Only delete the source if WE created it (uploaded file). Never
        # touch a file the user pointed to via the local-path tab.
        if media.is_temp and media.path.parent.exists():
            cleanup(media.path.parent)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _render_result_and_ai() -> None:
    """Render the stored transcription result and the AI panel, if any.

    Reads from ``st.session_state`` so the result persists across the
    reruns triggered by the chat widgets.
    """
    segments: list[TranscriptSegment] | None = st.session_state.get("result_segments")
    if not segments:
        return
    meta = st.session_state["result_meta"]

    render_results(segments, base_filename=meta["base"])
    st.caption(
        f"Idioma detectado: `{meta['language']}` "
        f"(probabilidad {meta['language_probability']:.2f}) · "
        f"duración {meta['duration']:.1f} s"
    )
    render_ai_panel(
        formatting.to_timestamped_text(segments),
        language=meta["language"],
        duration_seconds=meta["duration"],
    )


def render_page() -> None:
    """Compose the full Streamlit page."""
    st.set_page_config(
        page_title="Audicop",
        page_icon="🎙️",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    render_header()

    hw = _cached_hardware()
    choice = _cached_recommendation(hw)
    render_status_banner(hw, choice)

    settings = render_sidebar(choice)
    media = render_uploader()

    if media is not None:
        render_input_summary(media)
        if st.button("Transcribir", type="primary", use_container_width=True):
            _run_transcription(media=media, settings=settings)
    else:
        st.caption("Sube un archivo o pega una ruta local para empezar.")

    _render_result_and_ai()
    render_privacy_footer()


def main() -> None:
    """CLI entry point: launch streamlit pointed at this file."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())


# When invoked via `streamlit run audicop/app.py`, the module is executed
# top-to-bottom in the streamlit script context. The runtime guard avoids
# rendering when the module is imported for the CLI entry point.
if st.runtime.exists():
    render_page()
