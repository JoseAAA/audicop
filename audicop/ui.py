"""Streamlit UI components for Audicop.

These functions render reusable pieces of the interface (header, hardware
panel, sidebar with the advanced override). The orchestration lives in
:mod:`audicop.app`; everything in this module returns plain values (or
mutates ``st.session_state``) so the entry point stays tiny.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from audicop import config, formatting, llm, prompts
from audicop.hardware import HardwareInfo
from audicop.llm import ChatMessage, Provider
from audicop.recommender import ModelChoice
from audicop.transcriber import TranscriptSegment


@dataclass(frozen=True, slots=True)
class TranscriptionSettings:
    """Resolved settings the user (or recommender) chose for a run."""

    model_size: str
    compute_type: str
    device: str
    language: str | None
    task: str
    vad_filter: bool


@dataclass(frozen=True, slots=True)
class MediaInput:
    """One concrete media file to transcribe.

    Attributes:
        name: User-visible file name (basename, no path).
        path: Absolute path to the file on disk.
        size_bytes: File size in bytes.
        is_temp: Whether ``path`` lives in a temp dir we own (and should
            therefore delete after processing). For uploaded files this is
            ``True``; for files referenced by a local path it is ``False``.
    """

    name: str
    path: Path
    size_bytes: int
    is_temp: bool


def render_header() -> None:
    """Render the page title and tagline."""
    st.title(config.APP_TITLE)
    st.caption(config.APP_TAGLINE)


def _format_mb(size_bytes: int) -> str:
    return f"{size_bytes / 1024 / 1024:.1f} MB"


_FRIENDLY_MODEL_NAMES: dict[str, str] = {
    "tiny": "Tiny",
    "base": "Base",
    "small": "Small",
    "medium": "Medium",
    "large-v3": "Large v3",
}


def _short_gpu_name(name: str) -> str:
    """Strip vendor noise from a GPU marketing string for compact display."""
    return name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")


def render_status_banner(hw: HardwareInfo, choice: ModelChoice) -> None:
    """One-line status: where we run, what model, how fast.

    The full nerdy detail is hidden behind an expander so the main page
    stays calm. Non-technical users see "todo listo, vas a tardar X" and
    can ignore everything else.
    """
    factor = config.REALTIME_FACTORS.get((choice.model_size, choice.device), 1.0)
    minutes_per_hour = max(1, round(60 / factor))
    model_label = _FRIENDLY_MODEL_NAMES.get(choice.model_size, choice.model_size)

    if choice.device == "cuda" and hw.gpu_name:
        device_label = f"GPU **{_short_gpu_name(hw.gpu_name)}**"
    elif choice.device == "cuda":
        device_label = "**GPU NVIDIA**"
    else:
        device_label = f"**CPU** ({hw.cpu_cores_physical} núcleos)"

    st.success(
        f"✅ Listo. Voy a transcribir en {device_label} con el modelo "
        f"**{model_label}** · ~**{minutes_per_hour} min** por cada hora de audio "
        f"· hasta **{config.MAX_DURATION_HOURS:.0f} h** soportadas."
    )

    with st.expander("Ver detalles del sistema y por qué se eligió este modelo"):
        col_hw, col_model = st.columns(2)
        with col_hw:
            st.markdown("**Tu equipo**")
            if hw.has_cuda and hw.gpu_name:
                free = f"{hw.gpu_vram_free_gb:.1f}" if hw.gpu_vram_free_gb is not None else "?"
                total = f"{hw.gpu_vram_total_gb:.1f}" if hw.gpu_vram_total_gb is not None else "?"
                st.markdown(f"- 🎮 GPU: {hw.gpu_name} · `{free} / {total} GB libres`")
                if hw.gpu_driver_version:
                    st.markdown(f"- Driver NVIDIA: `{hw.gpu_driver_version}`")
            else:
                st.markdown("- 🎮 GPU: sin CUDA detectada")
            st.markdown(f"- 🧠 RAM: `{hw.ram_available_gb:.1f} / {hw.ram_total_gb:.1f} GB libres`")
            st.markdown(
                f"- 🖥️ CPU: {hw.cpu_cores_physical} físicos · "
                f"{hw.cpu_cores_logical} lógicos · {hw.os_name}"
            )

        with col_model:
            st.markdown("**Modelo elegido**")
            st.markdown(f"- Tamaño: `{choice.model_size}`")
            st.markdown(f"- Compute: `{choice.compute_type}`")
            st.markdown(f"- Dispositivo: `{choice.device}`")
            st.caption(choice.rationale)

        st.caption(
            "💡 La elección se basa en la memoria **libre** ahora mismo. Si "
            "cierras apps que consumen RAM/VRAM y recargas la página, puede "
            "subir a un modelo mejor. También puedes forzarlo en *Modo avanzado*."
        )


_LANGUAGE_LABELS: dict[str, str] = {
    "auto": "Auto-detectar",
    "es": "Español",
    "en": "Inglés",
    "pt": "Portugués",
    "fr": "Francés",
    "it": "Italiano",
    "de": "Alemán",
}

_TASK_LABELS: dict[str, str] = {
    "transcribe": "Transcribir (mismo idioma)",
    "translate": "Traducir a inglés",
}

_DEVICE_LABELS: dict[str, str] = {
    "cuda": "GPU (NVIDIA)",
    "cpu": "CPU",
}


def render_sidebar(default: ModelChoice) -> TranscriptionSettings:
    """Render the sidebar (language, action, advanced override).

    Args:
        default: The recommendation produced by `recommend()`. Used as the
            default value for every advanced control.

    Returns:
        The resolved :class:`TranscriptionSettings` for the next run.
    """
    st.sidebar.header("⚙️ Opciones")

    language = st.sidebar.selectbox(
        "Idioma del audio",
        options=list(config.DEFAULT_LANGUAGES),
        index=0,
        format_func=lambda code: _LANGUAGE_LABELS.get(code, code),
        help="Si no estás seguro, deja **Auto-detectar**.",
    )
    task = st.sidebar.selectbox(
        "Acción",
        options=list(config.DEFAULT_TASKS),
        index=0,
        format_func=lambda code: _TASK_LABELS.get(code, code),
    )
    vad_filter = st.sidebar.checkbox(
        "Saltar silencios automáticamente",
        value=config.DEFAULT_VAD_FILTER,
        help="Recomendado: mejora precisión y velocidad en audios con pausas largas.",
    )

    with st.sidebar.expander("🛠️ Modo avanzado"):
        st.caption(
            "Sólo si quieres forzar otro modelo o cómputo. La elección "
            "automática suele ser la correcta."
        )
        model_size = st.selectbox(
            "Tamaño del modelo",
            options=list(config.VALID_MODEL_SIZES),
            index=config.VALID_MODEL_SIZES.index(default.model_size),
            format_func=lambda code: _FRIENDLY_MODEL_NAMES.get(code, code),
        )
        compute_type = st.selectbox(
            "Tipo de cómputo (precisión)",
            options=list(config.VALID_COMPUTE_TYPES),
            index=config.VALID_COMPUTE_TYPES.index(default.compute_type),
        )
        device = st.selectbox(
            "Dispositivo",
            options=["cuda", "cpu"],
            index=0 if default.device == "cuda" else 1,
            format_func=lambda code: _DEVICE_LABELS.get(code, code),
        )

    return TranscriptionSettings(
        model_size=model_size,
        compute_type=compute_type,
        device=device,
        language=None if language == "auto" else language,
        task=task,
        vad_filter=vad_filter,
    )


def _resolve_local_path(path_str: str) -> MediaInput | None:
    """Validate a user-supplied local path and turn it into a MediaInput.

    Returns ``None`` (after rendering an ``st.error``) if the path is not
    a readable file with a supported extension.
    """
    if not path_str.strip():
        return None
    cleaned = path_str.strip().strip('"').strip("'")
    p = Path(cleaned).expanduser()
    if not p.exists():
        st.error(f"No encuentro el archivo: `{p}`")
        return None
    if not p.is_file():
        st.error("La ruta apunta a una carpeta o un dispositivo, no a un archivo.")
        return None
    ext = p.suffix.lower().lstrip(".")
    if ext not in config.SUPPORTED_EXTENSIONS:
        st.error(
            f"Extensión `.{ext}` no soportada. "
            f"Soportadas: {', '.join(config.SUPPORTED_EXTENSIONS)}."
        )
        return None
    return MediaInput(name=p.name, path=p.resolve(), size_bytes=p.stat().st_size, is_temp=False)


def _save_upload_to_temp(upload: st.runtime.uploaded_file_manager.UploadedFile) -> MediaInput:
    """Persist an uploaded file to a temp dir and return its MediaInput."""
    tmp_dir = Path(tempfile.mkdtemp(prefix=config.TEMP_PREFIX))
    safe_name = Path(upload.name).name
    target = tmp_dir / safe_name
    target.write_bytes(upload.getbuffer())
    return MediaInput(name=safe_name, path=target, size_bytes=target.stat().st_size, is_temp=True)


def render_uploader() -> MediaInput | None:
    """Render the input section (upload + local path tabs).

    Returns:
        A :class:`MediaInput` describing the chosen file, or ``None`` if
        the user has not provided one yet (or the path is invalid).
    """
    st.subheader("Sube tu archivo")
    extensions_str = ", ".join(config.SUPPORTED_EXTENSIONS)

    tab_upload, tab_local = st.tabs(
        ["📤 Subir archivo", "📁 Archivo local (recomendado para vídeos grandes)"]
    )

    with tab_upload:
        upload = st.file_uploader(
            f"Audio o vídeo ({extensions_str})",
            type=list(config.SUPPORTED_EXTENSIONS),
            accept_multiple_files=False,
            help=(
                f"Tamaño máximo de subida: {config.MAX_UPLOAD_MB / 1000:.0f} GB. "
                f"Si tu archivo es mayor (vídeos largos en HD), usa la pestaña 'Archivo local'."
            ),
        )
        if upload is not None:
            return _save_upload_to_temp(upload)

    with tab_local:
        st.caption(
            "Pega la **ruta absoluta** del archivo. Audicop lo lee directamente "
            "del disco — no se hace subida, así que no hay límite de tamaño y "
            "es más rápido. Sólo funciona porque la app corre en tu máquina."
        )
        path_str = st.text_input(
            "Ruta absoluta",
            placeholder=r"C:\Users\tu_usuario\Videos\reunion.mp4",
            label_visibility="collapsed",
        )
        if path_str:
            return _resolve_local_path(path_str)

    return None


def render_input_summary(media: MediaInput) -> None:
    """Show file metadata once the user has picked a media input."""
    location = "📤 Subido" if media.is_temp else "📁 Local"
    st.success(f"{location} · **{media.name}** · {_format_mb(media.size_bytes)}")


def _resolve_chat_input(transcript_chars: int) -> str | None:
    """Render quick-action buttons + chat box, return the user's prompt if any."""
    st.caption("Atajos:")
    cols = st.columns(len(prompts.QUICK_ACTIONS))
    clicked: str | None = None
    for col, action in zip(cols, prompts.QUICK_ACTIONS, strict=True):
        if col.button(action.label, use_container_width=True):
            clicked = action.prompt

    if transcript_chars > config.LONG_TRANSCRIPT_CHARS:
        st.warning(
            "La transcripción es muy larga. Puede superar el límite del modelo o "
            "encarecer la consulta. Considera un modelo de contexto amplio "
            "(p. ej. Gemini) o preguntar por tramos."
        )

    typed = st.chat_input("Pregunta sobre el audio… (cita momentos como [12:34])")
    return clicked or typed


def render_ai_panel(transcript_timestamped: str, *, language: str, duration_seconds: float) -> None:
    """Render the bring-your-own-key AI chat/analysis over the transcript.

    Owns the chat interaction loop; delegates the actual provider call to
    :func:`audicop.llm.stream_chat`. The API key lives only in
    ``st.session_state`` (never written to disk).

    Args:
        transcript_timestamped: Transcript as ``[MM:SS] text`` (the AI context).
        language: Detected/forced language code of the audio.
        duration_seconds: Audio duration in seconds.
    """
    st.divider()
    st.subheader("🤖 Analiza con IA")
    st.caption(
        "Resume, extrae tareas o pregunta lo que quieras sobre el audio. "
        "Usa tu propia API key (gratis en Gemini)."
    )

    col_provider, col_model = st.columns(2)
    with col_provider:
        provider: Provider = st.selectbox(
            "Proveedor",
            options=["openai", "gemini"],
            index=1,  # Gemini default (free tier)
            format_func=lambda p: llm.PROVIDER_LABELS[p],
        )
    with col_model:
        model = st.selectbox("Modelo", options=list(llm.available_models(provider)))

    api_key = st.text_input(
        "API key",
        type="password",
        key="llm_api_key",
        help=llm.API_KEY_HELP[provider],
        placeholder="Se queda solo en memoria; no se guarda en disco.",
    )

    st.warning(
        f"⚠️ Al usar el chat, el **texto** de la transcripción se envía a "
        f"**{llm.PROVIDER_LABELS[provider]}** con tu API key. El audio nunca sale "
        f"de tu equipo; sólo el texto, y sólo cuando preguntas."
    )

    history: list[ChatMessage] = st.session_state.setdefault("chat_history", [])

    top = st.container()
    with top:
        cols = st.columns([1, 1, 4])
        if cols[0].button("🧹 Limpiar", use_container_width=True):
            history.clear()
            st.rerun()

    for msg in history:
        st.chat_message(msg.role).markdown(msg.content)

    user_text = _resolve_chat_input(len(transcript_timestamped))
    if not user_text:
        return

    if not (api_key or "").strip():
        st.error("Pega tu API key arriba para usar el chat.")
        return

    history.append(ChatMessage(role="user", content=user_text))
    st.chat_message("user").markdown(user_text)

    system = (
        prompts.SYSTEM_PROMPT
        + "\n\n"
        + prompts.build_context(
            transcript_timestamped, language=language, duration_seconds=duration_seconds
        )
    )

    with st.chat_message("assistant"):
        try:
            stream = llm.stream_chat(
                provider=provider,
                api_key=api_key,
                model=model,
                system=system,
                messages=history,
            )
            reply = st.write_stream(stream)
        except llm.LLMError as exc:
            st.error(str(exc))
            history.pop()  # drop the user turn we couldn't answer
            return

    history.append(ChatMessage(role="assistant", content=str(reply)))


def render_privacy_footer() -> None:
    """Render a footer clarifying what stays local and what leaves the machine."""
    with st.expander("🔒 Privacidad — qué hace y qué NO hace Audicop"):
        st.markdown(
            """
**La transcripción es 100% local.** Tu audio nunca sale de este equipo. La
única conexión a internet para transcribir es la **descarga inicial del
modelo** desde HuggingFace, y sólo la primera vez que usas cada tamaño.

**El chat con IA es la excepción.** Si lo usas, el **texto** de la
transcripción se envía al proveedor que elijas (OpenAI o Gemini) con tu
propia API key. El audio original **nunca** se sube; sólo el texto, y sólo
cuando tú usas el chat. La key vive sólo en memoria — no se guarda en disco.

**Para detectar el hardware** uso únicamente:
- `psutil` → cuenta de cores y memoria total/libre. No lee procesos ni archivos.
- `nvidia-smi` → binario oficial de NVIDIA (viene con el driver). Sólo
  consulta nombre de GPU y memoria.
- `platform` (stdlib de Python) → nombre del sistema operativo.

**Lo que NUNCA hace:**
- Subir tus archivos de audio/vídeo a ninguna nube.
- Enviar telemetría, analytics o "phone home".
- Acceder a webcam, micrófono ni portapapeles.
- Guardar tu API key en disco o en logs.
            """
        )


def render_results(segments: list[TranscriptSegment], base_filename: str) -> None:
    """Render the transcription with plain / timestamped / export views.

    Args:
        segments: Decoded segments (carry start/end for timestamps + subtitles).
        base_filename: Source file stem, used for download file names.
    """
    plain = formatting.to_plain_text(segments)
    timestamped = formatting.to_timestamped_text(segments)

    st.success("✅ Transcripción completada.")

    tab_plain, tab_ts, tab_export = st.tabs(["📄 Texto", "⏱️ Con marcas de tiempo", "⬇️ Exportar"])

    with tab_plain:
        st.caption("Texto corrido, listo para copiar y pegar.")
        st.code(plain or "(sin texto)", language="text")

    with tab_ts:
        st.caption("Estilo YouTube: cada línea con el momento en que se dijo.")
        st.code(timestamped or "(sin texto)", language="text")

    with tab_export:
        st.caption("Descarga en el formato que necesites.")
        col_txt, col_srt, col_vtt = st.columns(3)
        with col_txt:
            st.download_button(
                "📄 .txt",
                data=plain.encode("utf-8"),
                file_name=f"{base_filename}{config.TXT_SUFFIX}",
                mime="text/plain",
                use_container_width=True,
            )
        with col_srt:
            st.download_button(
                "🎬 .srt",
                data=formatting.to_srt(segments).encode("utf-8"),
                file_name=f"{base_filename}{config.SRT_SUFFIX}",
                mime="text/plain",
                use_container_width=True,
            )
        with col_vtt:
            st.download_button(
                "🌐 .vtt",
                data=formatting.to_vtt(segments).encode("utf-8"),
                file_name=f"{base_filename}{config.VTT_SUFFIX}",
                mime="text/plain",
                use_container_width=True,
            )
        st.caption(
            "💡 ¿Quieres analizarlo con IA (resumen, puntos clave, tareas)? "
            "Usa el panel de abajo, o copia el texto y pégalo en tu IA favorita."
        )
