# AGENTS.md — Reglas operativas de Audicop (fuente de verdad)

> Este archivo manda. Si algo en el código contradice esto, gana este archivo
> (o actualiza este archivo conscientemente). Pensado para agentes de IA y
> humanos que toquen el repo.

---

## §1. Qué es Audicop

App web **local** para transcribir y traducir audio y vídeo, escrita en Python
con **Streamlit** + **faster-whisper**. El usuario clona, ejecuta un script y
transcribe en minutos, **sin configurar nada**: la app autodetecta el hardware
y elige modelo y parámetros.

La **función principal es transcribir**. Todo lo demás (timestamps, export,
chat IA) es valor añadido y no debe estorbar ese flujo.

---

## §2. Decisiones no negociables

1. **Local primero.** La transcripción corre 100% en la máquina del usuario.
   La única red permitida sin consentimiento es la descarga del modelo Whisper.
2. **Cero configuración obligatoria.** El usuario no elige modelo, dispositivo
   ni compute_type para empezar. La autodetección decide; el "Modo avanzado"
   permite sobrescribir.
3. **Cero dependencias de sistema.** ffmpeg viene vía `imageio-ffmpeg`; CUDA vía
   wheels PyPI. Nada de `apt`/`brew`/CUDA toolkit del sistema.
4. **UI en español, código en inglés.** Identifiers, docstrings y comentarios en
   inglés (estándar OSS). Todo lo que ve el usuario, en español.
5. **API keys solo en memoria.** Las keys de IA viven en `st.session_state`.
   Nunca se escriben a disco, nunca se loguean, input siempre `type="password"`.
6. **Fácil para no-técnicos.** Si una pantalla necesita un manual, está mal
   diseñada. Lee `DESIGN.md`.

---

## §3. Privacidad (qué sale del equipo y qué no)

| Acción                     | ¿Sale a la red?                                    |
|----------------------------|----------------------------------------------------|
| Detección de hardware      | No (psutil, nvidia-smi, platform — solo lectura)   |
| Conversión a WAV (ffmpeg)  | No (binario local empaquetado)                     |
| Transcripción (Whisper)    | No (modelo local; solo se descarga la 1ª vez)      |
| **Chat / análisis IA**     | **Sí** — el texto de la transcripción va al        |
|                            | proveedor cloud elegido (OpenAI/Gemini) con la key |
|                            | del usuario. Hay que **avisarlo antes** del 1er uso |

El audio original **nunca** se sube a la nube. El chat IA solo envía el **texto**
ya transcrito, y solo si el usuario lo usa activamente.

---

## §4. Arquitectura (por módulo)

```
audicop/
├── config.py        Constantes: formatos, tabla de modelos, IA, export. Sin lógica.
├── hardware.py      detect_hardware() → HardwareInfo (psutil + nvidia-smi).
├── recommender.py   recommend(hw) → ModelChoice (memoria LIBRE, no total).
├── audio.py         to_wav_16k() vía ffmpeg empaquetado. Limpieza de temporales.
├── transcriber.py   Wrapper de WhisperModel. DLL CUDA en Windows. Fallback sin symlinks.
├── formatting.py    Segments → texto plano / timestamped / SRT / VTT. Funciones puras.
├── prompts/         Paquete: __init__.py carga los .md editables (system, context, actions/*).
├── llm.py           Cliente agnóstico (OpenAI/Gemini), streaming, BYO key.
├── ui.py            Componentes Streamlit. Presentación, sin lógica de negocio.
└── app.py           Entry point delgado: orquesta módulos + session_state.
```

Flujo: **Upload/Path → ffmpeg → faster-whisper → segments → UI → (IA opcional)**.

`app.py` y `ui.py` son la capa de presentación. La lógica testeable vive en los
otros módulos.

---

## §5. Convenciones de código (no negociables)

- **PEP 8**, line length **100**.
- `from __future__ import annotations` al tope de cada archivo.
- **Type hints** en toda función pública. **Docstrings estilo Google** en
  módulos/clases/funciones públicas; las privadas (`_x`) pueden tener una línea.
- **Logging** con `logging.getLogger(__name__)`. Nunca `print()`.
- **`pathlib.Path`**, no `os.path`.
- **Dataclasses** para estructuras (`HardwareInfo`, `ModelChoice`, `TranscriptSegment`…).
- **Manejo de errores**: nunca dejar que un fallo de ffmpeg/CUDA/IO/IA reviente
  la app. Capturar, loguear, mostrar `st.error()` amable con el siguiente paso.
- Constantes en `config.py`; nada de magic numbers en la lógica.
- Sin código muerto, sin imports sin usar, sin `# type: ignore` salvo motivo
  comentado.

---

## §6. Tooling y gates (deben pasar antes de commit)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy audicop/hardware.py audicop/recommender.py audicop/audio.py \
            audicop/transcriber.py audicop/formatting.py audicop/llm.py \
            audicop/prompts/__init__.py
uv run pytest --cov            # ≥ 80% en módulos no-UI
```

- **uv** gestiona Python y dependencias. `uv.lock` es la fuente de verdad de
  versiones (hash-verificado). No hay `requirements.txt`.
- mypy estricto en módulos de lógica; `app.py`/`ui.py` exentos (Streamlit no
  juega bien con mypy).

---

## §7. Dependencias

Stack base: `streamlit`, `faster-whisper`, `imageio-ffmpeg`, `psutil`.
Extras opcionales:
- `cuda` = `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` (el launcher las instala
  solo si detecta `nvidia-smi`).
- `ai` = `openai`, `google-genai` (chat/análisis con IA).
- `dev` = `ruff`, `mypy`, `pytest`, `pytest-cov`.

**Pregunta antes de añadir cualquier dependencia fuera de esta lista.**

---

## §8. Estrategia multi-LLM

- Proveedores soportados: **OpenAI** y **Google Gemini**. Adaptador agnóstico en
  `llm.py` con imports perezosos (la app arranca sin las libs instaladas).
- Añadir un proveedor = nueva función `_stream_<provider>` + entrada en las
  constantes; sin refactor del resto.
- Streaming por defecto (`st.write_stream`) para feedback inmediato.

---

## §9. Prohibiciones absolutas

- ❌ Telemetría, analytics o "phone home" de cualquier tipo.
- ❌ Persistir API keys a disco (salvo opt-in explícito futuro, hoy no existe).
- ❌ Subir el audio del usuario a ninguna nube.
- ❌ Acceder a webcam, micrófono o portapapeles.
- ❌ Romper el flujo local: la app debe transcribir aunque no haya internet
  (con el modelo ya cacheado) ni API keys.
- ❌ Mostrar jerga técnica cruda al usuario (ver `DESIGN.md`).

---

## §10. Roadmap

- [x] Autodetección de hardware + recomendación por memoria libre.
- [x] Fallback de descarga sin symlinks (Windows restringido).
- [x] Timestamps estilo YouTube + export SRT/VTT.
- [x] Chat / análisis IA con BYO key (OpenAI/Gemini).
- [ ] Diarización (separar hablantes).
- [ ] Proceso por lotes (varios archivos / carpetas).
- [ ] Más proveedores IA (Anthropic, modelos locales vía Ollama).

---

## §11. Cómo trabajar en este repo

1. Lee este archivo, `DESIGN.md` y el `README.md` antes de tocar nada.
2. Empieza por la lógica (`config` → módulo) con sus tests en verde antes de
   pasar a la UI.
3. No avances de módulo hasta tener test verde + docstrings completos.
4. Corre el gate de §6 antes de cada commit.
5. Verifica el flujo end-to-end con un archivo real antes de dar algo por hecho.
