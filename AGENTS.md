# AGENTS.md — Reglas operativas de Audicop (fuente de verdad)

> Este archivo manda. Si algo en el código contradice esto, gana este archivo
> (o actualiza este archivo conscientemente). Pensado para agentes de IA y
> humanos que toquen el repo.

---

## §1. Qué es Audicop

App web **local** para transcribir y traducir audio y vídeo, escrita en Python
con **FastAPI** (backend) + un **frontend propio HTML/CSS/JS vanilla** +
**faster-whisper** (motor). El usuario clona, ejecuta un script y transcribe en
minutos, **sin configurar nada**: la app autodetecta el hardware y elige modelo
y parámetros.

Es **un solo proceso local** (uvicorn sirve API + frontend en localhost).
**Sin Docker, sin Node, sin build** — esa simplicidad es deliberada (era la razón
de usar Streamlit antes; ahora la conservamos con uvicorn).

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
5. **API keys solo en memoria.** Las keys de IA viven en la memoria del navegador
   y se mandan por request; el backend es stateless con ellas. Nunca se escriben a
   disco, nunca se loguean, input siempre `type="password"`.
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

## §4. Arquitectura (frontend / backend, por capas)

```
backend/app/
├── core/config.py       Constantes: formatos, tabla de modelos, IA, export. Sin lógica.
├── adapters/            Drivers finos sobre integraciones externas:
│   ├── hardware.py        detect_hardware() → HardwareInfo (psutil + nvidia-smi).
│   ├── audio.py           to_wav_16k() vía ffmpeg empaquetado + limpieza.
│   ├── transcriber.py     Wrapper de WhisperModel. DLL CUDA Windows. Fallback sin symlinks.
│   └── llm.py             Cliente agnóstico (OpenAI/Gemini), streaming, BYO key.
├── services/            Lógica de negocio pura (testeable):
│   ├── recommender.py     recommend(hw) → ModelChoice (memoria LIBRE, no total).
│   ├── formatting.py      Segments → texto plano / timestamped / SRT / VTT.
│   └── pipeline.py        iter_transcription() → stream de eventos; cache de Transcriber.
├── api/                 Rutas FastAPI (capa I/O):
│   ├── hardware.py        GET /api/hardware
│   ├── transcribe.py      POST /api/transcribe + GET .../events (SSE)
│   └── chat.py            POST /api/chat (SSE)
├── prompts/             Paquete: __init__.py carga los .md editables (system, context, actions/*).
└── main.py              FastAPI: monta routers + sirve frontend/.

frontend/                Estático, vanilla, offline (sin Node/CDN):
├── index.html, styles.css ("Slate"), app.js (fetch + SSE)
```

Flujo: **Frontend → API → services/adapters → ffmpeg / faster-whisper / LLM**,
con progreso por SSE. `api/` y `main.py` son la capa I/O; la lógica testeable
vive en `services/` y `adapters/`.

**Streaming de progreso:** el decode bloqueante corre en un thread; cada evento
se publica en una `asyncio.Queue` (`call_soon_threadsafe`) que el endpoint SSE
drena al navegador. Jobs en un dict en memoria (single-user local).

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
  la app. Capturar, loguear, devolver un error amable (HTTP o evento SSE
  `{"type":"error"}`) con el siguiente paso.
- Constantes en `core/config.py`; nada de magic numbers en la lógica.
- Sin código muerto, sin imports sin usar, sin `# type: ignore` salvo motivo
  comentado.

---

## §6. Tooling y gates (deben pasar antes de commit)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy backend/app/core/config.py backend/app/adapters/*.py \
            backend/app/services/*.py backend/app/prompts/__init__.py
uv run pytest --cov            # ≥ 80% en módulos no-API
```

- **uv** gestiona Python y dependencias. `uv.lock` es la fuente de verdad de
  versiones (hash-verificado). No hay `requirements.txt`.
- mypy estricto en `core`/`adapters`/`services`/`prompts`; `api/` y `main.py`
  exentos (capa I/O), cubiertos por tests de `TestClient`.

---

## §7. Dependencias

Stack base: `fastapi`, `uvicorn[standard]`, `python-multipart`,
`faster-whisper`, `imageio-ffmpeg`, `psutil`, `openai`, `google-genai`.
(Las libs de IA van en base a propósito: `uv sync` poda lo que no esté en los
extras activos, así que un extra opcional se desinstalaría en cada arranque y el
chat se rompería. Son ligeras, no como torch.)

Extras opcionales:
- `cuda` = `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` (el launcher las instala
  solo si detecta `nvidia-smi`).
- `dev` = `ruff`, `mypy`, `pytest`, `pytest-cov`.

Frontend: **cero dependencias** (vanilla, sin Node/CDN).

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
