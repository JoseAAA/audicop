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
5. **Privacidad de extremo a extremo.** El análisis IA corre **100% local**
   (llama.cpp). No hay proveedores cloud ni API keys: ni el audio ni el texto
   salen del equipo.
6. **Fácil para no-técnicos.** Si una pantalla necesita un manual, está mal
   diseñada. Lee `DESIGN.md`.

---

## §3. Privacidad (qué sale del equipo y qué no)

| Acción                     | ¿Sale a la red?                                    |
|----------------------------|----------------------------------------------------|
| Detección de hardware      | No (psutil, nvidia-smi, platform — solo lectura)   |
| Detección de reunión       | No (lee del registro qué app usa el micrófono; ni red ni audio) |
| Grabación (mic / loopback) | No (soundcard local; el WAV se queda en el equipo) |
| Conversión a WAV (ffmpeg)  | No (binario local empaquetado)                     |
| Transcripción (Whisper)    | No (modelo local; solo se descarga la 1ª vez)      |
| **Análisis IA (resumen/chat)** | No (modelo local llama.cpp; solo se descarga la 1ª vez) |

**Nada sale del equipo**: ni el audio ni el texto. El análisis IA usa un modelo
local (llama.cpp) elegido según el hardware. El único acceso a red es la
**descarga inicial** de los modelos (Whisper y el LLM local), que luego quedan
en caché.

**Biblioteca de reuniones (disco local):** cada transcripción terminada se
guarda con sus notas de IA en `~/.audicop/meetings.db` (SQLite) y una copia
comprimida del audio en `~/.audicop/audio/{id}.m4a` (AAC ~48 kbps, para poder
reescucharla al reabrir) — **todo solo en el disco del usuario**. Borrar una
reunión desde la UI (🗑) elimina texto, notas y audio, permanente. Documentado
en el panel de Privacidad de la UI.

**Defensa local:** el servidor escucha solo en `127.0.0.1` (los launchers usan
`--host 127.0.0.1`) y dos middlewares lo protegen: (1) un guard de **Host**
rechaza cualquier petición cuyo `Host` no sea localhost — bloquea DNS
rebinding, que si no permitiría a una web maliciosa leer p. ej.
`/api/transcript`; (2) un guard de **Origin** rechaza POST/PUT/DELETE de
orígenes remotos (anti-CSRF: evita que una web dispare `/api/record/start`).
No hay autenticación porque es de un solo usuario en su propia máquina;
**no exponer el puerto a la red**.

---

## §4. Arquitectura (frontend / backend, por capas)

```
backend/app/
├── core/config.py       Constantes: formatos, tabla de modelos, grabación, IA, export. Sin lógica.
├── adapters/            Drivers finos sobre integraciones externas:
│   ├── hardware.py        detect_hardware() → HardwareInfo (psutil + nvidia-smi).
│   ├── audio.py           to_wav_16k() vía ffmpeg empaquetado + limpieza.
│   ├── transcriber.py     Wrapper de WhisperModel (turbo + batched GPU). DLL CUDA. Fallback sin symlinks.
│   ├── capture.py         Grabación local mic + loopback (soundcard) → WAV 16k. COM aislado por hilo.
│   ├── meeting.py         detect_active_meeting(): qué app usa el micrófono (winreg). Meet/Teams/Zoom.
│   ├── local_llm.py       IA local (llama.cpp): descarga+carga GGUF, streaming. 100% on-device.
│   └── cuda_dll.py        Pone las DLLs NVIDIA en el PATH (compartido con transcriber).
├── services/            Lógica de negocio pura (testeable):
│   ├── recommender.py     recommend(hw) → ModelChoice (memoria LIBRE, no total).
│   ├── formatting.py      Segments → texto plano / timestamped / SRT / VTT.
│   └── pipeline.py        iter_transcription() → stream de eventos; cache de Transcriber.
├── api/                 Rutas FastAPI (capa I/O):
│   ├── hardware.py        GET /api/hardware
│   ├── transcribe.py      POST /api/transcribe + GET .../events (SSE)
│   ├── record.py          GET /api/record/meeting + POST .../start|pause|resume|stop
│   └── chat.py            POST /api/chat (SSE)
├── prompts/             Paquete: __init__.py carga los .md editables (system, context, actions/*).
└── main.py              FastAPI: guard CSRF (Origin) + monta routers + sirve frontend/.

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

> **Las herramientas (ruff, mypy, pytest) viven en el extra `dev`** y
> `llama-cpp-python` lo instala el launcher fuera del lock (§7). Un `uv run`
> o `uv sync` "a secas" **poda ambos**: falla con *No module named pytest* o
> desinstala el motor de IA local. Flujo correcto: sincroniza UNA vez con
> `--inexact` y corre los gates con `--no-sync`:

```bash
uv sync --inexact --extra dev              # 1 vez (añade --extra cuda si hay GPU)
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy backend/app/core/config.py backend/app/adapters/*.py \
            backend/app/services/*.py backend/app/prompts/__init__.py
uv run --no-sync pytest --cov              # ≥ 80% en módulos no-API
```

- **uv** gestiona Python y dependencias. `uv.lock` es la fuente de verdad de
  versiones (hash-verificado). No hay `requirements.txt`.
- mypy estricto en `core`/`adapters`/`services`/`prompts`; `api/` y `main.py`
  exentos (capa I/O), cubiertos por tests de `TestClient`.

---

## §7. Dependencias

Stack base: `fastapi`, `uvicorn[standard]`, `python-multipart`,
`faster-whisper`, `imageio-ffmpeg`, `psutil`, `soundcard`. (`soundcard`, BSD-3,
da la captura mic + loopback de los modos de grabación; pip-only, sin deps de
sistema, sin torch.) **No hay SDKs de nube** (ni `openai` ni `google-genai`): el
análisis IA es 100% local.

Extras opcionales:
- `cuda` = `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cuda-runtime-cu12`
  (el launcher las instala solo si detecta `nvidia-smi`; el runtime aporta
  `cudart`, que necesita tanto Whisper como la GPU de llama.cpp).
- `dev` = `ruff`, `mypy`, `pytest`, `pytest-cov`.

**IA local (`llama-cpp-python`) — la instala el LAUNCHER, no está en el lock.**
No hay wheel universal: cada backend es un paquete distinto en un índice aparte
(`abetlen.github.io/llama-cpp-python/whl/{cpu,cu124,vulkan,metal}`) y un lock no
puede expresar "wheel CUDA si hay GPU". Por eso `scripts/start.*` hace
`uv sync --inexact` (para que el sync no lo pode) y luego `uv pip install` el
wheel correcto **según SO + GPU**, y lanza con `uv run --no-sync`:

| SO | NVIDIA | resto (GPU AMD/Intel o solo CPU) |
| --- | --- | --- |
| Windows | `cu124` | **`vulkan`** |
| Linux | `cu124` | `cpu` (ya es portable) |
| macOS | — | `metal` |

⚠️ En Windows **no** se usa el índice `cpu`: abetlen lo compila *native* en un
runner de CI con AVX-512 (su workflow pone `-DGGML_NATIVE=off` solo en
Linux/macOS), así que crashea con `0xc000001d` (illegal instruction) en las
laptops sin AVX-512 (Intel 12ª gen+, Core Ultra, casi todo Ryzen móvil). El
wheel `vulkan` es portable (`-DGGML_NATIVE=off`) y además descarga capas a
cualquier GPU Vulkan (iGPU/dGPU Intel/AMD/NVIDIA). `local_llm._describe_load_failure`
traduce ese fallo a un mensaje claro si aun así ocurre.

`adapters/local_llm.py` lo importa perezosamente (degrada con mensaje si falta)
y `adapters/cuda_dll.py` pone las DLLs NVIDIA en el PATH (compartido con el
transcriber) antes de importarlo.

Frontend: **cero dependencias** (vanilla, sin Node/CDN).

**Pregunta antes de añadir cualquier dependencia fuera de esta lista.**

---

## §8. IA local (on-device)

- El análisis (resumen / puntos / tareas / acta + chat libre) corre **100%
  local** con `llama.cpp` (`llama-cpp-python`). Sin nube, sin API keys.
- `services/recommender.recommend_llm(hw, *, gpu_offload)` elige el GGUF por
  memoria **libre** (RAM/VRAM) menos una reserva explícita, igual que Whisper.
  `gpu_offload` (de `local_llm.supports_gpu_offload()`) indica si el motor
  instalado puede usar la GPU; si no, dimensiona contra la RAM.
- `adapters/local_llm.py`: descarga (1ª vez) + carga perezosa + streaming token
  a token (SSE). Catálogo en `config.LLM_MODELS` (Qwen2.5 3B/1.5B, Llama 3.2 1B).
- Añadir un modelo = nueva entrada en `LLM_MODELS` + (si hace falta) un tier.

---

## §9. Prohibiciones absolutas

- ❌ Telemetría, analytics o "phone home" de cualquier tipo.
- ❌ Mandar el audio **o el texto** a ninguna nube / servicio externo. El
  análisis IA es 100% local; no se añaden SDKs de nube ni API keys.
- ❌ Acceder a webcam o portapapeles.
- ⚠️ Micrófono y audio del sistema (loopback): **solo** en los modos "Grabar
  mi voz" y "Grabar reunión", **siempre** por acción explícita del usuario
  (botón) y, para reuniones, con su consentimiento marcado. El audio grabado
  se procesa 100% local y nunca sale del equipo. Nada de captura silenciosa
  ni automática.
- ❌ Romper el flujo local: con los modelos ya cacheados, la app debe
  transcribir **y analizar** sin internet.
- ❌ Mostrar jerga técnica cruda al usuario (ver `DESIGN.md`).

---

## §10. Roadmap

- [x] Autodetección de hardware + recomendación por memoria libre.
- [x] Fallback de descarga sin symlinks (Windows restringido).
- [x] Timestamps estilo YouTube + export SRT/VTT.
- [x] Análisis IA **100% local** (llama.cpp + Qwen/Llama), elegido por hardware.
- [x] Modelo `large-v3-turbo` + batched pipeline (GPU) + pista de vocabulario.
- [x] Modos de grabación local: "mi voz" (mic) y "reunión" (mic + loopback).
- [ ] Resumen map-reduce para audios largos (que exceden el contexto del modelo).
- [ ] Diarización (separar hablantes): "Tú vs los demás" por pistas + sherpa-onnx.
- [ ] Proceso por lotes (varios archivos / carpetas).

---

## §11. Cómo trabajar en este repo

1. Lee este archivo, `DESIGN.md` y el `README.md` antes de tocar nada.
2. Empieza por la lógica (`config` → módulo) con sus tests en verde antes de
   pasar a la UI.
3. No avances de módulo hasta tener test verde + docstrings completos.
4. Corre el gate de §6 antes de cada commit.
5. Verifica el flujo end-to-end con un archivo real antes de dar algo por hecho.
