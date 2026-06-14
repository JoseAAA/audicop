# 🎙️ Audicop

> Suelta cualquier audio o vídeo. Recibe el texto, con marcas de tiempo, y
> analízalo con IA. Todo local, sin configurar nada.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![faster-whisper](https://img.shields.io/badge/engine-faster--whisper-0f172a.svg)](https://github.com/SYSTRAN/faster-whisper)
[![uv](https://img.shields.io/badge/deps-uv-2563eb.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

App web **local** que transcribe y traduce audio/vídeo con Whisper. Autodetecta
tu hardware y elige el mejor modelo que tu equipo puede correr — sin que tengas
que pensarlo. Además te da timestamps estilo YouTube, export a SRT/VTT y un chat
con IA (tu propia API key) para resumir y analizar lo dicho.

Es **un solo proceso local** (FastAPI + uvicorn sirviendo un frontend propio).
**Sin Docker, sin Node, sin build** — doble clic y listo.

---

## 📋 Contenido

- [🚀 Instalación](#-instalación)
- [✨ Funciones](#-funciones)
- [🧠 Cómo funciona](#-cómo-funciona)
- [🖥️ Hardware soportado](#️-hardware-soportado)
- [🎞️ Formatos y límites](#️-formatos-y-límites)
- [🤖 Análisis con IA](#-análisis-con-ia)
- [🔐 Privacidad](#-privacidad)
- [🩺 Problemas comunes](#-problemas-comunes)
- [🧰 Stack](#-stack)
- [🗺️ Roadmap](#️-roadmap)
- [👤 Autor](#-autor)
- [📜 License](#-license)

---

## 🚀 Instalación

### Windows (la mayoría de usuarios)

1. **Descarga el proyecto.** Botón verde "Code" → "Download ZIP". Descomprime
   donde quieras.
2. **Doble clic en `scripts\run.bat`.** Se abre una ventana que va informando
   de cada paso.
3. **Espera 5–10 minutos** la primera vez (instala todo y descarga el modelo).
   Las siguientes veces arranca en segundos.
4. Se abre tu navegador en `http://localhost:8000`. **Listo.**

### macOS / Linux

```bash
git clone https://github.com/JoseAAA/audicop.git
cd audicop
./scripts/run.sh
```

### ¿Qué hace el script solo?

1. Instala [`uv`](https://docs.astral.sh/uv/) (gestor de Python rápido) si no lo tienes.
2. Detecta GPU NVIDIA y, si la hay, instala las librerías CUDA automáticamente.
3. Crea un entorno aislado con dependencias pinchadas vía `uv.lock`.
4. Lanza la app.

> **No necesitas** Python instalado (uv lo trae), ni ffmpeg (va empaquetado), ni
> CUDA toolkit (solo el driver NVIDIA). El chat con IA viene incluido.

---

## ✨ Funciones

- 🤖 **Autodetección de hardware** — elige modelo y `compute_type` por ti, según
  la memoria **libre** (no la total), para no ahogar tu equipo.
- 🎬 **Multi-formato** — audio (mp3, wav, m4a, ogg, flac, aac) y vídeo (mp4, mkv,
  mov, avi, webm — extrae sólo el audio).
- ⏱️ **Timestamps estilo YouTube** — cada línea con el minuto en que se dijo.
- ⬇️ **Export** — descarga `.txt`, `.srt` y `.vtt`.
- 🧠 **Chat con IA (BYO key)** — resume, saca tareas o pregunta lo que quieras
  sobre el audio con OpenAI o Gemini.
- 📦 **Sin dependencias de sistema** — ffmpeg y CUDA vienen vía pip.
- 🔒 **Local y privado** — la transcripción nunca sale de tu equipo.
- 🌍 **Multi-idioma** — autodetección o forzado (es, en, pt, fr, it, de).

---

## 🧠 Cómo funciona

```
 Navegador (frontend vanilla)          Backend (FastAPI · uvicorn, local)
┌───────────────────────────┐  HTTP   ┌───────────────────────────────────────┐
│ Subir / Ruta → progreso    │ ─────▶  │ ffmpeg → faster-whisper → segments      │
│ (SSE en vivo) → resultado  │ ◀─SSE─  │ (16kHz mono)   (CTranslate2)            │
│ → chat IA                  │         │ chat → OpenAI/Gemini (tu key)           │
└───────────────────────────┘         └───────────────────────────────────────┘
       (todo local salvo el chat IA y la descarga inicial del modelo)
```

1. Subes un archivo (≤ 2 GB) o pegas una ruta local (sin límite de tamaño).
2. El backend extrae el audio a 16 kHz mono con el ffmpeg empaquetado.
3. faster-whisper decodifica y va emitiendo segmentos; el progreso llega al
   navegador en vivo por SSE.
4. El frontend los muestra (texto / timestamps / export) y ofrece el chat con IA.
5. Los temporales se borran al terminar.

---

## 🖥️ Hardware soportado

Audicop elige el modelo según la memoria **libre** en el momento de detección.
Tener 16 GB de RAM no significa poder dedicarlos todos: el SO y otras apps
consumen una parte, y respetarla evita que el equipo se ahogue.

| Recurso libre                     | model_size  | compute_type   |
|-----------------------------------|-------------|----------------|
| GPU CUDA, VRAM libre ≥ 8 GB       | large-v3    | float16        |
| GPU CUDA, VRAM libre 4–8 GB       | large-v3    | int8_float16   |
| GPU CUDA, VRAM libre 2.5–4 GB     | medium      | int8_float16   |
| GPU CUDA, VRAM libre < 2.5 GB     | small       | int8_float16   |
| Solo CPU, RAM libre ≥ 6 GB        | small       | int8           |
| Solo CPU, RAM libre 3–6 GB        | base        | int8           |
| Solo CPU, RAM libre < 3 GB        | tiny        | int8           |

> Abre **Modo avanzado** en la barra lateral para forzar otro modelo, o cierra
> apps y recarga para que recalcule con más memoria libre.

---

## 🎞️ Formatos y límites

- **Audio:** mp3, wav, m4a, ogg, flac, aac
- **Vídeo:** mp4, mkv, mov, avi, webm — *se extrae sólo la pista de audio*.
- **Duración:** hasta **3 horas** probadas (Whisper usa ventanas de 30 s, así que
  la VRAM no crece con la duración).
- **Subida:** hasta **2 GB** por la pestaña "Subir archivo". ¿Vídeo más grande?
  usa **"Archivo local"** y pega la ruta absoluta — Audicop lee del disco, sin
  subida ni límite.

### ¿Cuánto tarda? (referencia, 1 hora de audio)

| Hardware                  | Modelo              | Estimado |
|---------------------------|---------------------|----------|
| GPU NVIDIA (≥ 4 GB libre) | `large-v3` int8_fp16| ~10 min  |
| GPU NVIDIA gama media     | `medium`            | ~6 min   |
| Solo CPU, 16 GB RAM       | `small` int8        | ~60 min  |
| Solo CPU, < 8 GB RAM      | `tiny` int8         | ~12 min  |

---

## 🤖 Análisis con IA

Tras transcribir, aparece el panel **"Analiza con IA"**:

1. Elige proveedor: **OpenAI** o **Google Gemini** (capa gratis generosa).
2. Pega tu **API key** (se queda solo en memoria; nunca se guarda en disco).
3. Usa los atajos (Resumen, Puntos clave, Tareas y acuerdos, Acta) o escribe tu
   propia pregunta. Las respuestas **citan los minutos** `[MM:SS]`.

> Consigue tu key gratis en [Google AI Studio](https://aistudio.google.com/apikey)
> o en [OpenAI](https://platform.openai.com/api-keys).

¿Prefieres otra IA? Copia el texto (con o sin timestamps) y pégalo en Claude,
ChatGPT, etc.

---

## 🔐 Privacidad

| Acción                    | ¿Sale a la red?                                        |
|---------------------------|--------------------------------------------------------|
| Detección de hardware     | ❌ (psutil, nvidia-smi, platform — solo lectura)       |
| Conversión + transcripción| ❌ (local; el modelo solo se descarga la 1ª vez)       |
| **Chat con IA**           | ✅ Envía el **texto** de la transcripción al proveedor  |
|                           | elegido con **tu** API key. El audio nunca se sube.    |

La API key vive solo en la sesión: **nunca** se escribe a disco ni a logs.
Sin telemetría, sin analytics, sin acceso a webcam/micrófono/portapapeles.
Detalle completo en [AGENTS.md](AGENTS.md) §3.

---

## 🩺 Problemas comunes

| Problema | Solución |
|----------|----------|
| El modelo tarda en la 1ª descarga | Normal (`large-v3` ~3 GB). Luego sale de caché. |
| `WinError 1314` / "privilegio requerido" | Audicop ya lo maneja: descarga sin symlinks a `~/.cache/audicop/models`. Si persiste, borra `~/.cache/huggingface` y reabre. |
| No detecta mi GPU NVIDIA | Verifica que `nvidia-smi` funciona. Relanza `run.bat`/`run.sh`: instala CUDA solo. |
| `ffmpeg failed to convert` | El origen está corrupto o usa un códec raro. Reconviértelo o ábrelo en VLC. |
| `CUDA out of memory` | Abre **Modo avanzado** y baja de modelo (`medium`/`small`). |
| El chat IA dice "no está instalado" | Reinstala dependencias: `uv sync` (las libs de IA vienen incluidas). |

---

## 🧰 Stack

| Capa            | Tecnología                                              |
|-----------------|---------------------------------------------------------|
| Backend         | FastAPI + uvicorn (un solo proceso local)               |
| Frontend        | HTML/CSS/JS vanilla, tema "Slate" — ver [DESIGN.md](DESIGN.md) (sin Node) |
| Motor ASR       | faster-whisper (CTranslate2) · modelo Whisper de OpenAI |
| Audio           | imageio-ffmpeg (binario empaquetado)                    |
| Hardware        | psutil + nvidia-smi                                     |
| GPU (opcional)  | nvidia-cublas-cu12 + nvidia-cudnn-cu12                  |
| IA              | openai + google-genai (bring-your-own-key)              |
| Tooling         | uv · ruff · mypy · pytest                               |

Créditos: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Whisper](https://github.com/openai/whisper) (OpenAI),
[FastAPI](https://fastapi.tiangolo.com/), [uv](https://docs.astral.sh/uv/).

---

## 🗺️ Roadmap

- [x] Autodetección de hardware + recomendación por memoria libre.
- [x] Fallback de descarga sin symlinks (Windows restringido).
- [x] Timestamps estilo YouTube + export SRT/VTT.
- [x] Chat / análisis con IA (OpenAI/Gemini, BYO key).
- [ ] Diarización (separar hablantes).
- [ ] Proceso por lotes (varias carpetas).
- [ ] Más proveedores IA (Anthropic, modelos locales vía Ollama).

Issues y PRs bienvenidos. Convenciones del proyecto en [AGENTS.md](AGENTS.md)
y [DESIGN.md](DESIGN.md).

---

## 👤 Autor

**JoseAAA** · [github.com/JoseAAA](https://github.com/JoseAAA)

---

## 📜 License

MIT — ver [LICENSE](LICENSE).