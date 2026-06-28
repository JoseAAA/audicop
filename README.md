# 🎙️ Audicop

> Suelta cualquier audio o vídeo. Recibe el texto, con marcas de tiempo, y
> analízalo con IA. Todo local, sin configurar nada.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![faster-whisper](https://img.shields.io/badge/engine-faster--whisper-0f172a.svg)](https://github.com/SYSTRAN/faster-whisper)
[![uv](https://img.shields.io/badge/deps-uv-2563eb.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

App web **local** que transcribe y traduce audio/vídeo con Whisper — sube un
archivo, **graba tu voz** o **graba una reunión** (Meet/Teams/Zoom). Autodetecta
tu hardware y elige el mejor modelo que tu equipo puede correr, sin que tengas
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

**Prerrequisitos:** solo **Git**. Ni Python, ni Node, ni Docker, ni ffmpeg — el
script instala todo (incluido Python, vía `uv`) en una carpeta aislada del
proyecto, sin tocar tu sistema.

```bash
# 1. Clonar
git clone https://github.com/JoseAAA/audicop.git
cd audicop
```

```bash
# 2. Arrancar (instala dependencias la 1ª vez y abre el navegador solo)
scripts\start.cmd          # Windows  (doble clic o desde la terminal)
./scripts/start.sh         # Linux / macOS
```

La 1ª vez tarda 5–10 min (instala todo y descarga el modelo); las siguientes,
segundos. Se abre solo en **http://localhost:8000**.

> **Windows — usa `scripts\start.cmd`** (doble clic). Funciona aunque tu equipo
> bloquee scripts de PowerShell no firmados (políticas corporativas
> *AllSigned*/*RemoteSigned*, o archivos marcados como "descargados" por estar
> en OneDrive): el `.cmd` lanza el instalador de una forma que la política de
> ejecución no restringe. Si prefieres PowerShell directo y no está bloqueado,
> `.\scripts\start.ps1` sigue funcionando.
>
> 💡 Si puedes, **clona el proyecto fuera de OneDrive** (p. ej. `C:\dev\audicop`):
> evita que OneDrive sincronice el entorno virtual y la marca de "descargado".

<details>
<summary>¿Qué hace el script por debajo?</summary>

1. Instala [`uv`](https://docs.astral.sh/uv/) (gestor de Python) si no lo tienes.
2. Detecta tu GPU NVIDIA y, si la hay, añade el soporte CUDA automáticamente.
3. Crea un entorno aislado con versiones fijas (`uv.lock`).
4. Levanta el servidor local (FastAPI + uvicorn) y abre el navegador.

Todo queda dentro de la carpeta del proyecto. **No toca tu Python del sistema.**
</details>

---

## ✨ Funciones

- 🎚️ **Tres formas de obtener audio** — (1) **subir** un archivo (o pegar la ruta
  de un vídeo grande), (2) **grabar tu voz** (nota/dictado) o (3) **grabar una
  reunión** (lo que dicen los demás + tu micrófono), con **Pausa** para no
  grabar momentos privados. Detecta cuándo estás en una reunión (Meet, Teams o
  Zoom, por el uso del micrófono) y te avisa para grabar. Todo 100% local.
- 🤖 **Autodetección de hardware** — elige modelo y `compute_type` por ti, según
  la memoria **libre** (no la total), para no ahogar tu equipo.
- ⚡ **Whisper turbo + batched** — usa `large-v3-turbo` (calidad casi `large-v3`,
  mucho más rápido) y el *batched pipeline* en GPU para acelerar el decode.
- 🗣️ **Pista de vocabulario** — escribe nombres, marcas o jerga de tu grabación
  y Whisper los transcribe con más precisión (`initial_prompt`).
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
│ Subir / Grabar → progreso  │ ─────▶  │ ffmpeg → faster-whisper → segments      │
│ (SSE en vivo) → resultado  │ ◀─SSE─  │ (16kHz mono)   (CTranslate2)            │
│ → chat IA                  │         │ chat → OpenAI/Gemini (tu key)           │
└───────────────────────────┘         └───────────────────────────────────────┘
       (todo local salvo el chat IA y la descarga inicial del modelo)
```

1. Eliges la entrada: **subes** un archivo (≤ 2 GB) o pegas una ruta local (sin
   límite), **grabas tu voz**, o **grabas una reunión** (mic + audio del sistema).
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

| Recurso libre                     | model_size       | compute_type   |
|-----------------------------------|------------------|----------------|
| GPU CUDA, VRAM libre ≥ 6 GB       | large-v3-turbo   | float16        |
| GPU CUDA, VRAM libre 2.5–6 GB     | large-v3-turbo   | int8_float16   |
| GPU CUDA, VRAM libre 1.5–2.5 GB   | small            | int8_float16   |
| GPU CUDA, VRAM libre < 1.5 GB     | base             | int8_float16   |
| Solo CPU, RAM libre ≥ 6 GB        | small            | int8           |
| Solo CPU, RAM libre 3–6 GB        | base             | int8           |
| Solo CPU, RAM libre < 3 GB        | tiny             | int8           |

> **`large-v3-turbo`** es la versión con decoder destilado de `large-v3`:
> calidad casi idéntica en los idiomas comunes (es/en…) pero **mucho más
> rápida** y con menos VRAM. En GPU se usa además el *batched pipeline* de
> faster-whisper para acelerar aún más.
>
> Abre **Opciones → Modo avanzado** (Paso 2) para forzar otro modelo
> (`large-v3` para máxima calidad, o turbo en CPU si priorizas precisión sobre
> velocidad), o cierra apps y recarga para que recalcule con más memoria libre.

---

## 🎞️ Formatos y límites

- **Audio:** mp3, wav, m4a, ogg, flac, aac
- **Vídeo:** mp4, mkv, mov, avi, webm — *se extrae sólo la pista de audio*.
- **Duración:** hasta **3 horas** probadas (Whisper usa ventanas de 30 s, así que
  la VRAM no crece con la duración).
- **Subida:** hasta **2 GB** por la pestaña **Subir**. ¿Vídeo más grande? dentro
  de **Subir**, despliega *"¿Vídeo grande? Pega la ruta del archivo"* y pega la
  ruta absoluta — Audicop lee del disco, sin subida ni límite.

### ¿Cuánto tarda? (referencia, 1 hora de audio)

| Hardware                  | Modelo                   | Estimado |
|---------------------------|--------------------------|----------|
| GPU NVIDIA (≥ 6 GB libre) | `large-v3-turbo` float16 | ~4 min   |
| GPU NVIDIA (2.5–6 GB)     | `large-v3-turbo` int8    | ~4 min   |
| Solo CPU, 16 GB RAM       | `small` int8             | ~60 min  |
| Solo CPU, < 8 GB RAM      | `tiny` int8              | ~12 min  |

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
| Detección de reunión      | ❌ (lee del registro qué app usa el micrófono)         |
| Grabación (mic / sistema) | ❌ (soundcard local; el WAV se queda en tu equipo)     |
| Conversión + transcripción| ❌ (local; el modelo solo se descarga la 1ª vez)       |
| **Chat con IA**           | ✅ Envía el **texto** de la transcripción al proveedor  |
|                           | elegido con **tu** API key. El audio nunca se sube.    |

La API key vive solo en la sesión: **nunca** se escribe a disco ni a logs.
Sin telemetría, sin analytics, sin acceso a webcam ni portapapeles. El
**micrófono y el audio del sistema** solo se capturan en los modos de
**grabación**, por tu acción explícita (y consentimiento, en reuniones); el
audio grabado nunca sale del equipo. El servidor escucha solo en `localhost`
y rechaza peticiones de orígenes externos (anti-CSRF). Detalle en
[AGENTS.md](AGENTS.md) §3.

---

## 🩺 Problemas comunes

| Problema | Solución |
|----------|----------|
| El modelo tarda en la 1ª descarga | Normal (`large-v3` ~3 GB). Luego sale de caché. |
| `WinError 1314` / "privilegio requerido" | Audicop ya lo maneja: descarga sin symlinks a `~/.cache/audicop/models`. Si persiste, borra `~/.cache/huggingface` y reabre. |
| No detecta mi GPU NVIDIA | Verifica que `nvidia-smi` funciona. Relanza `start.cmd`/`start.sh`: instala CUDA solo. |
| `start.ps1` "no está firmado digitalmente" | Política de PowerShell. Usa **`scripts\start.cmd`** (doble clic): evita esa restricción. |
| `ffmpeg failed to convert` | El origen está corrupto o usa un códec raro. Reconviértelo o ábrelo en VLC. |
| `CUDA out of memory` | Abre **Modo avanzado** y baja de modelo (`medium`/`small`). |
| El chat IA dice "no está instalado" | Reinstala dependencias: `uv sync` (las libs de IA vienen incluidas). |
| La reunión no capta a los demás | El loopback graba lo que suena por tus **altavoces**. Con auriculares Bluetooth o salidas raras puede no capturarse; usa los altavoces del equipo. |
| Me silencié en Meet pero igual me grabó | El micrófono se graba a nivel del sistema; silenciarte en la app no lo detiene. Usa **Pausar** o desmarca "Incluir mi micrófono". |
| "Este equipo no tiene captura de audio disponible" | No hay dispositivos de audio (o reiniciaste el server con código viejo). Reinicia con `start.cmd`/`start.sh`. |

---

## 🧰 Stack

| Capa            | Tecnología                                              |
|-----------------|---------------------------------------------------------|
| Backend         | FastAPI + uvicorn (un solo proceso local)               |
| Frontend        | HTML/CSS/JS vanilla, tema "Slate" — ver [DESIGN.md](DESIGN.md) (sin Node) |
| Motor ASR       | faster-whisper (CTranslate2) · modelo Whisper de OpenAI |
| Audio           | imageio-ffmpeg (binario empaquetado)                    |
| Grabación       | soundcard (mic + loopback del sistema, BSD-3, sin torch)|
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
- [x] `large-v3-turbo` + batched (GPU) + pista de vocabulario.
- [x] Grabación local: tu voz y reuniones (mic + audio del sistema).
- [ ] Diarización (separar hablantes): "Tú vs los demás" + sherpa-onnx.
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