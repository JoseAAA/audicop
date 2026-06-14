# 🎨 Sistema de diseño — Audicop "Slate"

> Fuente de verdad visual del proyecto. Antes de tocar la UI, lee esto.
> El frontend es **HTML/CSS/JS vanilla** servido por FastAPI — sin frameworks,
> sin build, sin CDN (funciona offline). Los tokens viven en
> [`frontend/styles.css`](frontend/styles.css) como variables CSS.

---

## 1. Filosofía

Audicop lo usa gente técnica y **no técnica**. El diseño prioriza, en orden:

1. **Guía por pasos.** La pantalla acompaña: 1 Sube → 2 Transcribe → 3 Resultado
   → 4 IA. Cada paso se habilita cuando toca; nada abruma de golpe.
2. **Calma.** Fondo oscuro slate, sin ruido, un solo color de acción.
3. **Lenguaje humano.** Nada de jerga (`int8_float16`, `cuda`) en primer plano;
   lo técnico va plegado.
4. **Honestidad.** Si algo sale a la nube (chat IA), se dice. Si va a tardar, se estima.

---

## 2. Tokens de color (CSS variables)

Definidos en `:root` de `frontend/styles.css`. Tema oscuro "Slate".

| Variable             | Valor       | Uso                                        |
|----------------------|-------------|--------------------------------------------|
| `--primary`          | `#2563eb`   | Único acento de acción (botones, foco)     |
| `--primary-hover`    | `#1d4ed8`   | Hover del primario                         |
| `--bg`               | `#0f172a`   | Lienzo principal (slate-900)               |
| `--surface`          | `#1e293b`   | Tarjetas, inputs (slate-800)               |
| `--surface-raised`   | `#334155`   | Hover, dropdowns (slate-700)               |
| `--text`             | `#f8fafc`   | Texto principal                            |
| `--text-2`           | `#cbd5e1`   | Secundario / descripciones                 |
| `--text-muted`       | `#94a3b8`   | Etiquetas de bajo contraste                |
| `--border`           | `#334155`   | Divisores estándar                         |
| `--success`          | `#10b981`   | Estado positivo                            |
| `--warning`          | `#f59e0b`   | Aviso (privacidad, descarga, audio largo)  |
| `--danger`           | `#f43f5e`   | Error                                       |

**Regla:** el azul `#2563eb` es el **único** color de marca. No introducir
segundos acentos decorativos. Rebrandear = cambiar solo estas variables.

---

## 3. Tipografía

- Una sola familia: `Inter, system-ui, ...` (la del sistema; sin webfont externa
  para no depender de la red). Sin segundas tipografías.
- Jerarquía: `h1` (28px/700) título · `h2` (18px/700) paso · cuerpo 15px/400 ·
  `.muted` para metadatos. Pesos < 400 prohibidos (ilegibles en oscuro).
- Datos crudos (timestamps, salida) en `monospace`.

---

## 4. Layout

```
┌───────────────────────────────────────────────┐
│ 🎙️ Audicop  +  tagline                          │
│ ✅ banner de estado (1 línea) + detalles plegados│
│                                                 │
│ ┌─ Paso 1 · Sube tu archivo ─────────────────┐ │
│ │  [Subir | Archivo local]  dropzone          │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─ Paso 2 · Transcribe (se habilita) ────────┐ │
│ │  ⚙️ opciones (plegado)   [ ▶ Transcribir ]   │ │
│ │  barra de progreso + texto en vivo          │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─ Paso 3 · Resultado ───────────────────────┐ │
│ │  [Texto | Tiempos | Exportar]               │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─ Paso 4 · Analiza con IA (opcional) ───────┐ │
│ │  proveedor·modelo·key  atajos  chat          │ │
│ └─────────────────────────────────────────────┘ │
│ 🔒 Privacidad (plegado)                          │
└───────────────────────────────────────────────┘
```

- Columna central centrada, `max-width: 760px` (no se estira en monitores anchos).
- Cada paso es una **tarjeta** (`.step`). Los pasos no disponibles van con
  `.is-disabled` (atenuados, sin interacción) hasta que el anterior se completa.

---

## 5. Componentes (clases en `styles.css`)

- **Banner de estado** (`.banner`): una línea — dónde corre, qué modelo, cuánto
  tarda. Verde (`--ok`) con GPU, neutro con CPU. Detalle técnico en `<details>`.
- **Pasos** (`.step`, `.step__num`): tarjeta numerada; `.is-disabled` hasta su turno.
- **Pestañas** (`.tab` / `.tab-panel`): entrada (Subir/Local) y resultado
  (Texto/Tiempos/Exportar). Controladas por `wireTabs()` en `app.js`.
- **Dropzone** (`.dropzone`): arrastrar o clic; resalta en `dragover`.
- **Botones**: `.btn--primary` (azul, acción principal), `.btn--ghost`
  (secundario), `.quick-btn` (atajos de IA, pill).
- **Progreso** (`.progress`, `.live-text`): barra + texto en vivo durante el decode.
- **Alertas**: `.alert--error`, `.alert--warn` (privacidad), `.alert--info`.
- **Chat** (`.bubble--user` / `.bubble--assistant`): burbujas; respuesta en streaming.
- **Toast** (`.toast`): confirmaciones efímeras ("Copiado", "Descargando modelo").

---

## 6. Tono de copy (UI en español)

- Frases cortas e imperativas: "Sube tu archivo", "Transcribir", "Listo".
- Explica el porqué al degradar: "usamos `base` para no saturar la RAM".
- Los errores dan **el siguiente paso**, no solo el síntoma.
- Emojis con moderación, como anclas (✅ 📤 ⏱️ 🔒 ⚠️ 🤖), no decoración.
- Selectores legibles: "Español", "GPU (NVIDIA)", "Large v3" — nunca `es`,
  `cuda`, `large-v3` crudos (mapeados en `LABELS` de `app.js`).

---

## 7. Reglas (no negociables)

- ❌ Un segundo color de marca o acentos decorativos.
- ❌ Jerga técnica en primer plano (va en `<details>` / Modo avanzado).
- ❌ Webfonts/JS/CSS desde CDN (rompe el offline). Todo vendorizado/local.
- ❌ Frameworks de frontend o paso de build (sin Node).
- ✅ Rebrandear = cambiar solo las variables CSS de `:root`.
