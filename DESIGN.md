# 🎨 Sistema de diseño — Audicop "Studio" (claro)

> Fuente de verdad visual del proyecto. Antes de tocar la UI, lee esto.
> El frontend es **HTML/CSS/JS vanilla** servido por FastAPI — sin frameworks,
> sin build, sin CDN (funciona offline). Los tokens viven en
> [`frontend/styles.css`](frontend/styles.css) como variables CSS.

---

## 1. Filosofía

Audicop lo usa gente técnica y **no técnica**. El diseño (inspirado en apps de
reuniones tipo Meetily) prioriza, en orden:

1. **Biblioteca primero.** Sidebar con tus reuniones (buscar, reabrir, borrar);
   la pantalla principal es la reunión activa: **Notas ⟷ Transcripción**.
2. **Calma.** Superficies blancas, bordes suaves, un color de acción (azul) y
   el rojo reservado a "grabando".
3. **Lenguaje humano.** Nada de jerga (`int8_float16`, `cuda`) en primer plano;
   lo técnico va plegado.
4. **Honestidad.** Nada sale a la nube — y se dice. Lo que se guarda en disco
   (reuniones) se dice y se puede borrar. Si algo tarda, se avisa y se estima.

---

## 2. Tokens de color (CSS variables)

Definidos en `:root` de `frontend/styles.css`. Tema claro "Studio".

| Variable          | Valor     | Uso                                          |
|-------------------|-----------|----------------------------------------------|
| `--primary`       | `#2563eb` | Único acento de acción (botones, enlaces, [MM:SS]) |
| `--primary-soft`  | `#eff6ff` | Fondos suaves del primario (hover, selección) |
| `--bg`            | `#f6f7f8` | Lienzo de la app                              |
| `--surface`       | `#ffffff` | Tarjetas, sidebar, inputs                     |
| `--surface-2`     | `#f3f4f6` | Fondos secundarios (pestañas, search)         |
| `--text`          | `#111827` | Texto principal                               |
| `--text-2`        | `#6b7280` | Secundario / metadatos                        |
| `--border`        | `#e5e7eb` | Divisores estándar                            |
| `--rec`           | `#ef4444` | SOLO grabación (botón, punto pulsante)        |
| `--success`       | `#059669` | Estado positivo (GPU lista, privacidad)       |
| `--warning`       | `#b45309` | Avisos                                        |
| `--danger`        | `#dc2626` | Errores y borrar                              |

**Regla:** el azul es el único color de marca; el rojo significa exclusivamente
"grabando/borrar". Rebrandear = cambiar solo estas variables.

---

## 3. Tipografía

- Una sola familia: `Inter, system-ui, ...` (la del sistema; sin webfont externa
  para no depender de la red). Sin segundas tipografías.
- Jerarquía: `h1` (28px/700) título · `h2` (18px/700) paso · cuerpo 15px/400 ·
  `.muted` para metadatos. Pesos < 400 prohibidos (pierden legibilidad).
- Datos crudos (timestamps, salida) en `monospace`.

---

## 4. Layout (sidebar + vistas)

```
┌─ sidebar (290px) ──────┬─ main ─────────────────────────────────────┐
│ 🎙️ Audicop             │  VISTA "new" (por defecto)                  │
│ [＋ Nueva transcripción]│   ¿Qué transcribimos hoy?                   │
│ 🔎 Buscar reuniones…    │   [Subir | Grabar mi voz | Grabar reunión]  │
│ ─ Hoy ─                │   dropzone · ⚙️ opciones · ▶ Transcribir     │
│  • Sprint planning     │   progreso + texto en vivo                  │
│  • Llamada cliente 📝  │  ───────────────────────────────────────── │
│ ─ Ayer ─               │  VISTA "meeting" (al abrir/terminar)        │
│  • Nota de voz         │   [título editable]        meta   🗑        │
│                        │   (● Notas | Transcripción)  ← segmented    │
│ ✅ chip hardware        │   Notas: 🔒 pill · atajos IA · chat · nota   │
│ ⚙️/🔒 details plegados  │   Transcripción: [Texto|Tiempos|Exportar]   │
└────────────────────────┴────────────────────────────────────────────┘
          (grabando → píldora flotante inferior: ● 00:42 ⏸ ⏹)
```

- Dos vistas en `main`: `#view-new` y `#view-meeting` (`showView()` en `app.js`).
- La biblioteca (sidebar) agrupa por fecha (Hoy/Ayer/…), busca contra título,
  transcripción y notas, y marca 📝 si la reunión tiene nota guardada.
- Grabando, la píldora `.rec-bar` flota abajo-centro sobre cualquier vista.

---

## 5. Componentes (clases en `styles.css`)

- **Sidebar** (`.sidebar`, `.meeting-item`, `.meetings__group`): biblioteca de
  reuniones; activa con fondo `--primary-soft`.
- **Chip de estado** (`.banner`): compacto en el pie del sidebar — dónde corre y
  qué modelo. Verde con GPU. Detalle técnico en `<details>`.
- **Segmented** (`.seg`/`.seg__btn`): alterna Notas ⟷ Transcripción.
- **Pestañas** (`.tab` / `.tab-panel`): entrada (Subir/Grabar) y transcripción
  (Texto/Tiempos/Exportar). Controladas por `wireTabs()` en `app.js`.
- **Dropzone** (`.dropzone`): arrastrar o clic; resalta en `dragover`.
- **Botones**: `.btn--primary` (azul, acción), `.btn--rec` (rojo, SOLO grabar),
  `.btn--ghost` (secundario), `.quick-btn` (atajos IA, pill), `.icon-btn` (🗑).
- **Progreso** (`.progress`, `.live-text`): barra + texto en vivo durante el decode.
- **Alertas y pills**: `.alert--error/--warn/--ok`, `.pill--ok` (privacidad IA).
- **Chat/Notas** (`.bubble--user`/`--assistant`, `.note-toolbar`): burbujas con
  streaming; la última respuesta es "la nota" (se autoguarda, copia y exporta).
- **Barra de grabación** (`.rec-bar`): píldora flotante con punto pulsante.
- **Toast** (`.toast`): confirmaciones efímeras ("Copiado", "Reunión borrada").

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
