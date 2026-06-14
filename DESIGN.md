# 🎨 Sistema de diseño — Audicop "Slate"

> Fuente de verdad visual del proyecto. Antes de añadir o cambiar UI, lee esto.
> Audicop usa **Streamlit**, no Tailwind/Next.js — así que "Slate" se aplica con
> el tema nativo de Streamlit + disciplina de componentes, no con clases CSS.

---

## 1. Filosofía

Audicop lo usa gente técnica y **no técnica**. El diseño prioriza, en este orden:

1. **Calma.** Una pantalla tranquila, oscura, sin ruido. El usuario llega a
   transcribir, no a leer documentación.
2. **Una decisión a la vez.** Lo esencial visible; lo avanzado plegado.
3. **Lenguaje humano.** Nada de jerga (`int8_float16`, `cuda`) en primer plano.
4. **Honestidad.** Si algo sale a la nube, se dice. Si va a tardar, se estima.

---

## 2. Tokens de color

Tema oscuro "Slate" aplicado en `.streamlit/config.toml [theme]`.

| Token                 | Valor       | Uso                                        |
|-----------------------|-------------|--------------------------------------------|
| `primaryColor`        | `#2563eb`   | Único acento de acción (botones, foco)     |
| `backgroundColor`     | `#0f172a`   | Fondo principal (slate 900)                |
| `secondaryBackground` | `#1e293b`   | Tarjetas, sidebar, contenedores (slate 800)|
| `textColor`           | `#e2e8f0`   | Texto principal (slate 200)                |

**Regla:** el azul `#2563eb` es el **único** color de marca. No introducir
segundos acentos decorativos. Los colores semánticos los pone Streamlit:

| Estado    | Componente Streamlit | Cuándo                                      |
|-----------|----------------------|---------------------------------------------|
| Éxito     | `st.success`         | "Listo", transcripción completada           |
| Info      | `st.info`            | Estimaciones, contexto neutro               |
| Aviso     | `st.warning`         | Descarga de modelo, audio > 3 h, privacidad |
| Error     | `st.error`           | Fallo de ffmpeg/modelo/IA — siempre amable  |

---

## 3. Tipografía

- Una sola familia (la sans por defecto del tema). **Sin** segundas tipografías.
- Jerarquía con tamaño/peso de Streamlit: `st.title` → `st.subheader` →
  `st.markdown` → `st.caption`.
- Datos numéricos (RAM, VRAM, tiempos) en `st.metric` o ``code`` inline.

---

## 4. Layout

```
┌──────────────┬────────────────────────────────────────────┐
│  Sidebar     │  Header: 🎙️ Audicop + tagline               │
│  (Opciones)  │                                             │
│              │  ✅ Banner de estado (1 línea) + expander    │
│  - Idioma    │                                             │
│  - Acción    │  Sube tu archivo  [Subir | Archivo local]   │
│  - VAD       │                                             │
│  - Avanzado  │  [ Transcribir ]                            │
│   (plegado)  │                                             │
│              │  Resultados: texto / timestamps / export    │
│              │  Panel IA (chat + quick actions)            │
│              │  🔒 Privacidad (plegado)                     │
└──────────────┴────────────────────────────────────────────┘
```

- **Sidebar** = ajustes. **Centro** = flujo principal (subir → transcribir → usar).
- `layout="centered"`: la columna principal no se estira en monitores anchos.

---

## 5. Patrones de componente

- **Banner de estado** (`st.success` de una línea): dónde se ejecuta, qué modelo,
  cuánto tarda. El detalle técnico va en un `st.expander` debajo, plegado.
- **Entradas en pestañas** (`st.tabs`): "Subir archivo" y "Archivo local". Nunca
  obligues a elegir entre jerga; las etiquetas explican el cuándo.
- **Resultados en pestañas**: Texto plano · Con timestamps · Exportar. Cada vista
  ofrece copiar (`st.code`) y/o descargar (`st.download_button`).
- **Panel IA**: proveedor + modelo + API key (`type="password"`) arriba; aviso de
  privacidad **antes** del primer envío; quick-actions como botones; chat con
  `st.chat_input`/`st.chat_message` y respuesta en streaming (`st.write_stream`).
- **Selectores legibles**: usa `format_func` para mostrar "Español", "GPU (NVIDIA)",
  "Large v3" — nunca `es`, `cuda`, `large-v3` crudos al usuario.

---

## 6. Tono de copy (UI en español)

- Frases cortas, imperativas y amables: "Sube tu archivo", "Listo", "Transcribiendo…".
- Explica el porqué cuando degradas algo: "usamos `base` para no saturar la RAM".
- Los errores siempre dan **el siguiente paso**, no solo el síntoma.
- Emojis con moderación, como anclas visuales (✅ 📥 🔒 ⚠️ 🎮 🧠), no decoración.

---

## 7. Accesibilidad y rendimiento

- Contraste AA: texto `#e2e8f0` sobre `#0f172a` cumple holgado.
- No bloquear la UI: trabajos largos van con `st.status` + barra de progreso + ETA.
- Streaming en el chat IA para feedback inmediato.
- Cachear lo caro (`st.cache_data` para hardware, `st.cache_resource` para el modelo).

---

## 8. Reglas (no negociables)

- ❌ Un segundo color de marca o acentos decorativos.
- ❌ Jerga técnica en primer plano (va en expanders/avanzado).
- ❌ Pesos de fuente < 400 ni segundas tipografías.
- ❌ Pantallas que obliguen a hacer scroll antes de poder subir un archivo.
- ✅ Rebrandear = cambiar solo `primaryColor` y los tokens de `[theme]`.
