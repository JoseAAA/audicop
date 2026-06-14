# Contribuir a Audicop

¡Gracias por tu interés! Las contribuciones son bienvenidas.

Antes de proponer cambios grandes, lee [AGENTS.md](AGENTS.md) (reglas
operativas, fuente de verdad) y [DESIGN.md](DESIGN.md) (sistema visual).

## Cómo empezar

1. Haz fork del repositorio y clónalo localmente.
2. Sincroniza dependencias (dev + IA para tocar el chat):
   ```bash
   uv sync --extra dev --extra ai
   ```
   El primer `uv sync` también descarga la versión de Python indicada en
   `.python-version` si no la tienes.

## Antes de abrir un PR

```bash
uv run pytest --cov
uv run ruff check .
uv run ruff format --check .
uv run mypy audicop/hardware.py audicop/recommender.py audicop/audio.py \
            audicop/transcriber.py audicop/formatting.py audicop/prompts/__init__.py \
            audicop/llm.py
```

`app.py` y `ui.py` están exentos de mypy y de cobertura (Streamlit no juega
bien con el tipado estricto). La lógica testeable vive en los demás módulos.

## Código de conducta

Sé amable. Asume buena fe.
