# Contribuir a Audicop

¡Gracias por tu interés! Las contribuciones son bienvenidas.

## Cómo empezar

1. Haz fork del repositorio y clónalo localmente.
2. Sincroniza dependencias (incluyendo las de desarrollo):
   ```bash
   uv sync --extra dev
   ```
   El primer `uv sync` también descarga la versión de Python indicada en
   `.python-version` si no la tienes.

## Antes de abrir un PR

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy audicop/hardware.py audicop/recommender.py audicop/audio.py audicop/transcriber.py
```

## Código de conducta

Sé amable. Asume buena fe. Lee la guía de estilo del README antes de
proponer cambios grandes.
