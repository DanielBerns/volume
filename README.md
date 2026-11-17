# Plant Volume Estimator (Volumen)

A computer vision web application for estimating above-ground plant and vegetation volume from multi-view photographs (XY, YZ, XZ views). Supports calibrated metric 3D visual hull reconstruction ($\text{cm}^3$) and the upper/lower-bound projective volume estimation method of Chopin, Laga, & Miklavcic (2017) (`projective_bounds`), using a red reference trihedron.

## Installation

Ensure [uv](https://docs.astral.sh/uv/) is installed on your system.

```bash
git clone git@github.com:DanielBerns/volume.git
cd volume
uv sync
```

## Configuration

The application uses consolidated configuration in `pyproject.toml`.
Operational directories:
- Images and run metadata: `~/Info/volume/` (created automatically on first run).
- Default trihedron arm length: `20.0 cm` (`200 mm`, configurable via `arm_length_cm`).
- Default trihedron bar width: `0.5 cm` (`5 mm`, configurable via `arm_width_cm`).
- Volume estimation methods: `visual_hull` (3D voxel carving) and `projective_bounds` (Chopin et al., 2017).
- Output volume units: cubic centimeters ($\text{cm}^3$) and cubic meters ($\text{m}^3$).

## Initialization

Run automated verification tests and code checks:

```bash
uv run pytest
uv run ruff check .
```

## Execution

Start the production web server with Gunicorn and Uvicorn workers:

```bash
uv run gunicorn website.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Access the web interface at `http://localhost:8000`.
