# Volume

## How to run the server

uv run gunicorn website.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
