# Composed Video Retrieval

Pick a reference video from a fixed gallery, describe a change in natural language
("walking instead of running"), and retrieve the videos that best match the
composed query. Model: **FDCA + AIM** (Composed Video Retrieval).

```
┌───────────┐   HTTP    ┌───────────────┐   in-process   ┌──────────────────┐
│ frontend/ │ ────────► │   backend/    │ ─────────────► │ FDCA model (CLIP │
│ static UI │ ◄──────── │  FastAPI API  │ ◄───────────── │ ViT-L/14) + AIM  │
└───────────┘           └───────────────┘                │ gallery (numpy)  │
                                                          └──────────────────┘
```

## Layout

| path | what |
|---|---|
| `frontend/` | dependency-free static UI (`index.html` + `src/`). Calls `/api/videos` and `/api/search`. |
| `backend/`  | self-contained FastAPI service + the vendored FDCA model slice. See `backend/README.md`. |
| `docs/`     | interface contract (`前后端接口说明.md`) and design notes. |

Large files live under `backend/` but are gitignored: `weights/` (checkpoints),
`artifacts/` (the built index), `data/` (raw embeddings). See `backend/README.md`
for how to (re)produce them.

## Quick start

```bash
# backend (first run creates the env; ~lean, no training deps)
cd backend && uv sync
uv run uvicorn app:app --host 127.0.0.1 --port 8000
#   real thumbnails on a box that has the gallery frames:
#   FDCA_FRAMES_DIR=/path/to/video uv run uvicorn app:app --port 8000

# frontend (separate terminal)
cd frontend && python3 -m http.server 5173
```

Then add this before `src/app.js` in `frontend/index.html` and open
`http://localhost:5173/`:

```html
<script>window.APP_CONFIG = { API_BASE: "http://127.0.0.1:8000" };</script>
```

`frontend/?mock=1` runs the UI against built-in demo data with no backend.

The model training/eval code is **not** in this repo — it lives in the separate
CoVR project. `backend/` only vendors the minimal inference slice it needs.
