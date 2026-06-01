# Restructure: serving-only `backend/`, clean frontend/backend split

**Date:** 2026-06-02
**Status:** approved, executed

## Goal

Turn the working-but-cluttered tree (a full CoVR training repo with `serve/`
buried inside, frontend at top level, weights/embeddings scattered) into a clean
project with a clear frontend/backend separation and only the code the serving
API actually needs.

## Decisions (from the user)

- **Scope:** serving-only. Vendor the minimal FDCA inference slice into a
  self-contained `backend/`; drop the CoVR training repo from this project.
- **CoVR fate:** delete it. Safe — it has remote `github.com:Skye-ye/CoVR.git`,
  `main` tracks `origin/main`, **zero unpushed commits**; the only local-only
  changes were files created this session and vendored into `backend/`.
- **Root rename:** `VEDIO_retriever_projetct_frontend` → `compose-video-retrieval`
  (path-safe kebab-case).
- **Big inputs:** keep in `backend/`, gitignored.

## Target layout

```
compose-video-retrieval/
├── README.md                 # top-level overview
├── .gitignore                # ignores backend/{.venv,weights,artifacts,data}, __pycache__
├── frontend/                 # unchanged static UI
├── backend/
│   ├── pyproject.toml        # lean serving deps (no hydra/lightning/pandas/transformers)
│   ├── README.md
│   ├── app.py runtime.py config.py build_index.py strip_ckpt.py text_utils.py
│   ├── model/                # vendored fdca slice (package)
│   │   ├── __init__.py base.py combiner.py
│   │   └── backbone/{__init__.py, clip.py, model.py}
│   ├── weights/   (gitignored)  fdca.ckpt, fdca_aim.serve.pth
│   ├── artifacts/ (gitignored)  index.faiss, gallery_raw.npy, id_map.json, metadata.json
│   └── data/      (gitignored)  video_embedding/
└── docs/
    ├── 前后端接口说明.md
    └── superpowers/specs/2026-06-02-restructure-design.md
```

## Vendoring + import rewire (only code edits)

- `model/combiner.py`: `from src.model.base import …` → `from .base import …`.
- `runtime.py`: drop the `sys.path`+`from src.model.fdca…` hack →
  `from model import Combiner`, `from text_utils import remove_neg` (insert
  `backend/` on `sys.path` for CWD-independence).
- `app.py`/`config.py`: `from serve import config` → `import config`; default
  paths point at `backend/weights` and `backend/artifacts`.
- New `__init__.py` for `model/` and `model/backbone/` (CoVR used namespace pkgs).
- `text_utils.py`: vendored `remove_neg` (drops the unused torchvision pull-in).

The serving slice depends only on torch/torchvision/clip/numpy/faiss(build-only)/
fastapi/uvicorn — not the training stack.

## Order of operations (destructive last)

1. Scaffold `backend/`, vendor + rewire, write support files (pyproject,
   text_utils, READMEs, root README, `.gitignore`, design doc).
2. Move weights/artifacts/data/docs into place (instant, same volume — no rebuild).
3. `uv sync` in `backend/`; **verify**: model loads + one real `/api/search` over
   the full 212k gallery.
4. Only after verification: **delete `CoVR/`**.
5. **Rename the root dir last** (renaming earlier would break the freshly-built
   venv's absolute paths); final smoke test at the new path.

## Run commands (after)

```bash
cd backend && uv sync
uv run uvicorn app:app --port 8000          # +FDCA_FRAMES_DIR=… for real thumbnails
uv run python strip_ckpt.py weights/fdca.ckpt -o weights/fdca_aim.serve.pth
uv run python build_index.py --emb-dir data/video_embedding --out-dir artifacts
```

## Notes

- Git: outer repo tracked only `frontend/` + docs; `backend/` code becomes new
  tracked content. Commit left to the user.
- Carried-over verified facts: CLIP = ViT-L/14 (768-d), search via numpy (faiss
  segfaults with torch on macOS), low-confidence threshold tunable, `fdca.ckpt`
  looks like a first-epoch checkpoint.
