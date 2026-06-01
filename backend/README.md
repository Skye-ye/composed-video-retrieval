# backend — composed video retrieval API

Self-contained FastAPI service implementing the two endpoints the frontend
(`../frontend`) calls (`GET /api/videos`, `POST /api/search`) per
`../docs/前后端接口说明.md`. It bundles the minimal **FDCA + AIM** inference code
(the `model/` package, vendored from the CoVR training repo) so it runs without
the rest of that project.

```
frontend (app.js) ──HTTP──► app.py ──py──► runtime.py ──► FDCA Combiner (CLIP ViT-L/14)
                              │                 │
                              │                 └─► gallery (numpy matmul over gallery_raw.npy)
                              └─► /static/thumbs/<id>.jpg  or  /api/thumb/<id>
```

## Files

| path | role |
|---|---|
| `app.py`        | FastAPI: `/api/videos`, `/api/search`, `/api/thumb/<id>`, `/static`, `/health` |
| `runtime.py`    | loads model + in-memory gallery; `encode_query()` / `search()` (numpy) |
| `config.py`     | paths / threshold / device, all overridable via env vars |
| `build_index.py`| one-off: AIM embeddings → `artifacts/{index.faiss, gallery_raw.npy, id_map.json, metadata.json}` |
| `strip_ckpt.py` | one-off: slim a trained checkpoint to weights-only for serving |
| `text_utils.py` | `remove_neg` (vendored from the training repo) |
| `model/`        | vendored FDCA slice: `base.py`, `combiner.py`, `backbone/` (CLIP ViT-L/14) |
| `weights/`      | (gitignored) `fdca.ckpt` (raw) + `fdca_aim.serve.pth` (stripped, what the server loads) |
| `artifacts/`    | (gitignored) the built index the server loads at startup |
| `data/`         | (gitignored) `video_embedding/` — the raw embeddings, only needed to rebuild |

## Setup (one-time)

```bash
cd backend
uv sync            # creates .venv with the lean serving deps (no hydra/lightning/pandas)
```

Then produce the two gitignored inputs (already done on this machine — these are
only for rebuilding):

```bash
# 1. strip the training checkpoint → portable weights-only file
uv run python strip_ckpt.py weights/fdca.ckpt -o weights/fdca_aim.serve.pth
#    inspect first if unsure:  uv run python strip_ckpt.py weights/fdca.ckpt --dry-run

# 2. build the index from the AIM embeddings (212,642 videos, (8,768) f16 → 768-d)
uv run python build_index.py --emb-dir data/video_embedding --out-dir artifacts
#    fast demo gallery:                 add  --limit 3000
#    bake static thumbnails from frames: add  --video-root /path/to/video
```

## Run

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
# GPU:                FDCA_DEVICE=cuda  uv run uvicorn app:app --port 8000
# real thumbnails:    FDCA_FRAMES_DIR=/path/to/video  uv run uvicorn app:app --port 8000
curl http://localhost:8000/health
```

Env vars (see `config.py`): `FDCA_ARTIFACTS_DIR`, `FDCA_CKPT`, `FDCA_FRAMES_DIR`,
`FDCA_CLIP_MODEL` (default `ViT-L/14`), `FDCA_DEVICE` (`cpu`/`cuda`),
`FDCA_THRESHOLD` (low-confidence cutoff, default `0.3`), `FDCA_CORS_ORIGINS`.

## Notes / gotchas (verified end-to-end)

- **CLIP model is `ViT-L/14`, not `RN50x4`.** The checkpoint's `fc.weight` is
  `(768, 2304)` ⇒ `feature_dim=768` (ViT-L/14's text dim); the AIM embeddings are
  also 768-d. CoVR's `configs/model/fdca.yaml` said `RN50x4` (640) — stale;
  building with it makes the checkpoint fail to load.
- **Search uses numpy, not faiss (macOS OpenMP).** Importing faiss in the same
  process as torch loads a second OpenMP runtime and **segfaults the server** on
  macOS (verified SIGSEGV). So the server ranks with a numpy matmul over
  `gallery_raw.npy` — a flat 212k×768 search is ~0.16 s end-to-end. `build_index.py`
  still writes `index.faiss` (spec-compatible, usable by a Linux backend), but
  **the server never reads it**; faiss is a build-time-only dependency.
  `KMP_DUPLICATE_LIB_OK=TRUE` is set defensively but is no longer load-bearing.
- **Low-confidence threshold.** Cosine scores land ~0.25–0.30 with the current
  checkpoint, so most results trip the default `0.3` cutoff. Tune `FDCA_THRESHOLD`
  (e.g. `0.2`) for the demo.
- **Checkpoint sanity check.** `fdca.ckpt` reports `epoch: 0, best_score: -inf` —
  looks like a first-epoch `last.ckpt` rather than a finished `best.ckpt`. It loads
  and runs; confirm it's the checkpoint you meant to ship (may explain low scores).
- **Memory.** The server holds ~0.65 GB (gallery) + ~1.1 GB (model) + ViT-L/14
  (~1.5 GB) ≈ 3.3 GB RSS over the full 212k gallery. Use `--limit` when building
  for a lighter laptop demo.

## Thumbnails

`thumbnail_url` in every API response drives the frontend tiles (`null` →
placeholder). Two ways to fill it, no code changes:

**A. Dynamic from the gallery (recommended).** Point one env var at the per-video
frame folders; the server serves a mid-frame at `/api/thumb/<video_id>` — no index
rebuild, no copying:

```bash
FDCA_FRAMES_DIR=/path/to/video uv run uvicorn app:app --port 8000
```

Expected layout `FDCA_FRAMES_DIR/<video_id>/<frames>.jpg` (same ids as the `.npy`
files; server picks the middle frame and caches the choice). `/health` then reports
`"thumbnails":"gallery"`. Unknown ids 404 (membership check also blocks path
traversal). Unset → `thumbnail_url` is `null` → placeholders.

**B. Bake static thumbnails at build time.** `build_index.py --video-root <dir>`
copies one mid-frame per video into `artifacts/thumbs/<id>.jpg` and serves it from
`/static`. Self-contained but duplicates images. Mode A wins if both are set.

## Assumptions to confirm with the model team

The interface doc imagines a clean `encode_query(ref, modification, retain, exclude)`.
The real model (`model/combiner.py::Combiner.encode_query_features`) returns a
5-tuple, so `runtime.py` adapts it:

1. **`retain_text` → `caption_pos`**; when absent, `caption_pos =
   remove_neg(modification_text)`. **`exclude_text` has no model hook** and is
   accepted-but-ignored — confirm the intended mapping.
2. **Debug branch scores** (`retain`/`inject`/`exclude`) are cosines of each result
   against three real query-side vectors (`fusion_fea_high`, `ref_high_feature_mean`,
   `fusion_fea_high_token_negation`). No action-classifier head, so
   `predicted_action_class` is always `null`. Debug is demo-only; it doesn't affect
   ranking.

Ranking is faithful to the upstream CoVR eval protocol: retrieval vector = `out[0]`
(`fusion_fea_high`), cosine over L2-normalized vectors.
