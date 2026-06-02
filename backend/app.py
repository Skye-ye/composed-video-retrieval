"""FastAPI backend for the combined-video-retrieval frontend.

Implements the two endpoints the frontend (frontend/src/app.js) calls, exactly
per `前后端接口说明.md`:

    GET  /api/videos?cursor=&limit=
    POST /api/search

plus thumbnail serving — baked files at /static/, or dynamic mid-frames from the
gallery at /api/thumb/<id> when FDCA_FRAMES_DIR is set. The heavy lifting (model
+ gallery) lives in runtime.py; this module is just HTTP, validation, pagination,
and the business rules from doc §2.4.

Run::

    uvicorn app:app --host 0.0.0.0 --port 8000      # from the backend/ dir
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+faiss OpenMP on macOS
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make backend/ importable

import json  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.requests import Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
from runtime import FdcaAimRuntime  # noqa: E402

app = FastAPI(title="Combined Video Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve build_index.py's baked thumbnails (copy mode): artifacts/thumbs/<id>.jpg
# -> /static/thumbs/<id>.jpg
if config.ARTIFACTS_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(config.ARTIFACTS_DIR)), name="static")

# Dynamic thumbnails straight from the gallery frame folders (no copy, no rebuild):
# set FDCA_FRAMES_DIR and the server serves a mid-frame at /api/thumb/<video_id>.
# Frames are laid out per source dataset (hvu_frames/<id>/, an_frames/<id>/, ...),
# not flat, so FDCA_ANNOTATIONS_DIR provides the video_id -> real-subfolder map.
_FRAMES_ROOT = Path(config.FRAMES_DIR) if config.FRAMES_DIR else None
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_frame_cache: dict[str, str | None] = {}


def _load_frame_index(ann_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Map each gallery video_id to its frame subfolder via the annotation json.

    Reads every ``<split>/{id2file.json,id2path.json}`` pair (id2file: idx -> id,
    id2path: idx -> ``<dataset>_frames/<id>``) and joins them by index. Returns the
    ``{video_id: relpath}`` map and the distinct dataset prefixes, the latter used
    to probe ids the annotations don't cover.
    """
    rel: dict[str, str] = {}
    for split in sorted(p.name for p in ann_dir.iterdir() if p.is_dir()):
        f_path, p_path = ann_dir / split / "id2file.json", ann_dir / split / "id2path.json"
        if not (f_path.is_file() and p_path.is_file()):
            continue
        files = json.loads(f_path.read_text(encoding="utf-8"))
        paths = json.loads(p_path.read_text(encoding="utf-8"))
        for idx, vid in files.items():
            if idx in paths:
                rel[vid] = paths[idx]
    prefixes = sorted({r.split("/", 1)[0] for r in rel.values() if "/" in r})
    return rel, prefixes


_id_to_relpath: dict[str, str] = {}
_frame_prefixes: list[str] = []
if _FRAMES_ROOT is not None and config.ANNOTATIONS_DIR:
    _ann_dir = Path(config.ANNOTATIONS_DIR)
    if _ann_dir.is_dir():
        _id_to_relpath, _frame_prefixes = _load_frame_index(_ann_dir)


def _frame_candidates(video_id: str, root: Path) -> list[Path]:
    """Folders that might hold this video's frames, most-likely first.

    Authoritative annotation path when known; otherwise probe each known dataset
    subfolder (so extra frames on disk still resolve) and finally the flat layout.
    """
    rel = _id_to_relpath.get(video_id)
    if rel is not None:
        return [root / rel]
    cands = [root / pfx / video_id for pfx in _frame_prefixes]
    cands.append(root / video_id)
    return cands


def _resolve_frame(video_id: str) -> str | None:
    """Filesystem path to a representative (mid) frame for video_id, or None.

    Cached: the chosen filename is stable, so each folder is listed at most once.
    """
    if video_id in _frame_cache:
        return _frame_cache[video_id]
    frame = None
    if _FRAMES_ROOT is not None:
        for folder in _frame_candidates(video_id, _FRAMES_ROOT):
            if folder.is_dir():
                imgs = sorted(
                    f for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in _IMG_EXT
                )
                if imgs:
                    frame = str(imgs[len(imgs) // 2])
                    break
    _frame_cache[video_id] = frame
    return frame


def _thumb_url(video_id: str, meta: dict) -> str | None:
    """When the gallery is mounted, point at the dynamic route; otherwise fall
    back to whatever build_index baked into metadata (a /static path, or null)."""
    if _FRAMES_ROOT is not None:
        return f"/api/thumb/{video_id}"
    return meta.get("thumbnail_url")


runtime: FdcaAimRuntime | None = None


# --- error handling: uniform {"error": {"code", "message"}} (doc §2.4.4) -------

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message


@app.exception_handler(ApiError)
async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.on_event("startup")
def _load_runtime() -> None:
    global runtime
    runtime = FdcaAimRuntime(
        artifacts_dir=config.ARTIFACTS_DIR,
        ckpt_path=config.CKPT_PATH,
        clip_model_name=config.CLIP_MODEL_NAME,
        device=config.DEVICE,
    )


def _rt() -> FdcaAimRuntime:
    if runtime is None:  # pragma: no cover - startup guarantees this
        raise ApiError(503, "NOT_READY", "model runtime is still loading")
    return runtime


# --- /api/videos ---------------------------------------------------------------

def _video_card(rt: FdcaAimRuntime, video_id: str) -> dict:
    meta = rt.metadata.get(video_id, {})
    card = {
        "video_id": video_id,
        "thumbnail_url": _thumb_url(video_id, meta),
        "duration_sec": meta.get("duration_sec"),
    }
    if meta.get("title"):
        card["title"] = meta["title"]
    return card


@app.get("/api/videos")
def list_videos(cursor: str | None = None, limit: int = config.DEFAULT_PAGE_LIMIT):
    rt = _rt()
    limit = max(1, min(int(limit), 100))

    offset = 0
    if cursor:
        try:
            offset = max(0, int(cursor))
        except ValueError:
            raise ApiError(400, "INVALID_INPUT", "invalid cursor")

    ids = rt.row_to_id
    page = ids[offset : offset + limit]
    next_offset = offset + len(page)
    next_cursor = str(next_offset) if next_offset < len(ids) else None

    return {
        "videos": [_video_card(rt, vid) for vid in page],
        "next_cursor": next_cursor,
    }


# --- /api/search ---------------------------------------------------------------

class SearchRequest(BaseModel):
    reference_video_id: str
    modification_text: str
    retain_text: str | None = None
    exclude_text: str | None = None
    top_k: int = Field(default=config.DEFAULT_TOP_K)
    debug: bool = False


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


@app.post("/api/search")
def search(req: SearchRequest):
    rt = _rt()

    modification = (req.modification_text or "").strip()
    if not modification:
        raise ApiError(400, "INVALID_INPUT", "modification_text is empty")

    top_k = int(req.top_k)
    if top_k < 1 or top_k > config.MAX_TOP_K:
        raise ApiError(400, "INVALID_INPUT", f"top_k must be in [1, {config.MAX_TOP_K}]")

    if not rt.has_video(req.reference_video_id):
        raise ApiError(404, "NOT_FOUND", "reference video not found")

    ref_vec = rt.ref_vector(req.reference_video_id)
    query_vec, debug = rt.encode_query(
        ref_vec,
        modification,
        retain_text=req.retain_text,
        exclude_text=req.exclude_text,
        return_debug=req.debug,
    )

    # Over-fetch one so we can drop the reference video itself (doc §2.4.1).
    hits = rt.search(query_vec, top_k + 1)
    hits = [(vid, s) for vid, s in hits if vid != req.reference_video_id][:top_k]

    branch_vectors = debug["branch_vectors"] if (debug and req.debug) else None
    predicted_action = debug.get("predicted_action_class") if debug else None
    results = []
    for video_id, score in hits:
        meta = rt.metadata.get(video_id, {})
        item = {
            "video_id": video_id,
            "score": round(score, 4),
            "thumbnail_url": _thumb_url(video_id, meta),
            "duration_sec": meta.get("duration_sec"),
            "low_confidence": score < config.LOW_CONFIDENCE_THRESHOLD,
        }
        if meta.get("title"):
            item["title"] = meta["title"]
        if branch_vectors is not None:
            rvec = rt.norm_vector(video_id)
            item["debug"] = {
                "branch_scores": {
                    "retain": round(_cosine(rvec, branch_vectors["retain"]), 4),
                    "inject": round(_cosine(rvec, branch_vectors["inject"]), 4),
                    "exclude": round(_cosine(rvec, branch_vectors["exclude"]), 4),
                },
                "predicted_action_class": predicted_action,
            }
        results.append(item)

    return {"query_id": f"q_{uuid.uuid4().hex[:6]}", "results": results}


@app.get("/api/thumb/{video_id}")
def thumb(video_id: str):
    rt = _rt()
    # has_video() doubles as authorization + path-traversal guard: only ids that
    # exist in the gallery id_map are ever turned into a filesystem path.
    if _FRAMES_ROOT is None or not rt.has_video(video_id):
        raise ApiError(404, "NOT_FOUND", "thumbnail not available")
    frame = _resolve_frame(video_id)
    if frame is None:
        raise ApiError(404, "NOT_FOUND", "no frame for video")
    return FileResponse(frame, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/health")
def health():
    info = {
        "status": "ok",
        "gallery_size": len(_rt().row_to_id),
        "thumbnails": "gallery" if _FRAMES_ROOT is not None else "placeholders",
    }
    if _id_to_relpath:
        info["frame_index"] = {
            "mapped_ids": len(_id_to_relpath),
            "datasets": _frame_prefixes,
        }
    return info
