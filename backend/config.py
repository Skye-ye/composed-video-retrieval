"""Serving configuration, overridable via environment variables.

Defaults assume the standard layout: artifacts in ``backend/artifacts`` and the
stripped checkpoint at ``backend/weights/fdca_aim.serve.pth``. Override any of
these without editing code, e.g.::

    FDCA_ARTIFACTS_DIR=/data/artifacts \\
    FDCA_CKPT=/data/fdca_aim.serve.pth \\
    FDCA_DEVICE=cuda \\
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ARTIFACTS_DIR = Path(os.getenv("FDCA_ARTIFACTS_DIR", _HERE / "artifacts"))
CKPT_PATH = Path(os.getenv("FDCA_CKPT", _HERE / "weights" / "fdca_aim.serve.pth"))

# Optional: root dir of per-video frame folders (the gallery, e.g. <dataset>/video).
# When set, the server serves a mid-frame per video at /api/thumb/<video_id> — no
# index rebuild, no copying. Leave empty (default) → thumbnail_url is null and the
# frontend shows placeholder tiles. Set this when you deploy on the Linux box that
# has the gallery:  FDCA_FRAMES_DIR=/path/to/dataset/video
FRAMES_DIR = os.getenv("FDCA_FRAMES_DIR", "")
# ViT-L/14 (feature_dim 768) — matches the trained checkpoint and the 768-dim
# AIM embeddings. NOT RN50x4 (640) from configs/model/fdca.yaml, which is stale.
CLIP_MODEL_NAME = os.getenv("FDCA_CLIP_MODEL", "ViT-L/14")
DEVICE = os.getenv("FDCA_DEVICE", "cpu")

# Results scoring below this cosine score are flagged low_confidence (doc §2.4.2).
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("FDCA_THRESHOLD", "0.3"))

DEFAULT_TOP_K = 20
MAX_TOP_K = 50
DEFAULT_PAGE_LIMIT = 24

# CORS origins allowed to call the API. "*" is fine for a course demo; tighten
# to the frontend origin in production.
CORS_ORIGINS = os.getenv("FDCA_CORS_ORIGINS", "*").split(",")
