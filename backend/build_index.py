"""Build the serving artifacts from precomputed AIM gallery embeddings.

Inputs you already have:
  * a directory of per-video AIM embeddings, one ``<video_id>.npy`` each
    (shape ``(D,)`` pooled, or ``(T, D)`` per-frame — both accepted),
  * (optional) the TF-CoVR video frame folders, to cut thumbnails from.

Outputs written to ``--out-dir`` (everything the backend loads at startup):
  * ``index.faiss``      — ``IndexFlatIP`` over L2-normalized gallery vectors
                           (inner product == cosine), matches the eval protocol.
  * ``gallery_raw.npy``  — ``(N, D)`` float32 *un-normalized* mean-pooled vectors,
                           row-aligned to the faiss index. The reference vector
                           fed into the model is read from here (faithful to how
                           ``encode_target_features`` produces gallery features),
                           not from ``faiss.reconstruct`` (which would be normalized).
  * ``id_map.json``      — ``{"row_to_id": [...], "id_to_row": {...}, "dim", "count"}``.
  * ``metadata.json``    — ``{video_id: {thumbnail_url, duration_sec, title}}``.
  * ``thumbs/<id>.jpg``  — one mid-frame per video (only if ``--video-root`` given).

The gallery is simply every embedding file found under ``--emb-dir``.

Example (run from backend/)::

    python build_index.py --emb-dir data/video_embedding --out-dir artifacts
    # thumbnails (if you have the frame folders):  add  --video-root /path/to/video
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _pool(arr: np.ndarray) -> np.ndarray:
    """Reduce a per-video embedding to a single ``(D,)`` vector."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:  # (T, D) -> mean over time, matching encode_target_features
        arr = arr.mean(axis=0)
    elif arr.ndim != 1:
        raise ValueError(f"unexpected embedding shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _mid_frame(folder: Path) -> Path | None:
    frames = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    )
    return frames[len(frames) // 2] if frames else None


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--emb-dir", type=Path, required=True,
        help="directory of per-video <video_id>.npy AIM embeddings",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="where to write serving artifacts",
    )
    parser.add_argument(
        "--video-root", type=Path, default=None,
        help="optional TF-CoVR video/<id>/ frame folders, used to cut thumbnails",
    )
    parser.add_argument(
        "--thumb-prefix", default="/static/thumbs",
        help="URL prefix the backend serves thumbnails under (default /static/thumbs)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="cap the gallery to the first N embeddings (0 = all); handy for a "
             "fast demo gallery instead of indexing the full set",
    )
    args = parser.parse_args()

    try:
        import faiss
    except ImportError:
        sys.exit("faiss not installed. Try: pip install faiss-cpu")

    emb_files = sorted(args.emb_dir.glob("*.npy"))
    if not emb_files:
        sys.exit(f"No .npy embeddings found under {args.emb_dir}")
    if args.limit and args.limit > 0:
        emb_files = emb_files[: args.limit]
    total = len(emb_files)

    out_dir = args.out_dir
    thumbs_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    row_to_id: list[str] = []
    vectors: list[np.ndarray] = []
    metadata: dict[str, dict] = {}
    dim: int | None = None
    n_thumbs = 0

    if args.video_root:
        thumbs_dir.mkdir(parents=True, exist_ok=True)

    for n, path in enumerate(emb_files, 1):
        if n % 20000 == 0 or n == total:
            print(f"  ...{n}/{total} embeddings", flush=True)
        video_id = path.stem
        vec = _pool(np.load(path))
        if dim is None:
            dim = int(vec.shape[0])
        elif vec.shape[0] != dim:
            sys.exit(
                f"dim mismatch: {video_id} has {vec.shape[0]}, expected {dim}"
            )

        thumbnail_url = None
        if args.video_root:
            folder = args.video_root / video_id
            frame = _mid_frame(folder) if folder.is_dir() else None
            if frame is not None:
                dest = thumbs_dir / f"{video_id}.jpg"
                shutil.copyfile(frame, dest)
                thumbnail_url = f"{args.thumb_prefix.rstrip('/')}/{video_id}.jpg"
                n_thumbs += 1

        row_to_id.append(video_id)
        vectors.append(vec)
        metadata[video_id] = {
            "thumbnail_url": thumbnail_url,
            "duration_sec": None,  # unknown from embeddings; enrich later if needed
            "title": None,         # frontend falls back to video_id
        }

    raw = np.stack(vectors).astype(np.float32)  # (N, D), un-normalized
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / np.clip(norms, 1e-12, None)

    index = faiss.IndexFlatIP(dim)
    index.add(normalized)
    faiss.write_index(index, str(out_dir / "index.faiss"))

    np.save(out_dir / "gallery_raw.npy", raw)

    id_map = {
        "row_to_id": row_to_id,
        "id_to_row": {vid: i for i, vid in enumerate(row_to_id)},
        "dim": dim,
        "count": len(row_to_id),
    }
    (out_dir / "id_map.json").write_text(json.dumps(id_map), encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[gallery] {len(row_to_id)} videos, dim={dim}")
    print(f"[thumbs]  {n_thumbs} written" if args.video_root else "[thumbs]  skipped")
    print(f"[out]     {out_dir.resolve()}")
    print("          index.faiss, gallery_raw.npy, id_map.json, metadata.json")


if __name__ == "__main__":
    main()
