"""Strip training-only fields from a trained checkpoint for serving.

``train.py`` saves checkpoints as::

    {"epoch": int, "model": <trainable state_dict>, "optimizer": <obj>, "scheduler": <obj>}

The ``optimizer`` and ``scheduler`` entries are *live pickled objects*, which is
why loading the raw ckpt needs ``weights_only=False`` and breaks whenever those
classes are not importable. For inference the serving side only ever reads
``checkpoint["model"]`` (see ``model/base.py::load_from_pretrained``), so this
tool produces a slim, portable ``{"model": <tensors-only>}`` checkpoint:

  * keeps only the ``model`` state dict, dropping ``optimizer`` / ``scheduler`` /
    ``epoch`` and any other top-level bookkeeping,
  * keeps only plain tensors inside ``model`` (defensive — non-tensor entries are
    dropped and reported),
  * optionally also strips a frozen-backbone prefix (``clip_model.``); the trained
    ``model`` dict normally has none, but ``--strip-clip`` is there if a full dump
    was saved.

It loads with a *tolerant* unpickler so it works even when the optimizer/scheduler
classes used at train time are not importable in the current environment.

Usage (run from backend/)::

    python strip_ckpt.py weights/fdca.ckpt -o weights/fdca_aim.serve.pth
    python strip_ckpt.py weights/fdca.ckpt --dry-run
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

_CLIP_PREFIX = "clip_model."


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class _Stub:
    """Placeholder for unpickled classes that no longer exist locally.

    Only the ``model`` state dict (tensors) is needed; optimizer / scheduler /
    fabric wrapper objects can be replaced with inert stubs and discarded.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self._state = state

    def __reduce__(self):  # never re-serialized, but keep it well-defined
        return (_Stub, ())


class _TolerantUnpickler(pickle.Unpickler):
    """Return an inert stub class whenever ``find_class`` would fail."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            return _Stub


def _tolerant_load(path: Path):
    """``torch.load`` with a tolerant unpickler so missing classes don't crash."""

    class _PickleModule:
        Unpickler = _TolerantUnpickler

        @staticmethod
        def load(file, *args, **kwargs):
            return _TolerantUnpickler(file).load()

    return torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=_PickleModule
    )


def extract_model_tensors(obj) -> tuple[dict, list[str]]:
    """Pull the tensor-only ``model`` state dict out of a loaded checkpoint."""
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    elif isinstance(obj, dict) and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    ):
        state = obj
    else:
        keys = list(obj.keys()) if isinstance(obj, dict) else type(obj)
        sys.exit(f"Unexpected checkpoint structure (top-level: {keys})")

    tensors: dict = {}
    dropped_nontensor: list[str] = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.detach().cpu().contiguous()
        else:
            dropped_nontensor.append(k)
    return tensors, dropped_nontensor


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("ckpt", type=Path, help="input checkpoint path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: <stem>.serve.pth next to input)",
    )
    parser.add_argument(
        "--strip-clip",
        action="store_true",
        help=f"also drop frozen backbone weights under {_CLIP_PREFIX!r}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without saving",
    )
    args = parser.parse_args()

    out_path = args.output or args.ckpt.with_name(f"{args.ckpt.stem}.serve.pth")

    obj = _tolerant_load(args.ckpt)
    if isinstance(obj, dict):
        print(f"[top-level keys] {list(obj.keys())}")

    tensors, dropped_nontensor = extract_model_tensors(obj)

    clip_keys = [k for k in tensors if k.startswith(_CLIP_PREFIX)]
    if args.strip_clip and clip_keys:
        for k in clip_keys:
            tensors.pop(k)

    n_params = sum(t.numel() for t in tensors.values())
    n_bytes = sum(t.numel() * t.element_size() for t in tensors.values())

    print(f"[input]   {args.ckpt}")
    print(f"[kept]    {len(tensors)} tensors, {n_params/1e6:.2f}M params, "
          f"{_human_bytes(n_bytes)}")
    if dropped_nontensor:
        print(f"[dropped] {len(dropped_nontensor)} non-tensor model entries: "
              f"{dropped_nontensor[:5]}{' ...' if len(dropped_nontensor) > 5 else ''}")
    if clip_keys:
        verb = "stripped" if args.strip_clip else "present (use --strip-clip to drop)"
        print(f"[clip]    {len(clip_keys)} {_CLIP_PREFIX!r} keys {verb}")

    if not tensors:
        sys.exit("No tensors found under 'model' — nothing to write.")

    if args.dry_run:
        print("[dry-run] not writing.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": tensors}, out_path)
    print(f"[saved]   {out_path}")
    print("Load it with: model.load_from_pretrained(<path>)  (reads checkpoint['model'])")


if __name__ == "__main__":
    main()
