"""Minimal FDCA model package, vendored from the CoVR training repo.

Only what the serving API needs to run inference:
  base.py        — BaseRetrievalModel (checkpoint loading)
  combiner.py    — the FDCA Combiner (encode_query_features / encode_target_features)
  backbone/      — the CLIP variant the combiner loads (ViT-L/14)

The training/eval code, other model variants, datasets, and configs live in the
separate CoVR repo and are intentionally not vendored here.
"""

from .combiner import Combiner

__all__ = ["Combiner"]
