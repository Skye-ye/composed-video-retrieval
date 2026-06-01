"""Model inference + retrieval runtime for the fdca_aim serving path.

This is the adapter the interface doc (`前后端接口说明.md` §2.3) calls the "model
inference layer". It wraps the trained FDCA ``Combiner`` and the precomputed
in-memory gallery into the two stateless functions the HTTP layer needs (search
is a numpy matmul, not faiss — see __init__ for why):

    encode_query(ref_embedding, modification_text, retain_text, exclude_text, return_debug)
        -> (query_vec: np.ndarray[D], debug: dict | None)
    search(query_vec, top_k) -> list[(video_id, score)]

How it maps onto the real model API (`Combiner.encode_query_features`, which
returns a 5-tuple and is *not* the clean single-vector function the doc imagines):

  * batch = {ref_vdo_fea: (1, D), caption: [modification_text],
             caption_pos: [retain_text or remove_neg(modification_text)]}
  * query vector = L2-normalize(out[0])  (out[0] == fusion_fea_high, the same
    element the evaluator uses for retrieval, see src/test/utils.py:27).

Retrieval is inner product over L2-normalized vectors (== cosine), matching the
eval protocol in src/test/tfcovr.py::eval_map.

ASSUMPTIONS TO CONFIRM WITH THE MODEL TEAM (documented in README.md):
  * retain_text -> caption_pos. The model has no clean hook for exclude_text;
    it is currently accepted but unused. Confirm the intended mapping.
  * debug branch scores below are a best-effort honest mapping onto three real
    query-side vectors the model produces; there is no action classifier head,
    so predicted_action_class is always None until the model team adds one.
"""

from __future__ import annotations

import os

# torch and faiss each bundle their own OpenMP runtime; on macOS that aborts with
# "libomp already initialized" unless we allow the duplicate. Must be set before
# torch/faiss import. (Course-demo workaround; harmless elsewhere.)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Make backend/ importable regardless of CWD, so the vendored `model` package
# and `text_utils` resolve.
_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from model import Combiner  # noqa: E402
from text_utils import remove_neg  # noqa: E402


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class FdcaAimRuntime:
    def __init__(
        self,
        artifacts_dir: str | Path,
        ckpt_path: str | Path,
        clip_model_name: str = "ViT-L/14",  # 768-dim, matches the checkpoint
        device: str = "cpu",
    ):
        # Retrieval is a plain numpy matmul over the in-memory gallery (see
        # search() below), NOT faiss. Importing faiss alongside torch loads a
        # second OpenMP runtime and segfaults the server on macOS; a flat
        # 212k×768 inner-product search is only a few ms in numpy regardless.
        torch.set_num_threads(1)

        self.device = torch.device(device)
        art = Path(artifacts_dir)

        id_map = json.loads((art / "id_map.json").read_text(encoding="utf-8"))
        self.row_to_id: list[str] = id_map["row_to_id"]
        self.id_to_row: dict[str, int] = id_map["id_to_row"]
        self.dim: int = id_map["dim"]

        self.metadata: dict[str, dict] = json.loads(
            (art / "metadata.json").read_text(encoding="utf-8")
        )

        self.gallery_raw: np.ndarray = np.load(art / "gallery_raw.npy").astype(
            np.float32
        )
        self.gallery_norm: np.ndarray = _l2(self.gallery_raw)

        self.model = Combiner(clip_model_name=clip_model_name)
        self.model.load_from_pretrained(str(ckpt_path))
        self.model.eval().to(self.device)

    # --- lookups the HTTP layer uses ---------------------------------------

    def has_video(self, video_id: str) -> bool:
        return video_id in self.id_to_row

    def ref_vector(self, video_id: str) -> np.ndarray:
        """Raw (un-normalized) gallery vector to feed the model as the reference.

        Faithful to encode_target_features (no normalization); deliberately NOT
        faiss.reconstruct, which returns the normalized search vector.
        """
        return self.gallery_raw[self.id_to_row[video_id]]

    def norm_vector(self, video_id: str) -> np.ndarray:
        return self.gallery_norm[self.id_to_row[video_id]]

    # --- the two model-team functions --------------------------------------

    @torch.no_grad()
    def encode_query(
        self,
        ref_embedding: np.ndarray,
        modification_text: str,
        retain_text: str | None = None,
        exclude_text: str | None = None,  # accepted, currently unused — see header
        return_debug: bool = False,
    ) -> tuple[np.ndarray, dict | None]:
        ref = torch.as_tensor(np.asarray(ref_embedding, dtype=np.float32))
        ref = ref.reshape(1, -1).to(self.device)  # (1, D)
        caption_pos = retain_text if retain_text else remove_neg(modification_text)

        batch = {
            "ref_vdo_fea": ref,
            "caption": [modification_text],
            "caption_pos": [caption_pos],
        }
        out = self.model.encode_query_features(batch)
        # out = (fusion_fea_high, fusion_fea_high_token, remained_text_features,
        #        ref_high_feature_mean, fusion_fea_high_token_negation)
        query = out[0]  # the retrieval vector the evaluator uses
        query_vec = _l2(query.squeeze(0).float().cpu().numpy())

        debug = None
        if return_debug:
            # Honest best-effort: three real query-side vectors, normalized, so
            # the server can score each result against them as "branch scores".
            inject = _l2(out[0].squeeze(0).float().cpu().numpy())  # combined query
            retain = _l2(out[3].squeeze(0).float().cpu().numpy())  # ref-mean cue
            exclude = _l2(out[4].squeeze(0).float().cpu().numpy())  # negation branch
            debug = {
                "branch_vectors": {
                    "retain": retain,
                    "inject": inject,
                    "exclude": exclude,
                },
                "caption": modification_text,
                "caption_pos": caption_pos,
                "predicted_action_class": None,  # no classifier head in this model
            }
        return query_vec, debug

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        q = _l2(np.asarray(query_vec, dtype=np.float32))
        scores = self.gallery_norm @ q  # (N,); cosine, both sides L2-normalized
        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]  # sort the k hits descending
        return [(self.row_to_id[int(i)], float(scores[i])) for i in top]
