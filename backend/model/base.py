import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


def _normalize_orig_mod(key: str) -> str:
    """Strip ``_orig_mod.`` segments inserted by ``torch.compile``."""
    key = key.replace("._orig_mod.", ".")
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod.") :]
    return key


class BaseRetrievalModel(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._register_load_state_dict_pre_hook(self._align_orig_mod_keys)

    def _align_orig_mod_keys(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Remap checkpoint keys so ``_orig_mod.`` presence matches the model.

        Handles both directions: compiled checkpoint → uncompiled model,
        and uncompiled checkpoint → compiled model.
        """
        norm_to_model = {_normalize_orig_mod(k): k for k in self.state_dict()}
        for key in list(state_dict.keys()):
            norm = _normalize_orig_mod(key)
            target = norm_to_model.get(norm, norm)
            if target != key:
                state_dict[target] = state_dict.pop(key)

    def load_from_pretrained(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        result = self.load_state_dict(checkpoint["model"], strict=False)
        frozen_params = {
            name for name, p in self.named_parameters() if not p.requires_grad
        }
        unexpected_trainable = [
            k for k in result.missing_keys if k not in frozen_params
        ]
        if unexpected_trainable:
            raise RuntimeError(
                f"Checkpoint is missing trainable keys: {unexpected_trainable}"
            )
        logging.info("load checkpoint from %s", filename)

    @abstractmethod
    def encode_query_features(
        self, batch: Any
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            "Subclasses must implement `encode_query_features` method."
        )

    @abstractmethod
    def encode_target_features(
        self, batch: Any
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            "Subclasses must implement `encode_target_features` method."
        )
