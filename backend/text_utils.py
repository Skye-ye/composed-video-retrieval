"""Tiny text helper vendored from the training repo (src/tools/data_utils.py).

Only `remove_neg` is needed at serving time; the rest of data_utils pulled in
torchvision image transforms the API never uses, so it isn't vendored.
"""


_NEG_PATTERNS = ("instead of", "rather than", "not")


def remove_neg(caption: str) -> str:
    """Strip negation connectors so the caption describes the positive target.

    FDCA's ``caption_pos`` branch wants the target's semantics without the
    negation scaffolding — stripping these markers produces a cleaner cue.
    """
    for pat in _NEG_PATTERNS:
        caption = caption.replace(pat, "")
    return caption
