"""Compatibility import for the canonical road_self RPNet implementation.

Historically this module contained a second, full-resolution copy of RPNet.
Keeping two implementations allowed training and inference to drift. Existing
imports remain valid, but all callers now receive the single implementation in
``model.model``.
"""

from .model import (  # noqa: F401
    ConvReLU,
    CrossAttentionLayer,
    DecoderBlock,
    Hourglass,
    RPNet,
    TrajProjector,
    Transformer,
    build_model,
    upsample,
)


__all__ = [
    "ConvReLU",
    "CrossAttentionLayer",
    "DecoderBlock",
    "Hourglass",
    "RPNet",
    "TrajProjector",
    "Transformer",
    "build_model",
    "upsample",
]
