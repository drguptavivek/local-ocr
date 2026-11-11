"""Shared (de)serialization helpers for batch-service wire payloads."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def encode_image(image: Image.Image) -> str:
    """PIL image -> base64 PNG string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_image(data: str) -> Image.Image:
    """base64 PNG string -> RGB PIL image."""
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def encode_ndarray(arr) -> str | None:
    """numpy array -> base64 .npy string (None passes through). Used for optional
    debug maps that are only serialized when a caller asks for them."""
    if arr is None:
        return None
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_ndarray(data: str | None):
    if data is None:
        return None
    return np.load(io.BytesIO(base64.b64decode(data)), allow_pickle=False)
