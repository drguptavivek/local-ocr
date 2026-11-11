"""Client for the shared fast-layout server (see surya.fast_layout.server)."""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.common.batch_service import BatchServiceClient
from surya.common.batch_service.serialize import encode_image
from surya.fast_layout.config import layout_service_config
from surya.layout.schema import LayoutBox, LayoutResult


def _decode_result(d: dict) -> LayoutResult:
    boxes = [
        LayoutBox(
            polygon=b["polygon"],
            label=b["label"],
            raw_label=b["raw_label"],
            position=b["position"],
            confidence=b.get("confidence"),
            count=b.get("count", 0),
        )
        for b in d["bboxes"]
    ]
    return LayoutResult(
        bboxes=boxes, image_bbox=d["image_bbox"], error=d.get("error", False)
    )


class FastLayoutServerClient:
    def __init__(self, checkpoint: Optional[str] = None):
        self._client = BatchServiceClient(
            config=layout_service_config(model_name=checkpoint),
            encode_item=encode_image,
            decode_result=_decode_result,
        )

    def __call__(
        self,
        images: List[Image.Image],
        threshold: Optional[float] = None,
        use_order: Optional[bool] = None,
    ) -> List[LayoutResult]:
        return self._client.infer(
            images, params={"threshold": threshold, "use_order": bool(use_order)}
        )
