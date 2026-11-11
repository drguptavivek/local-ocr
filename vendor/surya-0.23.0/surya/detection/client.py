"""Client for the shared text-detection server."""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.common.batch_service import BatchServiceClient
from surya.common.batch_service.serialize import decode_ndarray, encode_image
from surya.common.polygon import PolygonBox
from surya.detection.config import detection_service_config
from surya.detection.schema import TextDetectionResult


def _decode_map(data):
    # include_maps returns PIL Images (mode "L") from the local predictor; the
    # debug CLI calls .save() on them, so reconstruct a PIL Image, not an ndarray.
    arr = decode_ndarray(data)
    return Image.fromarray(arr) if arr is not None else None


def _decode_result(d: dict) -> TextDetectionResult:
    return TextDetectionResult(
        bboxes=[
            PolygonBox(polygon=b["polygon"], confidence=b.get("confidence"))
            for b in d["bboxes"]
        ],
        heatmap=_decode_map(d.get("heatmap")),
        affinity_map=_decode_map(d.get("affinity_map")),
        image_bbox=d["image_bbox"],
    )


class DetectionServerClient:
    def __init__(self, checkpoint: Optional[str] = None):
        self._client = BatchServiceClient(
            config=detection_service_config(model_name=checkpoint),
            encode_item=encode_image,
            decode_result=_decode_result,
        )

    def __call__(
        self, images: List[Image.Image], include_maps: bool = False
    ) -> List[TextDetectionResult]:
        return self._client.infer(images, params={"include_maps": bool(include_maps)})
