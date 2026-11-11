"""Client for the shared ocr-error server."""

from __future__ import annotations

from typing import List, Optional

from surya.common.batch_service import BatchServiceClient
from surya.ocr_error.config import ocr_error_service_config
from surya.ocr_error.schema import OCRErrorDetectionResult


class OCRErrorServerClient:
    def __init__(self, checkpoint: Optional[str] = None):
        self._client = BatchServiceClient(
            config=ocr_error_service_config(model_name=checkpoint),
            encode_item=lambda text: text,
            decode_result=lambda r: r,
        )

    def __call__(self, texts: List[str]) -> OCRErrorDetectionResult:
        # One unit per text; the server returns a {label, score} per text and we
        # reassemble the aggregate result the caller expects.
        results = self._client.infer(texts)
        return OCRErrorDetectionResult(
            texts=list(texts),
            labels=[r["label"] for r in results],
            scores=[r["score"] for r in results],
        )
