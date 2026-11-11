"""Shared ocr-error server: one DistilBert instance, continuous batching.

Run as ``python -m surya.ocr_error.server --port P``.
"""

from __future__ import annotations

import argparse
from typing import Any, List, Optional

from surya.common.batch_service import BatchEngine, run_server
from surya.logging import get_logger
from surya.ocr_error.config import ocr_error_service_config
from surya.settings import settings

logger = get_logger()


class OCRErrorEngine(BatchEngine):
    def __init__(self, checkpoint: Optional[str] = None):
        from surya.ocr_error import OCRErrorPredictor

        self.predictor = OCRErrorPredictor.local(checkpoint)
        self.predictor.disable_tqdm = True
        logger.info(f"ocr-error engine ready (device={self.predictor.model.device})")

    def run_batch(self, payloads: List[Any], params: List[dict]) -> List[Any]:
        # payloads are texts; batch_ocr_error_detection returns aligned label/score
        # lists over the whole coalesced batch.
        result = self.predictor.batch_ocr_error_detection(list(payloads))
        return [
            {"label": label, "score": score}
            for label, score in zip(result.labels, result.scores)
        ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Shared ocr-error server")
    ap.add_argument("--host", default=settings.OCR_ERROR_SERVER_HOST)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    engine = OCRErrorEngine(checkpoint=args.checkpoint)
    run_server(engine, ocr_error_service_config(), args.host, args.port)


if __name__ == "__main__":
    main()
