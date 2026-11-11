"""Shared fast-layout server: one rf-detr instance, continuous batching.

Run as ``python -m surya.fast_layout.server --port P``. Model specifics live in
``LayoutEngine``; the queue/coalescing/HTTP/lifecycle come from
``surya.common.batch_service``. Clients talk to it via FastLayoutServerClient.
"""

from __future__ import annotations

import argparse
import threading
from typing import Any, List, Optional

from surya.common.batch_service import BatchEngine, run_server
from surya.common.batch_service.serialize import decode_image
from surya.common.order.predictor import load_order_predictor
from surya.common.rfdetr_torch import load_detector, resolve_model_dir
from surya.fast_layout import build_layout_result
from surya.fast_layout.config import layout_service_config
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


class LayoutEngine(BatchEngine):
    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint or settings.FAST_LAYOUT_MODEL_CHECKPOINT
        model_dir = resolve_model_dir(self.checkpoint)
        self.model = load_detector(
            model_dir,
            num_threads=settings.FAST_LAYOUT_NUM_THREADS,
            device=settings.FAST_DETECTOR_DEVICE,
        )
        self._order = None
        self._order_attempted = False
        self._order_lock = threading.Lock()
        logger.info(
            f"layout engine ready (model={self.checkpoint}, device={self.model.device})"
        )

    def _load_order(self):
        with self._order_lock:
            if not self._order_attempted:
                self._order_attempted = True
                self._order = load_order_predictor(
                    device=settings.FAST_DETECTOR_DEVICE or "cpu"
                )
                if self._order is None:
                    logger.warning(
                        "Reading-order model not available; raster-sorting pages "
                        "that request order."
                    )
        return self._order

    def decode_item(self, item: Any, params: dict):
        return decode_image(item)

    def encode_result(self, result) -> dict:
        return {
            "bboxes": [
                {
                    "polygon": b.polygon,
                    "label": b.label,
                    "raw_label": b.raw_label,
                    "position": b.position,
                    "confidence": b.confidence,
                    "count": b.count,
                }
                for b in result.bboxes
            ],
            "image_bbox": result.image_bbox,
            "error": result.error,
        }

    def run_batch(self, payloads: List[Any], params: List[dict]) -> List[Any]:
        order = self._load_order() if any(p.get("use_order") for p in params) else None
        results: List[Any] = [None] * len(payloads)
        # detect() takes a single threshold, so bucket by it (usually one bucket).
        buckets: dict[float, List[int]] = {}
        for i, p in enumerate(params):
            threshold = p.get("threshold")
            if threshold is None:
                threshold = settings.FAST_LAYOUT_CONFIDENCE_THRESHOLD
            buckets.setdefault(threshold, []).append(i)

        for threshold, idxs in buckets.items():
            want_feats = order is not None and any(
                params[i].get("use_order") for i in idxs
            )
            dets = self.model.detect(
                [payloads[i] for i in idxs],
                threshold=threshold,
                batch_size=len(idxs),
                return_features=want_feats,
            )
            for i, det in zip(idxs, dets):
                results[i] = build_layout_result(
                    payloads[i], det, order if params[i].get("use_order") else None
                )
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Shared fast-layout (rf-detr) server")
    ap.add_argument("--host", default=settings.FAST_LAYOUT_SERVER_HOST)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    engine = LayoutEngine(checkpoint=args.checkpoint)
    run_server(engine, layout_service_config(), args.host, args.port)


if __name__ == "__main__":
    main()
