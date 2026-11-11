"""Shared text-detection server: one EfficientViT instance, continuous batching.

Run as ``python -m surya.detection.server --port P``. The forward pass is batched
across all clients; the (CPU) heatmap->box post-processing runs in the server's
thread pool. ``include_maps`` (debug heatmaps) is honored and serialized on request.
"""

from __future__ import annotations

import argparse
from typing import Any, List, Optional

from surya.common.batch_service import BatchEngine, run_server
from surya.common.batch_service.serialize import decode_image, encode_ndarray
from surya.detection.config import detection_service_config
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


class DetectionEngine(BatchEngine):
    def __init__(self, checkpoint: Optional[str] = None):
        from surya.detection import DetectionPredictor

        self.predictor = DetectionPredictor.local(checkpoint)
        self.predictor.disable_tqdm = True
        logger.info(f"detection engine ready (device={self.predictor.model.device})")

    def decode_item(self, item: Any, params: dict):
        return decode_image(item)

    def encode_result(self, result) -> dict:
        return {
            "bboxes": [
                {"polygon": b.polygon, "confidence": b.confidence}
                for b in result.bboxes
            ],
            "image_bbox": result.image_bbox,
            "heatmap": encode_ndarray(result.heatmap),
            "affinity_map": encode_ndarray(result.affinity_map),
        }

    def run_batch(self, payloads: List[Any], params: List[dict]) -> List[Any]:
        results: List[Any] = [None] * len(payloads)
        # The forward pass is identical regardless of include_maps (it only changes
        # post-processing), but __call__ applies one flag to the whole call, so
        # bucket by it. include_maps=True is debug-only, so this is one bucket in
        # practice and the whole batch shares a forward.
        buckets: dict[bool, List[int]] = {}
        for i, p in enumerate(params):
            buckets.setdefault(bool(p.get("include_maps")), []).append(i)

        for include_maps, idxs in buckets.items():
            preds = self.predictor(
                [payloads[i] for i in idxs], include_maps=include_maps
            )
            for i, pred in zip(idxs, preds):
                results[i] = pred
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Shared text-detection server")
    ap.add_argument("--host", default=settings.DETECTOR_SERVER_HOST)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    engine = DetectionEngine(checkpoint=args.checkpoint)
    run_server(engine, detection_service_config(), args.host, args.port)


if __name__ == "__main__":
    main()
