"""FastLayoutPredictor — rf-detr page-layout detector, served from one shared instance.

Drop-in alternative to surya.layout.LayoutPredictor: same LayoutResult/LayoutBox output, but
a lightweight rf-detr object detector instead of the VLM. Labels are canonicalized through the
same LAYOUT_PRED_RELABEL map the VLM layout model uses, so downstream consumers (marker) are unchanged.

The model always runs in a single shared server process (see surya.fast_layout.server);
FastLayoutPredictor is a thin client of it. This is the only path: N worker processes
(e.g. marker) would otherwise each load their own model and thread pool and thrash the
CPU/GPU — there's no benefit to more than one layout model on a host. The first client to
run attaches to a running server or spawns one; the rest attach.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.layout.label import LAYOUT_PRED_RELABEL
from surya.layout.schema import LayoutBox, LayoutResult
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _poly(b):
    x0, y0, x1, y1 = b
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _merge_contained_boxes(boxes: List[LayoutBox], threshold: float) -> List[LayoutBox]:
    """Drop same-label boxes that are (near-)contained in a larger same-label box.

    rf-detr sometimes emits a big block plus smaller duplicate/fragment blocks of
    the same type inside it (e.g. a Text column plus Text fragments). For each box,
    if at least `threshold` of its area lies within a larger box of the *same*
    label, the smaller is removed and its extent merged into the larger (the larger
    grows to their union, so any sliver poking out is preserved). Restricted to the
    same label so a distinct element that merely falls within another's bbox (e.g. a
    PageHeader inside a page-spanning Text box) is kept. `threshold` > 1 disables it.
    """
    if threshold > 1 or len(boxes) < 2:
        return boxes
    # Largest first, so smaller boxes get absorbed into the bigger survivor.
    survivors: List[LayoutBox] = []
    for b in sorted(boxes, key=lambda x: x.area, reverse=True):
        absorbed = False
        for s in survivors:
            if s.label != b.label:
                continue
            if b.area > 0 and s.intersection_area(b) / b.area >= threshold:
                sb, bb = s.bbox, b.bbox
                s.polygon = _poly(
                    [
                        min(sb[0], bb[0]),
                        min(sb[1], bb[1]),
                        max(sb[2], bb[2]),
                        max(sb[3], bb[3]),
                    ]
                )
                absorbed = True
                break
        if not absorbed:
            survivors.append(b)
    # Re-number reading order contiguously over the survivors.
    survivors.sort(key=lambda x: x.position)
    for rank, b in enumerate(survivors):
        b.position = rank
    return survivors


def build_layout_result(image: Image.Image, dets, order) -> LayoutResult:
    """Turn one page's raw rf-detr detections into an ordered LayoutResult.

    `dets` is the per-image detection list from ``RfDetrTorch.detect`` (optionally
    carrying an encoder feature map on ``.features``). `order` is an OrderPredictor
    or None. Shared by the in-process predictor and the server so ordering/relabel
    behaviour is identical on both paths.
    """
    # Reading order: the learned AR head (cross-attends to the encoder feature map) when
    # available, else a top-to-bottom / left-to-right raster sort.
    feats = getattr(dets, "features", None)
    if order is not None and feats is not None and dets:
        positions = order.order_page(
            feats,
            [d["bbox"] for d in dets],
            [d["label"] for d in dets],
            image.width,
            image.height,
        )
    else:
        # Raster sort: the normal path when order is off for this call.
        if order is not None and feats is None and dets:
            # Order model loaded but no feature map came back — it should have run
            # but didn't. Surface this; the "model never loaded" case is logged
            # once at first load.
            logger.warning(
                "Reading-order model loaded but detector returned no feature map; "
                "falling back to raster sort for this page."
            )
        raster = sorted(
            range(len(dets)),
            key=lambda i: (dets[i]["bbox"][1], dets[i]["bbox"][0]),
        )
        positions = [0] * len(dets)
        for rank, i in enumerate(raster):
            positions[i] = rank
    boxes = []
    for d, pos in zip(dets, positions):
        raw = d["label"]
        boxes.append(
            LayoutBox(
                polygon=_poly(d["bbox"]),
                label=LAYOUT_PRED_RELABEL.get(raw, raw),
                raw_label=raw,
                position=pos,
                confidence=d["score"],
            )
        )
    boxes = _merge_contained_boxes(boxes, settings.FAST_LAYOUT_CONTAINMENT_THRESHOLD)
    boxes.sort(key=lambda b: b.position)
    return LayoutResult(
        bboxes=boxes,
        image_bbox=[0.0, 0.0, float(image.width), float(image.height)],
    )


class FastLayoutPredictor:
    """Thin client of the shared fast-layout server (surya.fast_layout.server).

    Holds no model — every call hands the batch to the one shared server, which
    owns the single rf-detr instance and does its own continuous batching across
    all clients. ``num_threads`` and ``batch_size`` are governed server-side
    (FAST_LAYOUT_NUM_THREADS / FAST_LAYOUT_SERVER_MAX_BATCH); the constructor
    still accepts ``num_threads`` for signature compatibility but it has no local
    effect here.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        num_threads: Optional[int] = None,
        use_order: Optional[bool] = None,
    ):
        from surya.fast_layout.client import FastLayoutServerClient

        self.use_order = (
            settings.FAST_LAYOUT_USE_ORDER if use_order is None else use_order
        )
        self._disable_tqdm = settings.DISABLE_TQDM
        self._client = FastLayoutServerClient(checkpoint=checkpoint)

    def to(
        self, *args, **kwargs
    ):  # API parity with other predictors (no-op; device is set server-side)
        return

    def __call__(
        self,
        images: List[Image.Image],
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,  # ignored: server controls batching
        use_order: Optional[bool] = None,
    ) -> List[LayoutResult]:
        if not images:
            return []
        threshold = (
            settings.FAST_LAYOUT_CONFIDENCE_THRESHOLD
            if threshold is None
            else threshold
        )
        use_order = self.use_order if use_order is None else use_order
        return self._client(images, threshold=threshold, use_order=use_order)
