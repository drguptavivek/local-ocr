"""OrderPredictor — runs the AR reading-order head on rf-detr detections + the encoder feature map.

Given, per page, the rf-detr projector feature map and the detected boxes (pixel xyxy) + labels,
returns a reading-order position for each detection (0 = read first). Used by FastLayoutPredictor
so layout always returns order.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from surya.common.order.order_ar import (
    ReadingOrderAR,
    canonical_order,
    box_features,
    LAYOUT_CLASSES,
    MAX_BOXES,
)


class OrderPredictor:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt_path = os.path.join(model_dir, "order_ar.pt")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.feat_dim = int(ck.get("feat_dim", 256))
        self.feat_hw = int(ck.get("feat_hw", 28))
        self.res = int(ck.get("res", 448))
        self.model = ReadingOrderAR(
            d=int(ck.get("d", 128)),
            layers=int(ck.get("layers", 3)),
            feat_dim=self.feat_dim,
            feat_hw=self.feat_hw,
            dropout=0.0,
        )
        self.model.load_state_dict(ck["model"])
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def order_page(self, feature_map, boxes_xyxy, labels, width, height) -> List[int]:
        """feature_map: [C,F,F] tensor (rf-detr projector output for this page).
        boxes_xyxy: [N,4] pixel coords. labels: list of label strings (canonical class names).
        Returns position[i] for each detection i (0 = read first)."""
        n = len(boxes_xyxy)
        if n == 0:
            return []
        if n == 1:
            return [0]
        if n > MAX_BOXES:  # fall back to raster order beyond the trained vocab width
            return _raster_positions(boxes_xyxy)

        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        b1000 = np.empty_like(boxes)
        b1000[:, [0, 2]] = boxes[:, [0, 2]] / max(1.0, width) * 1000.0
        b1000[:, [1, 3]] = boxes[:, [1, 3]] / max(1.0, height) * 1000.0

        order = canonical_order(b1000)  # raster pos -> original idx
        b_raster = b1000[order]
        lab_raster = [
            LAYOUT_CLASSES.index(labels[p])
            if labels[p] in LAYOUT_CLASSES
            else LAYOUT_CLASSES.index("Text")
            for p in order
        ]

        feats = torch.from_numpy(box_features(b_raster)).unsqueeze(0).to(self.device)
        labs = torch.tensor(lab_raster, dtype=torch.long, device=self.device).unsqueeze(
            0
        )
        mask = torch.ones(1, n, dtype=torch.bool, device=self.device)
        # [C,F,F] -> [1, HW, C]
        fmap = (
            feature_map.reshape(self.feat_dim, -1)
            .transpose(0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        pred = self.model.decode(feats, labs, mask, fmap)[
            0
        ]  # raster positions, in reading order

        # raster pos p -> original idx order[p]; reading sequence of original indices:
        reading = [order[p] for p in pred]
        position = [0] * n
        for rank, orig_idx in enumerate(reading):
            position[orig_idx] = rank
        return position


def _raster_positions(boxes_xyxy) -> List[int]:
    """Plain top-to-bottom, left-to-right fallback."""
    idx = sorted(
        range(len(boxes_xyxy)), key=lambda i: (boxes_xyxy[i][1], boxes_xyxy[i][0])
    )
    position = [0] * len(boxes_xyxy)
    for rank, i in enumerate(idx):
        position[i] = rank
    return position


def load_order_predictor(checkpoint: Optional[str] = None, device: str = "cpu"):
    """Resolve + load the order predictor, or return None if no checkpoint is configured/available."""
    from surya.common.rfdetr_torch import resolve_model_dir
    from surya.settings import settings

    ckpt = checkpoint or getattr(settings, "FAST_ORDER_MODEL_CHECKPOINT", None)
    if not ckpt:
        return None
    try:
        model_dir = resolve_model_dir(ckpt)
        return OrderPredictor(model_dir, device=device)
    except Exception:
        return None
