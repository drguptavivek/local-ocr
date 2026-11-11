from typing import List, Optional

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class LayoutBox(PolygonBox):
    label: str  # canonicalized via LAYOUT_PRED_RELABEL
    raw_label: str  # original model label, before canonicalization
    position: int  # reading order index
    count: int = 0  # model's token estimate for OCR output (multiple of 50)


class LayoutResult(BaseModel):
    bboxes: List[LayoutBox]
    image_bbox: List[float]
    raw: Optional[str] = None  # raw model output, useful for debugging
    error: bool = False
