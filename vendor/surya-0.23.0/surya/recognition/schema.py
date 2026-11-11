from typing import List

from pydantic import BaseModel

from surya.common.polygon import PolygonBox


class BlockOCRResult(PolygonBox):
    label: str  # canonicalized layout label (Picture, Text, ...)
    raw_label: str = ""  # original model label
    reading_order: int  # 0-indexed position in layout output
    html: str = ""  # block HTML (BLOCK_PROMPT output, "" if skipped)
    skipped: bool = False  # True if label was in SKIP_OCR_LABELS
    error: bool = False


class PageOCRResult(BaseModel):
    blocks: List[BlockOCRResult]
    image_bbox: List[float]
