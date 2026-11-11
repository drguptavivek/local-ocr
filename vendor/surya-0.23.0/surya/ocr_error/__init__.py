"""OCRErrorPredictor — served from one shared instance.

Public construction is client-backed: the DistilBert model runs in a single shared
server process (surya.ocr_error.server) and this object POSTs texts to it, so N
worker processes don't each load their own copy. Use ``OCRErrorPredictor.local()``
to build a process-local predictor that owns the model (that's what the server
uses internally). The __call__ signature and OCRErrorDetectionResult output are
unchanged, so callers (marker) are unaffected.
"""

import math
from typing import List, Optional

from tqdm import tqdm

from surya.common.predictor import BasePredictor
from surya.ocr_error.loader import OCRErrorModelLoader
from surya.ocr_error.model.config import ID2LABEL
from surya.ocr_error.schema import OCRErrorDetectionResult
from surya.settings import settings


class OCRErrorPredictor(BasePredictor):
    model_loader_cls = OCRErrorModelLoader
    batch_size = settings.OCR_ERROR_BATCH_SIZE
    default_batch_sizes = {"cpu": 8, "mps": 8, "cuda": 64}

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device=None,
        dtype=None,
        attention_implementation=None,
    ):
        # Client-backed by default: no model load here. device/dtype/attn are
        # accepted for signature parity with BasePredictor but live server-side.
        from surya.ocr_error.client import OCRErrorServerClient

        self._client = OCRErrorServerClient(checkpoint=checkpoint)
        self.model = None
        self.processor = None
        self._disable_tqdm = settings.DISABLE_TQDM

    @classmethod
    def local(cls, *args, **kwargs) -> "OCRErrorPredictor":
        """Process-local predictor that owns the model (used by the server)."""
        self = cls.__new__(cls)
        BasePredictor.__init__(self, *args, **kwargs)
        self._client = None
        return self

    def to(self, device_dtype=None):
        if self._client is not None:  # client-backed: device lives server-side
            return
        return super().to(device_dtype)

    def __call__(self, texts: List[str], batch_size: Optional[int] = None):
        if self._client is not None:
            return self._client(texts)
        return self.batch_ocr_error_detection(texts, batch_size)

    def batch_ocr_error_detection(
        self, texts: List[str], batch_size: Optional[int] = None
    ):
        if batch_size is None:
            batch_size = self.get_batch_size()

        num_batches = math.ceil(len(texts) / batch_size)
        texts_processed = self.processor(
            texts, padding="longest", truncation=True, return_tensors="pt"
        )
        predictions = []
        scores = []
        for batch_idx in tqdm(
            range(num_batches),
            desc="Running OCR Error Detection",
            disable=self.disable_tqdm,
        ):
            start_idx, end_idx = batch_idx * batch_size, (batch_idx + 1) * batch_size
            batch_input_ids = texts_processed.input_ids[start_idx:end_idx].to(
                self.model.device
            )
            batch_attention_mask = texts_processed.attention_mask[start_idx:end_idx].to(
                self.model.device
            )

            with settings.INFERENCE_MODE():
                pred = self.model(batch_input_ids, attention_mask=batch_attention_mask)
                probs = pred.logits.softmax(dim=1)
                predictions.extend(probs.argmax(dim=1).cpu().tolist())
                scores.extend(probs[:, 1].cpu().tolist())

        return OCRErrorDetectionResult(
            texts=texts,
            labels=[ID2LABEL[p] for p in predictions],
            scores=scores,
        )
