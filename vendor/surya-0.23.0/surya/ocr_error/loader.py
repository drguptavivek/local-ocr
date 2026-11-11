from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from surya.common.load import ModelLoader
from surya.common.s3 import S3DownloaderMixin, download_directory
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _resolve_checkpoint(checkpoint: str) -> str:
    """Resolve an ``s3://`` checkpoint to a local dir (downloading if needed);
    pass hub ids / local paths through unchanged."""
    if not checkpoint.startswith(S3DownloaderMixin.s3_prefix):
        return checkpoint
    local_path = S3DownloaderMixin.get_local_path(checkpoint)
    remote = checkpoint.replace(S3DownloaderMixin.s3_prefix, "")
    retries, delay, attempt = 3, 5, 0
    while attempt < retries:
        try:
            download_directory(remote, local_path)
            break
        except Exception as e:  # noqa: BLE001 - retried below
            attempt += 1
            logger.error(
                f"Error downloading ocr-error model from {remote}. "
                f"Attempt {attempt} of {retries}. Error: {e}"
            )
            if attempt < retries:
                import time

                time.sleep(delay)
            else:
                raise
    return local_path


class OCRErrorModelLoader(ModelLoader):
    """Loads the ocr-error DistilBert via stock transformers.

    The checkpoint is a standard ``DistilBertForSequenceClassification`` and loads
    correctly with ``AutoModelForSequenceClassification`` on transformers 5.x. The
    previously-vendored encoder copy (surya.ocr_error.model.encoder) silently
    produces near-constant logits (~0.47 for any input) on transformers 5.x, so it
    must not be used. Stock transformers also supports flash-attention via
    ``attn_implementation``, so nothing is lost.
    """

    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.OCR_ERROR_MODEL_CHECKPOINT

    def model(
        self,
        device=settings.TORCH_DEVICE_MODEL,
        dtype=settings.MODEL_DTYPE,
        attention_implementation: Optional[str] = None,
    ):
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        local_path = _resolve_checkpoint(self.checkpoint)
        kwargs = {"dtype": dtype}
        if attention_implementation is not None:
            kwargs["attn_implementation"] = attention_implementation
        model = (
            AutoModelForSequenceClassification.from_pretrained(local_path, **kwargs)
            .to(device)
            .eval()
        )
        logger.debug(f"Loaded ocr-error model from {local_path} onto device {device}")
        return model

    def processor(self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE):
        return AutoTokenizer.from_pretrained(_resolve_checkpoint(self.checkpoint))
