"""ServiceConfig for the shared ocr-error batch server."""

from __future__ import annotations

from typing import Optional

from surya.common.batch_service import ServiceConfig, service_config_from_settings
from surya.settings import settings


def ocr_error_service_config(model_name: Optional[str] = None) -> ServiceConfig:
    return service_config_from_settings(
        backend="ocr_error",
        prefix="OCR_ERROR",
        model_name=model_name or settings.OCR_ERROR_MODEL_CHECKPOINT,
        server_module="surya.ocr_error.server",
        default_max_batch=settings.OCR_ERROR_BATCH_SIZE or 64,
    )
