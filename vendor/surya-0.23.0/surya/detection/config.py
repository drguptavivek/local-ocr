"""ServiceConfig for the shared text-detection batch server."""

from __future__ import annotations

from typing import Optional

from surya.common.batch_service import ServiceConfig, service_config_from_settings
from surya.settings import settings


def detection_service_config(model_name: Optional[str] = None) -> ServiceConfig:
    return service_config_from_settings(
        backend="detection",
        prefix="DETECTOR",
        model_name=model_name or settings.DETECTOR_MODEL_CHECKPOINT,
        server_module="surya.detection.server",
        default_max_batch=settings.DETECTOR_BATCH_SIZE or 8,
    )
