"""ServiceConfig for the fast-layout batch server."""

from __future__ import annotations

from typing import Optional

from surya.common.batch_service import ServiceConfig, service_config_from_settings
from surya.settings import settings


def layout_service_config(model_name: Optional[str] = None) -> ServiceConfig:
    cfg = service_config_from_settings(
        backend="fast_layout",
        prefix="FAST_LAYOUT",
        model_name=model_name or settings.FAST_LAYOUT_MODEL_CHECKPOINT,
        server_module="surya.fast_layout.server",
        default_max_batch=settings.FAST_LAYOUT_BATCH_SIZE or 8,
    )
    return cfg
