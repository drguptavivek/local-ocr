"""Per-service configuration, built from the model's SETTINGS prefix.

Every batch service reads a uniform block of settings named ``<PREFIX>_SERVER_*``
(e.g. ``FAST_LAYOUT_SERVER_URL``, ``DETECTOR_SERVER_PORT``). ``service_config_from_settings``
reads that block so each model doesn't repeat the plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from surya.settings import settings


@dataclass
class ServiceConfig:
    backend: str  # sentinel/lock namespace + log filename, e.g. "fast_layout"
    model_name: str  # reported at /v1/models; checked on attach
    server_module: str  # python -m target for spawning, e.g. "surya.detection.server"
    host: str
    external_url: Optional[str]  # pinned server; skip spawn
    port: Optional[int]  # None = pick a free port
    autostart: bool
    startup_timeout: float
    request_timeout: float
    batch_wait_ms: int  # coalescing window
    max_batch: int  # ceiling on units per coalesced forward


def service_config_from_settings(
    *,
    backend: str,
    prefix: str,
    model_name: str,
    server_module: str,
    default_max_batch: int = 8,
) -> ServiceConfig:
    def s(name: str, default=None):
        return getattr(settings, f"{prefix}_SERVER_{name}", default)

    return ServiceConfig(
        backend=backend,
        model_name=model_name,
        server_module=server_module,
        host=s("HOST", "127.0.0.1"),
        external_url=s("URL"),
        port=s("PORT"),
        autostart=s("AUTOSTART", True),
        startup_timeout=s("STARTUP_TIMEOUT", 300.0),
        request_timeout=s("TIMEOUT", 600.0),
        batch_wait_ms=s("BATCH_WAIT_MS", 5),
        max_batch=s("MAX_BATCH") or default_max_batch,
    )
