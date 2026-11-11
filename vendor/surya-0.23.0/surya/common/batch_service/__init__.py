"""Generic single-instance batching service.

One model per host, served from a single process that coalesces work arriving
across many client processes into batched forward passes. Clients are thin: they
attach to a running server or spawn one, then POST work over localhost HTTP.

This is the shared machinery behind the fast-layout, text-detection, and
ocr-error servers — each supplies a small `BatchEngine` adapter (load the model,
run a coalesced batch, (de)serialize items/results) and a `ServiceConfig`; the
queue, continuous-batching worker, HTTP endpoints, and attach/spawn lifecycle
live here.
"""

from surya.common.batch_service.client import BatchServiceClient as BatchServiceClient
from surya.common.batch_service.config import ServiceConfig as ServiceConfig
from surya.common.batch_service.config import (
    service_config_from_settings as service_config_from_settings,
)
from surya.common.batch_service.server import BatchEngine as BatchEngine
from surya.common.batch_service.server import run_server as run_server
