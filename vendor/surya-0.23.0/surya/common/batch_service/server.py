"""Server side of the batch service: model owner + continuous-batching worker + HTTP.

A `BatchEngine` subclass supplies the model specifics. `run_server` wires it to a
request queue, a single worker thread that coalesces queued units across concurrent
client requests into one `run_batch` call, and an HTTP surface:

  GET  /health     -> 200 {"status": "ok"}
  GET  /v1/models  -> {"data": [{"id": <model_name>}]}   (spawn model check)
  POST /v1/infer   -> {"items": [<json unit>, ...], "params": {<shared params>}}
                   -> {"results": [<json result>, ...]}   (aligned to items)

Items are decoded in the request thread (parallel), then batched by the worker.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List

from surya.common.batch_service.config import ServiceConfig
from surya.logging import configure_logging, get_logger

logger = get_logger()


class BatchEngine:
    """Model adapter. Loaded once in the server process.

    Subclasses implement the model-specific parts; everything else (queueing,
    coalescing, HTTP) is generic.
    """

    def decode_item(self, item: Any, params: dict) -> Any:
        """Turn one JSON request unit into a model input (e.g. base64 -> PIL)."""
        return item

    def encode_result(self, result: Any) -> Any:
        """Turn one model output into a JSON-serializable value."""
        return result

    def run_batch(self, payloads: List[Any], params: List[dict]) -> List[Any]:
        """Run the model on a coalesced batch. `payloads[i]` carries the params in
        `params[i]` (units from different client requests may have different params,
        so bucket internally when a param affects the forward). Return one result per
        payload, aligned by index."""
        raise NotImplementedError


@dataclass
class _Job:
    payload: Any
    params: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: str | None = None


class _Batcher:
    """Owns the queue and the single worker thread."""

    def __init__(self, engine: BatchEngine, config: ServiceConfig):
        self.engine = engine
        self._queue: "queue.Queue[_Job]" = queue.Queue()
        self._max_batch = config.max_batch
        self._wait_s = max(0, config.batch_wait_ms) / 1000.0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, payload: Any, params: dict) -> _Job:
        job = _Job(payload=payload, params=params)
        self._queue.put(job)
        return job

    def _drain(self) -> List[_Job]:
        """Block for one job, then coalesce more arriving within the wait window,
        up to the batch ceiling. This is the continuous-batching step: units from
        independent client requests merge into one forward."""
        batch = [self._queue.get()]
        if self._wait_s > 0:
            deadline = time.monotonic() + self._wait_s
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
        else:
            while len(batch) < self._max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
        return batch

    def _run(self) -> None:
        while True:
            batch = self._drain()
            try:
                results = self.engine.run_batch(
                    [j.payload for j in batch], [j.params for j in batch]
                )
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"engine returned {len(results)} results for {len(batch)} inputs"
                    )
                for job, result in zip(batch, results):
                    job.result = result
                    job.done.set()
            except Exception as e:  # never let the worker thread die
                logger.exception("batch failed")
                for job in batch:
                    if not job.done.is_set():
                        job.error = str(e)
                        job.done.set()


def _make_handler(engine: BatchEngine, batcher: _Batcher, config: ServiceConfig):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence per-request stderr spam
            return

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/health":
                return self._send_json(200, {"status": "ok"})
            if path.endswith("/models"):
                return self._send_json(
                    200, {"data": [{"id": config.model_name, "object": "model"}]}
                )
            return self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/infer"):
                return self._send_json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length))
                params = req.get("params") or {}
                jobs = [
                    batcher.submit(engine.decode_item(item, params), params)
                    for item in req.get("items", [])
                ]
            except Exception as e:
                return self._send_json(400, {"error": f"bad request: {e}"})

            results = []
            for job in jobs:
                if not job.done.wait(timeout=config.request_timeout):
                    return self._send_json(504, {"error": "inference timed out"})
                if job.error is not None:
                    return self._send_json(500, {"error": job.error})
                results.append(engine.encode_result(job.result))
            self._send_json(200, {"results": results})

    return Handler


def run_server(
    engine: BatchEngine, config: ServiceConfig, host: str, port: int
) -> None:
    configure_logging()
    batcher = _Batcher(engine, config)
    httpd = ThreadingHTTPServer((host, port), _make_handler(engine, batcher, config))
    # The server is persistent: it is not tied to the process that spawned it, so
    # many client processes can share it and the spawner exiting does not tear it
    # down. Clients re-spawn it on demand if it ever goes away (crash/reboot/kill).
    logger.info(
        f"{config.backend} server listening on http://{host}:{port} "
        f"(model={config.model_name}, max_batch={config.max_batch})"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
