"""Private, localhost-only shared OCR queue service."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import stat
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from backend.ocr_limits import ERROR_DETAILS, InputError

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
MAX_ACTIVE_JOBS = 100
MAX_CLIENT_JOBS = 1000
MAX_STORED_JOBS = 5000
MAX_CLIENT_BYTES = 512 * 1024**2
MAX_STORED_BYTES = 2 * 1024**3
MAX_RESULT_CHARS = 5_000_000
POLL_SECONDS = 0.25
SAFE_CLIENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def _state_path(value: str | None = None) -> Path:
    return Path(
        value or os.getenv("OCR_SERVICE_STATE") or "~/.local/share/aiims-ocr"
    ).expanduser()


def _clients_path(state: Path, value: str | None = None) -> Path:
    return Path(
        value or os.getenv("OCR_SERVICE_CLIENTS") or state / "clients"
    ).expanduser()


def _private_dir(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError("OCR service directories cannot be symlinks")
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError("OCR service directory must be owned by the current user")
    if existed and stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError("OCR service directory must have owner-only permissions")
    if not existed:
        os.chmod(path, 0o700)


def _private_file(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError("OCR service files cannot be symlinks")
    if not path.exists():
        return
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError("OCR service file must be owned by the current user")
    os.chmod(path, 0o600)


def _job_language(stored: str) -> str | list[str] | None:
    return json.loads(stored) if stored.startswith("[") else stored or None


class BodyTooLarge(Exception):
    pass


class RequestLimit:
    """Count ingress bytes before Starlette parses a multipart request."""

    def __init__(self, app: Any, limit: int = MAX_REQUEST_BYTES):
        self.app = app
        self.limit = limit

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        content_length = next(
            (v for k, v in scope.get("headers", []) if k.lower() == b"content-length"),
            None,
        )
        if content_length:
            try:
                if int(content_length) > self.limit:
                    await self._reject(scope, send)
                    return
            except ValueError:
                await self._reject(scope, send)
                return
        total = 0

        async def counted_receive() -> dict:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.limit:
                    raise BodyTooLarge
            return message

        try:
            await self.app(scope, counted_receive, send)
        except BodyTooLarge:
            await self._reject(scope, send)

    @staticmethod
    async def _reject(scope: dict, send: Any) -> None:
        payload = b'{"detail":"request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


class OCRService:
    def __init__(self, state_dir: Path, clients_dir: Path):
        self.state_dir = state_dir
        self.clients_dir = clients_dir
        self.db_path = state_dir / "queue.sqlite3"
        self.temp_dir = state_dir / "tmp"
        self._lock_file: Any = None
        self._stop = asyncio.Event()
        self.worker_task: asyncio.Task | None = None

    def start(self) -> None:
        _private_dir(self.state_dir)
        _private_dir(self.clients_dir)
        _private_dir(self.temp_dir)
        lock_path = self.state_dir / "service.lock"
        _private_file(lock_path)
        lock_file = lock_path.open("a+b")
        _private_file(lock_path)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError("OCR service already owns this state directory") from exc
        self._lock_file = lock_file
        try:
            with self.connection() as db:
                db.execute(
                    "UPDATE jobs SET status='queued',updated=? WHERE status='processing'",
                    (time.time(),),
                )
            for temp_file in self.temp_dir.glob("job-*"):
                if temp_file.is_file() and not temp_file.is_symlink():
                    temp_file.unlink(missing_ok=True)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def _connect(self) -> sqlite3.Connection:
        _private_file(self.db_path)
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA secure_delete=ON")
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA wal_autocheckpoint=1")
        _private_file(self.db_path)
        db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, client_id TEXT NOT NULL, pipeline TEXT NOT NULL,
            language TEXT NOT NULL, content_hash TEXT NOT NULL, extension TEXT NOT NULL,
            payload BLOB, status TEXT NOT NULL, result_json TEXT, error_code TEXT,
            created REAL NOT NULL, updated REAL NOT NULL,
            UNIQUE(client_id,pipeline,content_hash,language))""")
        db.execute(
            "CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status,created)"
        )
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.db_path}{suffix}")
            if sidecar.exists():
                _private_file(sidecar)
        return db

    @contextmanager
    def connection(self):
        db = self._connect()
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def client_id(self, authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                401,
                "Bearer credential required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization[7:].strip()
        if not token or len(token) > 4096:
            raise HTTPException(
                401, "Invalid bearer credential", headers={"WWW-Authenticate": "Bearer"}
            )
        wanted = hashlib.sha256(token.encode()).digest()
        try:
            for path in self.clients_dir.glob("*.token"):
                if (
                    not SAFE_CLIENT.fullmatch(path.stem)
                    or path.is_symlink()
                    or not path.is_file()
                ):
                    continue
                info = path.stat()
                if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                    continue
                saved = path.read_text(encoding="utf-8").strip()
                if secrets.compare_digest(
                    hashlib.sha256(saved.encode()).digest(), wanted
                ):
                    return path.stem
        except OSError:
            pass
        raise HTTPException(
            401, "Invalid bearer credential", headers={"WWW-Authenticate": "Bearer"}
        )

    def catalog(self) -> dict[str, dict]:
        from backend.ocr_pipelines import pipeline_catalog

        return pipeline_catalog()

    @staticmethod
    def ensure_capacity(db, client_id, added_bytes=0, *, new_job=False):
        size = (
            "coalesce(length(payload),0)+coalesce(length(cast(result_json as blob)),0)"
        )
        for where, params, job_limit, byte_limit in (
            ("", (), MAX_STORED_JOBS, MAX_STORED_BYTES),
            ("WHERE client_id=?", (client_id,), MAX_CLIENT_JOBS, MAX_CLIENT_BYTES),
        ):
            count, stored = db.execute(
                f"SELECT count(*),coalesce(sum({size}),0) FROM jobs {where}", params
            ).fetchone()
            if count + int(new_job) > job_limit or stored + added_bytes > byte_limit:
                raise HTTPException(429, "OCR storage quota reached")

    @staticmethod
    def ensure_active_capacity(db):
        active = db.execute(
            "SELECT count(*) FROM jobs WHERE status IN ('queued','processing')"
        ).fetchone()[0]
        if active >= MAX_ACTIVE_JOBS:
            raise HTTPException(429, "OCR queue is full")

    def submit(
        self,
        client_id: str,
        pipeline: str,
        language: str | list[str] | None,
        extension: str,
        content: bytes,
    ) -> dict:
        catalog = self.catalog()
        meta = catalog.get(pipeline)
        if not meta:
            raise HTTPException(400, "Unknown OCR pipeline")
        if not meta.get("available", False):
            if "device" in meta and meta["device"] is None:
                raise HTTPException(
                    400, "OCR device or runtime unavailable for this pipeline"
                )
            raise HTTPException(400, "OCR pipeline unavailable")
        allowed = {str(x).lower() for x in meta.get("extensions", [])}
        if extension.lower() not in allowed:
            raise HTTPException(415, "File extension is not supported by this pipeline")
        hints = (
            language if isinstance(language, list) else [language] if language else []
        )
        allowed_hints = meta.get("languages")
        if isinstance(language, list):
            allowed_hints = meta.get("recognition_languages", allowed_hints)
        if allowed_hints is not None and any(
            hint not in allowed_hints for hint in hints
        ):
            raise HTTPException(422, "Language is unavailable for this pipeline")
        stored_language = (
            json.dumps(language, separators=(",", ":"))
            if isinstance(language, list)
            else language or ""
        )
        digest = hashlib.sha256(content).hexdigest()
        now = time.time()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            found = db.execute(
                "SELECT * FROM jobs WHERE client_id=? AND pipeline=? AND content_hash=? AND language=?",
                (client_id, pipeline, digest, stored_language),
            ).fetchone()
            if found:
                db.commit()
                return self.public_job(found)
            self.ensure_active_capacity(db)
            self.ensure_capacity(db, client_id, len(content), new_job=True)
            job_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    client_id,
                    pipeline,
                    stored_language,
                    digest,
                    extension.lower(),
                    content,
                    "queued",
                    None,
                    None,
                    now,
                    now,
                ),
            )
            db.commit()
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self.public_job(row)
        finally:
            db.close()

    @staticmethod
    def public_job(row: sqlite3.Row) -> dict:
        from backend.ocr_pipelines import execution_timeout_seconds

        language = _job_language(row["language"])
        hints = (
            language if isinstance(language, list) else [language] if language else []
        )
        result = {
            "job_id": row["id"],
            "pipeline": row["pipeline"],
            "language": hints[0] if len(hints) == 1 else None,
            "languages": hints,
            "status": row["status"],
            "execution_timeout_seconds": execution_timeout_seconds(row["pipeline"]),
            "created_at": row["created"],
            "updated_at": row["updated"],
        }
        if row["result_json"]:
            result["result"] = json.loads(row["result_json"])
        if row["error_code"]:
            result["error"] = row["error_code"]
            result["error_detail"] = ERROR_DETAILS.get(
                row["error_code"], ERROR_DETAILS["pipeline_failed"]
            )
        return result

    def workload(self, client_id: str) -> dict:
        with self.connection() as db:
            db.execute("BEGIN")
            active = dict(
                db.execute(
                    """SELECT status, count(*) FROM jobs
                   WHERE status IN ('queued', 'processing') GROUP BY status"""
                ).fetchall()
            )
            rows = db.execute(
                """SELECT pipeline, status, count(*) AS jobs,
                          sum(coalesce(length(payload), 0) +
                              coalesce(length(cast(result_json AS blob)), 0)) AS bytes
                   FROM jobs WHERE client_id=? GROUP BY pipeline, status
                   ORDER BY pipeline, status""",
                (client_id,),
            ).fetchall()
        counts = dict.fromkeys(("queued", "processing", "completed", "failed"), 0)
        pipelines = {}
        for row in rows:
            pipeline_counts = pipelines.setdefault(
                row["pipeline"], counts.fromkeys(counts, 0)
            )
            pipeline_counts[row["status"]] = row["jobs"]
            counts[row["status"]] += row["jobs"]
        return {
            "queue": {
                "queued": active.get("queued", 0),
                "processing": active.get("processing", 0),
                "active_limit": MAX_ACTIVE_JOBS,
                "available_slots": MAX_ACTIVE_JOBS - sum(active.values()),
            },
            "client": {
                "jobs_by_status": counts,
                "jobs_by_pipeline": pipelines,
                "stored_jobs": sum(counts.values()),
                "stored_bytes": sum(row["bytes"] for row in rows),
                "max_stored_jobs": MAX_CLIENT_JOBS,
                "max_stored_bytes": MAX_CLIENT_BYTES,
            },
        }

    def get_job(self, job_id: str, client_id: str) -> dict:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE id=? AND client_id=?", (job_id, client_id)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Job not found")
        return self.public_job(row)

    def delete_job(self, job_id: str, client_id: str) -> None:
        with self.connection() as db:
            cur = db.execute(
                "DELETE FROM jobs WHERE id=? AND client_id=? AND status IN ('completed','failed')",
                (job_id, client_id),
            )
            if not cur.rowcount:
                row = db.execute(
                    "SELECT status FROM jobs WHERE id=? AND client_id=?",
                    (job_id, client_id),
                ).fetchone()
                if row is None:
                    raise HTTPException(404, "Job not found")
                raise HTTPException(409, "Only completed or failed jobs can be deleted")
        self.checkpoint()

    def checkpoint(self) -> None:
        try:
            with self.connection() as db:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            logging.getLogger(__name__).warning("OCR database checkpoint deferred")

    def retry(self, job_id: str, client_id: str) -> dict:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, payload IS NOT NULL AS has_payload FROM jobs WHERE id=? AND client_id=?",
                (job_id, client_id),
            ).fetchone()
            if row is None:
                raise HTTPException(404, "Job not found")
            if row["status"] != "failed" or not row["has_payload"]:
                raise HTTPException(
                    409, "Only failed jobs with retained input can be retried"
                )
            self.ensure_active_capacity(db)
            db.execute(
                "UPDATE jobs SET status='queued',error_code=NULL,updated=? WHERE id=?",
                (time.time(), job_id),
            )
            return self.public_job(
                db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            )

    async def worker(self) -> None:
        while not self._stop.is_set():
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1"
                ).fetchone()
                if row:
                    db.execute(
                        "UPDATE jobs SET status='processing',updated=? WHERE id=?",
                        (time.time(), row["id"]),
                    )
                db.commit()
            if not row:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=POLL_SECONDS)
                except TimeoutError:
                    pass
                continue
            await self._process(row)

    async def _process(self, row: sqlite3.Row) -> None:
        fd, filename = tempfile.mkstemp(
            prefix="job-", suffix=row["extension"], dir=self.temp_dir
        )
        os.chmod(filename, 0o600)
        try:
            with os.fdopen(fd, "wb") as source_file:
                source_file.write(row["payload"] or b"")
            try:
                from backend.ocr_pipelines import run_pipeline

                result = await asyncio.to_thread(
                    run_pipeline,
                    row["pipeline"],
                    Path(filename),
                    _job_language(row["language"]),
                )
                if not isinstance(result, dict) or not isinstance(
                    result.get("text"), str
                ):
                    raise ValueError("invalid pipeline result")
                if len(result.get("text", "")) > MAX_RESULT_CHARS:
                    raise ValueError("pipeline result too large")
                encoded = json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "text",
                            "format",
                            "flags",
                            "model",
                            "version",
                            "metadata",
                        )
                        if k in result
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                if len(encoded) > MAX_RESULT_CHARS * 2:
                    raise ValueError("pipeline result too large")
                with self.connection() as db:
                    db.execute("BEGIN IMMEDIATE")
                    self.ensure_capacity(
                        db,
                        row["client_id"],
                        len(encoded.encode()) - len(row["payload"]),
                    )
                    db.execute(
                        "UPDATE jobs SET status='completed',payload=NULL,result_json=?,error_code=NULL,updated=? WHERE id=?",
                        (encoded, time.time(), row["id"]),
                    )
            except Exception as exc:
                if isinstance(exc, InputError):
                    error = exc.code
                elif isinstance(exc, TimeoutError):
                    error = "pipeline_timeout"
                elif isinstance(exc, HTTPException) and exc.status_code == 429:
                    error = "storage_quota"
                else:
                    error = "pipeline_failed"
                status = "queued" if self._stop.is_set() else "failed"
                with self.connection() as db:
                    db.execute(
                        "UPDATE jobs SET status=?,error_code=?,updated=? WHERE id=?",
                        (
                            status,
                            None if status == "queued" else error,
                            time.time(),
                            row["id"],
                        ),
                    )
            else:
                self.checkpoint()
        finally:
            Path(filename).unlink(missing_ok=True)


def create_app(
    state_dir: str | Path | None = None, clients_dir: str | Path | None = None
) -> FastAPI:
    state = _state_path(str(state_dir) if state_dir else None)
    clients = _clients_path(state, str(clients_dir) if clients_dir else None)
    service = OCRService(state, clients)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        from backend.ocr_pipelines import cancel_running, reset_cancellation

        from backend.ocr_device import device_mode, gpu_layers

        device_mode()
        gpu_layers()
        service.start()
        reset_cancellation()
        task = asyncio.create_task(service.worker())
        service.worker_task = task
        try:
            yield
        finally:
            service._stop.set()
            try:
                cancel_running()
                await task
            finally:
                service.close()

    api = FastAPI(
        title="Local OCR Service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def authenticated(authorization: str | None = Header(default=None)) -> str:
        return service.client_id(authorization)

    @api.get("/health")
    def health() -> dict:
        if service.worker_task is None or service.worker_task.done():
            raise HTTPException(503, "OCR worker unavailable")
        return {"status": "ok"}

    @api.get("/v1/pipelines")
    def pipelines(_: str = Depends(authenticated)) -> dict:
        return {"pipelines": service.catalog()}

    @api.get("/v1/workload")
    def workload(client: str = Depends(authenticated)) -> dict:
        return service.workload(client)

    @api.post("/v1/jobs")
    async def submit_job(
        file: UploadFile = File(...),
        pipeline: str = Form(default="docling"),
        language: str | None = Form(default=None),
        languages: list[str] | None = Form(default=None),
        client: str = Depends(authenticated),
    ) -> dict:
        hints: str | list[str] | None = (
            language.strip().lower() or None if language is not None else None
        )
        if languages is not None:
            if language is not None:
                raise HTTPException(422, "Send either language or languages")
            if not 1 <= len(languages) <= 8:
                raise HTTPException(422, "Send between 1 and 8 language hints")
            hints = list(dict.fromkeys(value.strip().lower() for value in languages))
        values = hints if isinstance(hints, list) else [hints] if hints else []
        if any(not re.fullmatch(r"[a-z][a-z-]{0,31}", value) for value in values):
            raise HTTPException(422, "Use a language code or name")
        filename = Path(file.filename or "upload").name
        extension = Path(filename).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", extension):
            extension = ".bin"
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if not content:
            raise HTTPException(400, "Upload is empty")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload exceeds size limit")
        return service.submit(client, pipeline, hints, extension, content)

    @api.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, client: str = Depends(authenticated)) -> dict:
        return service.get_job(job_id, client)

    @api.post("/v1/jobs/{job_id}/retry")
    def retry_job(job_id: str, client: str = Depends(authenticated)) -> dict:
        return service.retry(job_id, client)

    @api.delete("/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, client: str = Depends(authenticated)) -> None:
        service.delete_job(job_id, client)

    api.state.ocr_service = service
    return RequestLimit(api)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private local OCR API")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--clients-dir", default=None)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    app = create_app(args.state_dir, args.clients_dir)
    uvicorn.run(
        app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning"
    )


if __name__ == "__main__":
    main()
