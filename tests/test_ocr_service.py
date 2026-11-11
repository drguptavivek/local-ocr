from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.ocr_limits import InputError
from backend.ocr_service import MAX_REQUEST_BYTES, OCRService, _private_dir, create_app


class OCRServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ocr-service-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        clients = self.root / "clients"
        clients.mkdir(mode=0o700)
        self.token = "test-secret-token"
        token_file = clients / "test-app.token"
        token_file.write_text(self.token)
        token_file.chmod(0o600)
        other_token = clients / "other-app.token"
        other_token.write_text("different-secret")
        other_token.chmod(0o600)
        self.adapter = types.ModuleType("backend.ocr_pipelines")
        self.adapter.pipeline_catalog = lambda: {
            "stub": {"extensions": [".png"], "available": True}
        }
        self.adapter.execution_timeout_seconds = lambda pipeline: 180
        self.adapter.cancel_running = lambda: None
        self.adapter.reset_cancellation = lambda: None
        self.adapter.run_pipeline = lambda pipeline, source, language: {
            "text": f"recognized:{source.read_bytes().decode()}:{language}",
            "format": "text",
            "flags": [],
            "model": "stub-model",
            "version": "1.0",
            "metadata": {"source": "stub"},
        }
        self.module_patch = patch.dict(
            sys.modules, {"backend.ocr_pipelines": self.adapter}
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.client = TestClient(create_app(self.root / "state", clients))
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.other_headers = {"Authorization": "Bearer different-secret"}

    def wait_for(self, job_id: str, status: str) -> dict:
        for _ in range(100):
            result = self.client.get(f"/v1/jobs/{job_id}", headers=self.headers).json()
            if result.get("status") == status:
                return result
            time.sleep(0.02)
        self.fail(f"job did not reach {status}")

    def test_workload_is_authenticated_and_scoped_to_client(self):
        with self.client:
            self.assertEqual(self.client.get("/v1/workload").status_code, 401)
            job = self.client.post(
                "/v1/jobs",
                headers=self.headers,
                data={"pipeline": "stub"},
                files={"file": ("sample.png", b"sample", "image/png")},
            ).json()
            self.wait_for(job["job_id"], "completed")

            workload = self.client.get("/v1/workload", headers=self.headers).json()
            self.assertEqual(workload["client"]["stored_jobs"], 1)
            self.assertEqual(workload["client"]["jobs_by_status"]["completed"], 1)
            self.assertEqual(
                workload["client"]["jobs_by_pipeline"]["stub"]["completed"], 1
            )
            self.assertGreater(workload["client"]["stored_bytes"], 0)
            other = self.client.get("/v1/workload", headers=self.other_headers).json()
            self.assertEqual(other["queue"], workload["queue"])
            self.assertEqual(other["client"]["stored_jobs"], 0)
            self.assertEqual(other["client"]["stored_bytes"], 0)
            self.assertEqual(other["client"]["jobs_by_pipeline"], {})

            self.client.delete(f"/v1/jobs/{job['job_id']}", headers=self.headers)
            after = self.client.get("/v1/workload", headers=self.headers).json()
            self.assertEqual(after["client"]["stored_jobs"], 0)

    def test_auth_queue_result_dedupe_delete(self):
        with self.client:
            self.assertEqual(
                self.client.get("/v1/pipelines", headers=self.headers).status_code, 200
            )
            first = self.client.post(
                "/v1/jobs",
                headers=self.headers,
                data={"pipeline": "stub", "language": "hi"},
                files={"file": ("sample.png", b"sample", "image/png")},
            )
            self.assertEqual(first.status_code, 200)
            job = first.json()
            self.assertEqual(job["execution_timeout_seconds"], 180)
            duplicate = self.client.post(
                "/v1/jobs",
                headers=self.headers,
                data={"pipeline": "stub", "language": "hi"},
                files={"file": ("other.png", b"sample", "image/png")},
            )
            self.assertEqual(duplicate.json()["job_id"], job["job_id"])
            self.assertEqual(
                self.client.get(
                    f"/v1/jobs/{job['job_id']}", headers=self.other_headers
                ).status_code,
                404,
            )
            result = self.wait_for(job["job_id"], "completed")
            self.assertEqual(result["result"]["text"], "recognized:sample:hi")
            self.assertEqual(result["result"]["version"], "1.0")
            self.assertEqual(result["result"]["metadata"], {"source": "stub"})
            self.assertEqual(
                self.client.delete(
                    f"/v1/jobs/{job['job_id']}", headers=self.headers
                ).status_code,
                204,
            )
            self.assertEqual(
                self.client.get(
                    f"/v1/jobs/{job['job_id']}", headers=self.headers
                ).status_code,
                404,
            )

    def test_multilingual_upload_and_legacy_single_hint(self):
        self.adapter.pipeline_catalog = lambda: {
            "stub": {
                "extensions": [".png"],
                "available": True,
                "languages": ["en", "hi"],
            }
        }
        with self.client:

            def submit(**fields):
                return self.client.post(
                    "/v1/jobs",
                    headers=self.headers,
                    data={"pipeline": "stub", **fields},
                    files={"file": ("sample.png", b"bilingual", "image/png")},
                )

            multi = submit(languages=["EN", "hi", "en"]).raise_for_status().json()
            self.assertEqual(multi["languages"], ["en", "hi"])
            self.assertIsNone(multi["language"])
            duplicate = submit(languages=["en", "hi"]).raise_for_status().json()
            self.assertEqual(duplicate["job_id"], multi["job_id"])
            result = self.wait_for(multi["job_id"], "completed")
            self.assertEqual(
                result["result"]["text"], "recognized:bilingual:['en', 'hi']"
            )
            single = submit(language="en").raise_for_status().json()
            self.assertEqual(single["language"], "en")
            self.assertEqual(single["languages"], ["en"])
            self.assertEqual(submit(language="en", languages=["hi"]).status_code, 422)
            self.assertEqual(submit(languages=["en", "ja"]).status_code, 422)
            self.assertEqual(submit(languages=["en"] * 9).status_code, 422)

    def test_unauthorized_and_request_body_limit(self):
        with self.client:
            self.assertEqual(self.client.get("/v1/pipelines").status_code, 401)
            response = self.client.post(
                "/v1/jobs",
                content=b"x" * (MAX_REQUEST_BYTES + 1),
                headers={"Content-Type": "application/octet-stream"},
            )
            self.assertEqual(response.status_code, 413)

    def test_failed_job_can_be_retried_and_deleted(self):
        calls = 0

        def flaky(pipeline, source, language):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("private engine failure")
            return {
                "text": "retry succeeded",
                "format": "text",
                "flags": [],
                "model": "stub",
            }

        self.adapter.run_pipeline = flaky
        with self.client:
            submitted = self.client.post(
                "/v1/jobs",
                headers=self.headers,
                data={"pipeline": "stub"},
                files={"file": ("sample.png", b"retry", "image/png")},
            ).json()
            failed = self.wait_for(submitted["job_id"], "failed")
            self.assertEqual(failed["error"], "pipeline_failed")
            self.assertEqual(
                self.client.post(
                    f"/v1/jobs/{submitted['job_id']}/retry", headers=self.headers
                ).status_code,
                200,
            )
            result = self.wait_for(submitted["job_id"], "completed")
            self.assertEqual(result["result"]["text"], "retry succeeded")
            self.assertEqual(
                self.client.delete(
                    f"/v1/jobs/{submitted['job_id']}", headers=self.headers
                ).status_code,
                204,
            )

    def test_input_failure_reports_safe_specific_detail(self):
        def reject(pipeline, source, language):
            raise InputError("pdf_page_limit")

        self.adapter.run_pipeline = reject
        with self.client:
            job = (
                self.client.post(
                    "/v1/jobs",
                    headers=self.headers,
                    data={"pipeline": "stub"},
                    files={"file": ("sample.png", b"too many pages", "image/png")},
                )
                .raise_for_status()
                .json()
            )
            failed = self.wait_for(job["job_id"], "failed")
            self.assertEqual(failed["error"], "pdf_page_limit")
            self.assertIn("1 to 20 pages", failed["error_detail"])

    def test_pipeline_timeout_can_be_retried_with_the_same_job_id(self):
        original = self.adapter.run_pipeline

        def timeout_once(pipeline, source, language):
            self.adapter.run_pipeline = original
            raise TimeoutError("private engine details")

        self.adapter.run_pipeline = timeout_once
        with self.client:
            job = (
                self.client.post(
                    "/v1/jobs",
                    headers=self.headers,
                    data={"pipeline": "stub"},
                    files={"file": ("sample.png", b"same input", "image/png")},
                )
                .raise_for_status()
                .json()
            )
            job_id = job["job_id"]
            failed = self.wait_for(job_id, "failed")
            self.assertEqual(failed["error"], "pipeline_timeout")
            self.assertNotIn("result", failed)
            retried = (
                self.client.post(f"/v1/jobs/{job_id}/retry", headers=self.headers)
                .raise_for_status()
                .json()
            )
            self.assertEqual(retried["job_id"], job_id)
            self.assertEqual(retried["status"], "queued")
            self.assertEqual(retried["execution_timeout_seconds"], 180)
            self.assertNotIn("error", retried)
            completed = self.wait_for(job_id, "completed")
            self.assertEqual(completed["result"]["text"], "recognized:same input:None")

    def test_startup_recovers_interrupted_job(self):
        state_dir = self.root / "state"
        clients_dir = self.root / "clients"
        service = OCRService(state_dir, clients_dir)
        service.start()
        try:
            queued = service.submit("test-app", "stub", "hi", ".png", b"recovered")
            with service.connection() as db:
                db.execute(
                    "UPDATE jobs SET status='processing' WHERE id=?",
                    (queued["job_id"],),
                )
        finally:
            service.close()

        recovering_client = TestClient(create_app(state_dir, clients_dir))
        with recovering_client:
            result = self.wait_for(queued["job_id"], "completed")
            self.assertEqual(result["result"]["text"], "recognized:recovered:hi")

    def test_rejects_symlink_and_insecure_directory_without_chmod(self):
        target = self.root / "unrelated"
        target.mkdir(mode=0o755)
        target.chmod(0o755)
        alias = self.root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            _private_dir(alias)
        with self.assertRaises(RuntimeError):
            _private_dir(target)
        self.assertEqual(target.stat().st_mode & 0o777, 0o755)

    def test_exclusive_state_directory_lock(self):
        state = self.root / "locked-state"
        clients = self.root / "clients"
        first = OCRService(state, clients)
        second = OCRService(state, clients)
        first.start()
        try:
            with self.assertRaises(RuntimeError):
                second.start()
        finally:
            first.close()

    def test_submission_and_retry_respect_queue_limits(self):
        service = OCRService(self.root / "quota-state", self.root / "clients")
        service.start()
        self.addCleanup(service.close)
        failed = service.submit("test-app", "stub", None, ".png", b"first")
        with service.connection() as db:
            db.execute(
                "UPDATE jobs SET status='failed' WHERE id=?", (failed["job_id"],)
            )
        service.submit("test-app", "stub", None, ".png", b"second")
        with patch("backend.ocr_service.MAX_ACTIVE_JOBS", 1):
            with self.assertRaises(HTTPException) as denied:
                service.retry(failed["job_id"], "test-app")
            self.assertEqual(denied.exception.status_code, 429)
        with patch("backend.ocr_service.MAX_CLIENT_BYTES", 12):
            with self.assertRaises(HTTPException) as denied:
                service.submit("test-app", "stub", None, ".png", b"third")
            self.assertEqual(denied.exception.status_code, 429)
        self.assertEqual(
            service.get_job(failed["job_id"], "test-app")["status"], "failed"
        )


if __name__ == "__main__":
    unittest.main()
