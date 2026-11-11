# Agent guidance

This repository provides the shared local OCR API used by DigitVA, the AIIMS
document library, and other local applications. Start with [README.md](README.md)
and the [service and API guide](docs/shared-local-ocr.md).

## Runtime and data

- Use the repository-local `.venv` from `uv sync --locked` and reuse weights
  under `~/.models/ocr`. Keep optional OCR engines in `.venv-benchmark`.
  Changes to packages, caches, and model files need a task-specific reason.
- Keep the service bound to loopback. Authenticate application requests
  with per-app credentials and enforce job ownership across API endpoints.
- Keep service state and credentials in the private runtime directory.
  Keep source records and approved transcriptions in the owning application.
  Exclude documents, OCR text, and bearer tokens from logs and Git.
- Run one OCR job at a time. Preserve queue limits, explicit retry, restart
  recovery, and child-process cleanup when changing the worker.
- Use Docling's default PDF-aware mode for PDF jobs. Preserve readable
  embedded text and OCR image regions as needed. Expose image engines
  through the pipeline catalog with accurate capability and availability
  information.
- Return model provenance and quality flags. Non-empty output indicates
  that a model returned text. Transcription accuracy requires comparison
  with the source or a verified reference.

## Development and verification

- `backend/ocr_service.py` owns the API and queue. `backend/ocr_pipelines.py`
  owns pipeline selection and validation. `backend/ocr_pipeline_worker.py`
  runs the heavier image engines.
- Keep the vendored Surya SDK pinned to the revision recorded in the
  service guide. Preserve its license files.
- Run `make test` after code changes. Run Ruff on first-party Python files.
  For adapter changes, smoke-test the affected pipelines through the HTTP
  API with synthetic inputs. Use a PDF with both embedded text and a
  scanned page when changing Docling.
- Inspect the worktree before editing and preserve unrelated changes.
