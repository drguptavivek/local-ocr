# Shared local OCR service

DigitVA, the document library, and other local application backends can submit
files to one OCR service. It listens at `http://127.0.0.1:8092` and uses the
model files already installed under `~/.models/ocr`. Its SQLite queue and
client credentials live under `~/.local/share/aiims-ocr` by default.

## Start the service

Use Python 3.13 and uv. The Make targets use the repository's `.venv` by
default. The [main project dependencies](../pyproject.toml) and
[lockfile](../uv.lock) cover the API, Docling's PDF pipeline, and IndicOCR.
Set `OCR_PYTHON` to use another compatible interpreter. Create a credential
per application and start the service in a terminal.

```bash
uv sync --locked --python 3.13
make ocr-client CLIENT=digitva
make ocr-client CLIENT=document-library
make ocr-service
```

The creation command prints only the credential path and preserves existing
credentials. Each backend reads its own `clients/<name>.token` file. Keep these
credentials on the application server. To use another private state directory,
set `OCR_SERVICE_STATE`. Set `OCR_SERVICE_CLIENTS` to place the credentials in a
different private directory. Set `OCR_SERVICE_PORT` when starting with Make to
change the port.

## Choose a device

Set the device when starting the service. Applications continue to use the
same HTTP and MCP requests.

| Host | Start command | Device selection |
| --- | --- | --- |
| CPU-only server | `OCR_DEVICE=cpu make ocr-service` | CPU for Docling, IndicOCR and all llama.cpp models, including the vision projector |
| NVIDIA workstation | `OCR_DEVICE=cuda make ocr-service` | CUDA for Docling, IndicOCR and llama.cpp models when their respective runtimes detect it |
| Apple Silicon | `OCR_DEVICE=mps make ocr-service` | PyTorch MPS and llama.cpp Metal when available |
| Automatic | `make ocr-service` | Detect Apple or NVIDIA acceleration for each runtime, with CPU fallback |

Tesseract, Anuvaad, EasyOCR and PP-OCRv5 use the CPU in every mode.
`GET /v1/pipelines` reports the effective `device` and pipeline availability.
Values include `cpu`, `mps`, `cuda`, `MTL0`, and `CUDA0`. An explicit
accelerator that a runtime cannot detect leaves its pipelines unavailable.
The device is recorded in newly computed results. Previously cached results
can be returned for identical requests. Use a separate state directory
(`OCR_SERVICE_STATE`) when comparing runs on different devices.

For a Linux NVIDIA workstation, use a driver compatible with the CUDA
runtime of the locked PyTorch build, and put CUDA-enabled `llama-mtmd-cli`
and `llama-server` on the service's `PATH`. Check the two runtimes separately.

```bash
.venv/bin/python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())'
llama-mtmd-cli --list-devices
llama-server --list-devices
```

For a 12 GB RTX 4070, the GGUF pipelines use their existing quantized weights.
If a model runs out of VRAM, lower `OCR_GPU_LAYERS` to move some model layers
to the CPU, for example `OCR_DEVICE=cuda OCR_GPU_LAYERS=20 make ocr-service`.
The default requests 99 layers. This setting applies to llama.cpp, including
Surya, and leaves its vision projector on the GPU. CUDA workers retain job
time limits but skip the Linux virtual-address-space cap, which can prevent
CUDA initialization. Use the host's service manager for physical RAM limits
if needed.

### Prepare PaddleOCR-VL for CPU

The upstream PaddleOCR-VL 1.6 vision projector contains BF16 tensors. On
the tested Mac, image encoding on CPU took over six minutes with that file.
Prepare a Q8 copy once per model store using the installed upstream
projector. The service uses the smaller copy in CPU mode and the upstream projector
in GPU mode. The preparation tool leaves the upstream file in place.

```bash
uv run --no-project --python 3.13 --with gguf==0.19.0 tools/prepare-paddle-projector.py
```

Copy the prepared projector with the other model files if moving them to a
CPU-only server. The pipeline catalog reports when the CPU projector is missing.

### Ubuntu and Windows GPU testing

On Ubuntu, use a compatible NVIDIA Linux driver and verify with `nvidia-smi`.
On Windows, install the NVIDIA Windows driver and run the service in Ubuntu
under WSL2. Run `nvidia-smi` inside Ubuntu as well. The service uses POSIX
file permissions and process controls. Keep its checkout, models, and queue
in the Ubuntu filesystem. See the [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

From the Ubuntu shell, create the project-local environment with
`uv sync --locked --python 3.13`. Install Tesseract and its English,
orientation and script/Devanagari language data, and check these with
`tesseract --list-langs`. Install the CUDA-enabled llama.cpp executables
for the GGUF models and make the same model files available under
`~/.models/ocr`. The file paths are defined in
[the pipeline catalog](../backend/ocr_pipelines.py). Docling uses its own
local model cache. Install the optional environment below for Surya,
EasyOCR or PP-OCRv5.

Start with `OCR_DEVICE=cuda make ocr-service`. Confirm `/health` and check
`/v1/pipelines` with an app credential. Then submit a synthetic image and
a two-page PDF with embedded and scanned text, and check the returned
`result.metadata.device` before testing private inputs.

## Optional image engines

Surya, EasyOCR, and PP-OCRv5 use a separate repository-local environment.
Install its existing pinned versions with uv if you need those engines.

```bash
uv venv --python 3.13 .venv-benchmark
uv pip sync --python .venv-benchmark/bin/python requirements-benchmark.txt
```

Set `OCR_BENCHMARK_PYTHON` to point to another compatible interpreter.
Tesseract and llama.cpp are system binaries. Model weights and the Docling
model cache are shared separately from the Python environments.

## Check health and workload

| Endpoint | Authentication | Response |
| --- | --- | --- |
| `GET /health` | None | HTTP 200 with `{"status":"ok"}` while the worker is running, or HTTP 503 when the worker is unavailable |
| `GET /v1/jobs/{job_id}` | App bearer token | Job state, execution timeout, and result or error when available |
| `GET /v1/workload` | App bearer token | Shared queue occupancy and the caller's retained job and storage counts |

The workload response has two objects. `queue` contains `queued`,
`processing`, `active_limit`, and `available_slots` across the service.
`client` contains `jobs_by_status`, `jobs_by_pipeline` (counts by status for
each pipeline), `stored_jobs`, `stored_bytes`, `max_stored_jobs`, and
`max_stored_bytes`. Available slots describe the shared active-job limit.
App and service storage quotas also apply when submitting. Counts reflect
currently retained jobs and drop when jobs are deleted. The response contains
no job IDs, inputs, or extracted text.

## Submit and retrieve a job

Use `GET /v1/pipelines` to see the installed candidates, their input formats,
and their availability. For PDFs use the Docling pipeline. For raster images
select the appropriate handwriting or printed-text pipeline.

| Pipeline | Input | Processing |
| --- | --- | --- |
| `docling` | PDF | Docling default layout and region OCR with embedded text preserved |
| `indicocr_block`, `indicocr_layout` | Image | Bodhan handwriting and print recognition, with optional layout detection |
| `surya25` | Image | Surya 2.5 Q8 OCR via local llama.cpp |
| `tesseract`, `anuvaad` | Image | CPU printed-text baselines |
| `easyocr`, `ppocrv5` | Image | Devanagari CPU OCR candidates |
| `chandra2`, `lightonocr`, `paddleocr_vl`, `glm_ocr` | Image | Local GGUF comparison candidates |

## Install Tesseract language packs

The multilingual vision models use their own weights across languages.
EasyOCR and PP-OCRv5 use recognizer checkpoints selected for the script.
Tesseract uses one `.traineddata` file per language or script.

Install individual packs into the shared model store from the repository root.

```bash
.venv/bin/python tools/install-tessdata.py ben guj pan tel
tesseract --list-langs --tessdata-dir "$HOME/.models/ocr/tesseract-ocr/tessdata"
```

This downloads Bengali, Gujarati, Punjabi, and Telugu from the official
`tessdata_fast` release and preserves existing packs. Use the traineddata
code in `languages`, for example `["ben", "eng"]`. Existing aliases such as
`hi` and `en` still work. Use `GET /v1/pipelines` to confirm availability.
The [official list](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions)
contains the pack codes.

Docling's PDF OCR uses the system Tesseract data directory. On Ubuntu or
WSL2, install packages such as `tesseract-ocr-ben` and `tesseract-ocr-tel`.
On Homebrew, `brew install tesseract-lang` installs additional language data.
Run `tesseract --list-langs` to see the system inventory. You can also use the
installer's `--directory` option to write to an existing system tessdata
directory if your account has write access. Restart the OCR service after
changing the system inventory. See the [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html).

## Language hints

For English only, send `languages=en`. For English and Hindi, repeat the
multipart field `languages`. In a Python HTTPX upload, use
`data={"pipeline": "docling", "languages": ["en", "hi"]}`. The API accepts
up to eight language hints and returns them in `languages`, in request order
with duplicate values removed. The legacy `language` field accepts one
value. Send either `languages` or `language`. The job's `language` response
is populated when exactly one hint was supplied.

For an explicit list, Tesseract and Docling load the corresponding installed
recognition data in that order. Anuvaad accepts Hindi. The multilingual
neural models detect script from the image and use the hints for output
checks. A check can pass when text matches any requested script. The hints
do not establish which languages are present in the source. Consult
`GET /v1/pipelines` for `languages` and `recognition_languages` entries.
When no list is supplied, Docling keeps its existing Devanagari and English
configuration.

The Surya SDK source is pinned to v0.23.0 at commit
`9d5aadfebb07939755c24e8a80d0e61b3be80895`. Its source and model licenses are
included with the vendored SDK. The worker uses the same preinstalled
`llama-server` and weights used for the local evaluation.

`POST /v1/jobs` accepts a multipart upload and returns HTTP 200 with a job
record. Save the job ID with the source record before polling.

```json
{
  "job_id": "cf25e449-02a8-4a39-be5d-c45057f28e45",
  "pipeline": "indicocr_layout",
  "language": "hi",
  "languages": ["hi"],
  "status": "queued",
  "execution_timeout_seconds": 900,
  "created_at": 1794374576.91,
  "updated_at": 1794374576.91
}
```

`execution_timeout_seconds` is the pipeline's outer execution limit per
attempt. It excludes time waiting in the queue and is independent of the
caller's HTTP timeout. The limit is 180 seconds for Tesseract, Anuvaad, and
the direct GGUF pipelines on GPU, and 900 seconds for Docling, CPU GGUF
runs, and the Python image workers. The pipeline catalog also reports it.

```python
from pathlib import Path
from time import monotonic, sleep

import httpx


def wait_for_job(client, job_url, wait_seconds=60):
    deadline = monotonic() + wait_seconds
    while monotonic() < deadline:
        try:
            response = client.get(job_url, timeout=10)
            response.raise_for_status()
            job = response.json()
        except httpx.TransportError:
            sleep(2)
            continue
        if job["status"] in {"completed", "failed"}:
            return job
        sleep(2)
    return None


token = (Path.home() / ".local/share/aiims-ocr/clients/digitva.token").read_text().strip()
with httpx.Client(
    base_url="http://127.0.0.1:8092",
    headers={"Authorization": f"Bearer {token}"},
    timeout=30,
) as client:
    with Path("sample.jpg").open("rb") as image:
        job = client.post(
            "/v1/jobs",
            data={"pipeline": "indicocr_layout", "language": "hi"},
            files={"file": ("sample.jpg", image, "image/jpeg")},
        ).raise_for_status().json()

    job_url = f"/v1/jobs/{job['job_id']}"
    # Persist the job ID with the source record here.
    final = wait_for_job(client, job_url)
    if final and final["status"] == "failed" and final.get("error") == "pipeline_timeout":
        client.post(f"{job_url}/retry").raise_for_status()
        final = wait_for_job(client, job_url)

    if final is None:
        print(f"Resume polling {job_url} later")
    elif final["status"] == "completed":
        result = final["result"]
        # Persist result["text"] and result["flags"] in the calling application.
        print("Completed", job["job_id"], result["flags"])
    else:
        print("Failed", job["job_id"], final["error"])
```

A polling deadline or HTTP transport timeout leaves the job active. Resume
polling with the saved ID. If the upload times out before the ID arrives,
submit the same file with the same credential, pipeline, and language hints. The
queue deduplicates matching submissions while the job exists.

An engine timeout appears on `GET /v1/jobs/{job_id}` as HTTP 200 with
`"status": "failed"` and `"error": "pipeline_timeout"`. Other processing
errors use `pipeline_failed` or `storage_quota`. Retry uses the same job ID
and retained input. If the retry request times out, poll the job to see
whether the service accepted it. The example retries once per invocation
when it observes a pipeline timeout. In an app, persist the retry count
across polling sessions.

## Input limits and processing resolution

Input limits apply to the uploaded original.

| Limit | Maximum |
| --- | --- |
| File size | 32 MiB |
| PDF pages | 20, with a minimum of 1 |
| PDF page dimensions | 36 inches on either side |
| Raster image dimensions | 12,000 pixels on either side and 50 megapixels total |

An oversized upload returns HTTP 413 immediately. File readability, page
count and decoded dimensions are checked by the worker. Polling a failed
job returns HTTP 200 with a stable `error` and a readable `error_detail`.

| Worker error | Meaning |
| --- | --- |
| `invalid_image` | The file could not be read as a JPEG, PNG or TIFF |
| `image_too_large` | The decoded image exceeds the input bounds |
| `invalid_pdf` | The file could not be read as a PDF |
| `pdf_encrypted` | The PDF requires a password |
| `pdf_page_limit` | The PDF is outside the supported page count |
| `pdf_page_too_large` | A page exceeds the supported physical dimensions |
| `pipeline_unavailable` | The selected engine or its local prerequisites are unavailable |

Before inference, accepted raster images are reduced in memory as needed
to fit within 4,096 pixels per side and 12 megapixels, preserving aspect
ratio. The original upload remains intact. Successful image results report
`result.metadata.image_resize` when this resize was applied.

Docling keeps the PDF text layer available. Its OCR-region scale is bounded
by the same raster dimensions, with the default retained for ordinary
pages. If Docling needs a page-image fallback, it renders the page directly
at up to 300 DPI within the same bounds. Individual engines can use smaller
rasters.

## Process a folder and save sidecars

Create a dedicated client credential so the batch owns its queue jobs.

```bash
make ocr-client CLIENT=benchmark
.venv/bin/python tools/ocr-batch.py /path/to/inputs --client benchmark \
  --pipeline docling --pipeline paddleocr_vl --language-from-folder
```

The command walks subfolders and processes one file at a time. Repeat
`--pipeline` or use `--all-pipelines` for every available engine. The batch
skips file formats and language hints a pipeline does not support. Use
`--languages hi en` to set explicit hints, or `--language-from-folder` for
immediate parent folders such as `project/hindi/file.png`.

The command writes a private JSON file next to each input, for example
`page.png.paddleocr_vl.hi.json`. It saves the result or failure, the input
hash, the device reported by the catalog, and wall-clock seconds from submit
through polling. Terminal jobs are deleted from the service after the
sidecar is written. A rerun skips a completed sidecar matching the input
hash, pipeline, language and device. Use `--overwrite` to rerun completed
sidecars. Copy a clean input pack to each workstation when measuring
throughput, and use a dedicated benchmark queue to avoid pre-existing
cached jobs and unrelated work.

## Result storage and cleanup

Jobs and results are persisted in the private SQLite database
`~/.local/share/aiims-ocr/queue.sqlite3` by default. They survive service
restarts. On success the service clears the input bytes and keeps the result.
Failed jobs retain input for retry. The service has no age-based expiration.
After the application saves a result, use `DELETE /v1/jobs/{job_id}` to remove
the completed or failed record and any retained input. Success returns
HTTP 204.

The queue admits up to 100 active jobs, with 32 MiB per upload and private
storage limits of 512 MiB per app and 2 GiB total. For a bulk run, submit
in batches and delete each service job after storing its result.

DigitVA should save the text and source linkage in its private evaluation
directory. For the document library, store and sanitize the returned Markdown
through the existing archive application before synchronizing the record to
Typesense. OCR quality flags identify outputs that need review. Verify
transcription accuracy against the source image.
