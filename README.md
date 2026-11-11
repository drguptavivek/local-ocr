# Local OCR service

A localhost OCR service for DigitVA, the AIIMS document library, and other
local applications. It provides an authenticated HTTP API, a persistent
SQLite queue that resumes after a restart, and one OCR worker.

## Start

From the repository root, install the locked main environment with uv.
Then create a credential for each application that needs access.

```bash
uv sync --locked --python 3.13
make ocr-client CLIENT=digitva
make ocr-client CLIENT=document-library
make ocr-service
curl -fsS http://127.0.0.1:8092/health
```

Credential creation preserves existing tokens. If the service is already
running, use the health check directly. The default state directory is
`~/.local/share/aiims-ocr`. The service reads existing model weights from
`~/.models/ocr`. The main runtime uses `.venv` and the [uv lockfile](uv.lock).
The [optional benchmark requirements](requirements-benchmark.txt) use
`.venv-benchmark`.

## Choose a device

Set `OCR_DEVICE=cpu` for a CPU-only server, `OCR_DEVICE=cuda` for an NVIDIA
workstation, or `OCR_DEVICE=mps` for Apple Silicon. The default `auto` chooses
an available device for each runtime. See the
[device setup guide](docs/shared-local-ocr.md#choose-a-device) for requirements
and how to check the effective device.

## Choose a pipeline

| Input | Pipelines |
| --- | --- |
| PDF | Docling with layout detection, embedded text, and region OCR |
| Handwritten image | IndicOCR block or layout, Surya 2.5 |
| Printed image | Tesseract, Anuvaad, EasyOCR, PaddleOCR PP-OCRv5 |
| Image comparison | Chandra 2, LightOnOCR, PaddleOCR-VL, GLM-OCR |

Check `GET /v1/pipelines` for availability and supported inputs. Submit a
file to `POST /v1/jobs` and poll `GET /v1/jobs/{job_id}`. Store the job ID
with the source record in the calling app. The [API guide](docs/shared-local-ocr.md)
has the authenticated example, settings, limits, and retry commands.

## Use from an MCP client

With the OCR HTTP service running, set up the optional adapter from the repo
root.

```bash
uv sync --locked --group mcp
make ocr-client CLIENT=mcp
mkdir -p -m 700 private/inbox
```

Add the server entry from [mcp.json](mcp.json) to your MCP client's server
configuration. Replace the two example paths with absolute paths in your
checkout. The adapter runs over stdio and talks to the loopback OCR
service. Its default port is 8092 and its credential is
`~/.local/share/aiims-ocr/clients/mcp.token`. Set `OCR_MCP_PORT` or
`OCR_MCP_TOKEN_FILE` in the server entry for a custom service configuration.
A plain `uv sync --locked` removes the optional group, so use
`uv sync --locked --group mcp` when maintaining an MCP-enabled checkout.

Submit approved files from `private/inbox` or set `OCR_MCP_INPUT_ROOT` to a
different dedicated input directory. `ocr_job_status` omits extracted text
by default. Set `include_text` to `true` to send it to the connected MCP
client. See the [OCR skill](.agents/skills/local-ocr/SKILL.md) and
[llms.txt](llms.txt) for agent guidance.

## Validate

Run `make test` for the queue and adapter tests. The service was also tested
through HTTP with synthetic input on all 12 pipelines. The Docling test
verified both embedded PDF text and OCR from a scanned page.

See [AGENTS.md](AGENTS.md) for repository guidance.

## License

The service code is [MIT-licensed](LICENSE). The vendored Surya SDK includes
its own [source license](vendor/surya-0.23.0/LICENSE) and
[model license](vendor/surya-0.23.0/MODEL_LICENSE). Model weights are installed
separately under their respective terms.
