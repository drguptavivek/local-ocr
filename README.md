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

## Download model weights

Keep downloaded models in `~/.models/ocr/<publisher>/<model>`. The
[pipeline catalog code](backend/ocr_pipelines.py) lists the expected files.
Tesseract packs are public GitHub downloads. Hugging Face gates access to
some model repositories, including [Bodhan IndicOCR](https://huggingface.co/bodhan-ai/indic-ocr/tree/main).

1. Sign in on Hugging Face, open the gated model's page, review its license,
   and submit the access form. Wait for approval if the publisher requires it.
2. Create a read token for the same account with access to the repository.
   Authenticate on the machine that will download the files.

   ```bash
   .venv/bin/hf auth login
   .venv/bin/hf auth whoami
   ```

3. Download the model to the expected local directory.

   ```bash
   .venv/bin/hf download bodhan-ai/indic-ocr \
     --local-dir "$HOME/.models/ocr/bodhan-ai/indic-ocr"
   ```

Add `--revision <commit>` when reproducing a benchmark. For unattended
downloads, inject `HF_TOKEN` from your local secret store. Keep the token out
of Git, sidecars, and the workstation test pack. Once weights are present,
OCR inference uses the local files and runs without a Hugging Face token.
See the [Hugging Face gated models guide](https://huggingface.co/docs/hub/models-gated).

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

## Tested models, sizes, and languages

Sizes are decimal GB of weights on disk. GGUF totals include the vision
projector. The variants completed a local benchmark or smoke test. Check
the pipeline catalog for the variant exposed by the service on this host.

| Model family | Tested variants and weight sizes | Language capability |
| --- | --- | --- |
| Docling PDF layout + Tesseract | Heron + TableFormer Accurate **0.384 GB**, plus Tesseract data | The default uses English and Devanagari. Explicit OCR hints use the system Tesseract language packs. |
| Tesseract | Hindi **0.001 GB**. All seven local language packs **0.021 GB** | English, Hindi, Marathi, Tamil, Malayalam, Kannada, and Odia in the test installation. Additional languages use separate [Tesseract packs](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions). |
| Anuvaad | Hindi **0.011 GB** | Hindi. Uses its own fine-tuned Tesseract checkpoint. |
| EasyOCR | FP32 and runtime INT8 share a **0.299 GB** download | This service uses the Hindi and English reader. Other scripts use their own recognizer files. |
| PaddleOCR PP-OCRv5 | Detector + Devanagari recognizer **0.096 GB** | This service accepts Hindi, Marathi, and English hints. Other scripts use separate recognizer checkpoints. |
| PaddleOCR-VL 1.6 | Q4 + Q8 projector **0.898 GB**. Q4 + FP16 or BF16 projector **1.182 GB**. BF16 + BF16 projector **1.818 GB** | One multilingual OCR checkpoint. The CPU Q8 smoke tests use English and Hindi. [Model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) |
| Bodhan IndicOCR | BF16 block **1.734 GB**. BF16 block + FP32 layout **1.868 GB** | One recognizer covers English and 22 Indian languages for print, and English and 12 Indian languages for handwriting, including Hindi, Marathi, Tamil, Malayalam, Kannada, and Odia. [Supported languages](https://huggingface.co/bodhan-ai/indic-ocr#supported-languages) |
| Chandra OCR 2 | Q2 **2.491 GB**, Q4 **3.433 GB**, Q8 **5.525 GB** | One checkpoint. Upstream reports over 90 languages and includes Hindi in its benchmark. [Model card](https://huggingface.co/datalab-to/chandra-ocr-2) |
| Surya OCR 2 | Q4 **0.609 GB**, Q8 **0.879 GB**, FP16 **1.471 GB** | One checkpoint. Upstream evaluated English, Hindi, Marathi, Tamil, Malayalam, Kannada, Odia, and other languages. [Language results](https://huggingface.co/datalab-to/surya-ocr-2.5/blob/main/static/docs/multilingual.md) |
| Surya OCR 2.5 | Q8 **0.836 GB** | One checkpoint. Upstream evaluated the same 91-language suite as Surya 2. [Language results](https://huggingface.co/datalab-to/surya-ocr-2.5/blob/main/static/docs/multilingual.md) |
| GLM-OCR | Q2 **0.846 GB**, Q4 **1.033 GB**, Q8 **1.435 GB** | One checkpoint. The model card lists Chinese, English, French, Spanish, Russian, German, Japanese, and Korean. Hindi was exploratory in the local benchmark. [Model card](https://huggingface.co/zai-org/GLM-OCR) |
| LightOnOCR 3 0.8B | Q4 **0.734 GB**, Q8 **1.017 GB** | One checkpoint. The model card tags English. Hindi was exploratory in the local benchmark. [Model card](https://huggingface.co/lightonai/LightOnOCR-3-0.8B) |
| Gemma 3n E2B | MLX 4-bit **4.463 GB** | Benchmark-only general vision model. The upstream language training list describes the base model. [Model card](https://ai.google.dev/gemma/docs/gemma-3n/model_card) |
| TrOCR Small Handwritten | PyTorch weights **0.246 GB** | English single-line handwriting checkpoint. The standalone smoke test returned different text from the synthetic reference. [Model card](https://huggingface.co/microsoft/trocr-small-handwritten) |
| TrOCR Devanagari 2 | Safetensors weights **0.736 GB** | Devanagari single-line OCR checkpoint. The standalone Hindi smoke test returned different text from the synthetic reference. [Model card](https://huggingface.co/paudelanil/trocr-devanagari-2) |

The language column describes configuration and upstream capability. Local
accuracy on a specific script or handwriting style requires verified
reference text. See [installing language packs](docs/shared-local-ocr.md#install-tesseract-language-packs)
and [batching a folder](docs/shared-local-ocr.md#process-a-folder-and-save-sidecars).

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
