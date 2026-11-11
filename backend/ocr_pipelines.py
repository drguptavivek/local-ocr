"""Bounded, local-only adapters for the OCR engines in the local catalog.

The module does not download weights. Model files are resolved below
``~/.models/ocr`` and optional Python engines are advertised only when both
their local weights and runtime dependencies are present.
"""

from __future__ import annotations

from collections import Counter
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import tempfile
from functools import cache
from math import isfinite
from pathlib import Path
from typing import Any

if __package__:
    from .ocr_device import (
        device_mode,
        llama_args,
        llama_device,
        probe_torch_device,
        torch_device,
    )
    from .ocr_limits import (
        MAX_PAGES,
        MAX_PDF_PAGE_POINTS,
        MAX_IMAGE_SIDE,
        MAX_IMAGE_PIXELS,
        InputError,
        bounded_raster_scale,
    )
else:
    from ocr_device import (
        device_mode,
        llama_args,
        llama_device,
        probe_torch_device,
        torch_device,
    )
    from ocr_limits import (
        MAX_PAGES,
        MAX_PDF_PAGE_POINTS,
        MAX_IMAGE_SIDE,
        MAX_IMAGE_PIXELS,
        InputError,
        bounded_raster_scale,
    )

MODEL_ROOT = Path.home() / ".models" / "ocr"
MAX_INPUT_BYTES = 100 * 1024 * 1024
TIMEOUT_SECONDS = 900
SHORT_TIMEOUT_SECONDS = 180
OUTPUT_CHAR_LIMIT = 2_000_000
_LANG = {
    "en": ("eng", "Latin"),
    "english": ("eng", "Latin"),
    "hi": ("hin", "Devanagari"),
    "hindi": ("hin", "Devanagari"),
    "mr": ("mar", "Devanagari"),
    "ta": ("tam", "Tamil"),
    "ml": ("mal", "Malayalam"),
    "kn": ("kan", "Kannada"),
    "or": ("ori", "Odia"),
    "od": ("ori", "Odia"),
    "marathi": ("mar", "Devanagari"),
    "tamil": ("tam", "Tamil"),
    "malayalam": ("mal", "Malayalam"),
    "kannada": ("kan", "Kannada"),
    "odia": ("ori", "Odia"),
}
_SCRIPT_RANGES = {
    "en": (0x0041, 0x007A),
    "english": (0x0041, 0x007A),
    "hi": (0x0900, 0x097F),
    "hindi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
    "marathi": (0x0900, 0x097F),
    "ta": (0x0B80, 0x0BFF),
    "tamil": (0x0B80, 0x0BFF),
    "ml": (0x0D00, 0x0D7F),
    "malayalam": (0x0D00, 0x0D7F),
    "kn": (0x0C80, 0x0CFF),
    "kannada": (0x0C80, 0x0CFF),
    "or": (0x0B00, 0x0B7F),
    "od": (0x0B00, 0x0B7F),
    "odia": (0x0B00, 0x0B7F),
}
_GGUF = {
    "chandra2": (
        "prithivMLmods/chandra-ocr-2-GGUF/chandra-ocr-2.Q4_K_M.gguf",
        "prithivMLmods/chandra-ocr-2-GGUF/chandra-ocr-2.mmproj-q8_0.gguf",
        "Chandra OCR 2 Q4_K_M",
    ),
    "lightonocr": (
        "richardyoung/LightOnOCR-3-0.8B-GGUF/LightOnOCR-3-0.8B-Q4_K_M.gguf",
        "richardyoung/LightOnOCR-3-0.8B-GGUF/mmproj-LightOnOCR-3-0.8B-F16.gguf",
        "LightOnOCR-3 Q4_K_M",
    ),
    "paddleocr_vl": (
        "PaddlePaddle/PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6.Q4_K_M.gguf",
        "PaddlePaddle/PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        "PaddleOCR-VL 1.6 Q4_K_M",
    ),
    "glm_ocr": (
        "mradermacher/GLM-OCR-GGUF/GLM-OCR.Q4_K_M.gguf",
        "mradermacher/GLM-OCR-GGUF/GLM-OCR.mmproj-Q8_0.gguf",
        "GLM-OCR Q4_K_M",
    ),
}
_CHANDRA_LAYOUT_PROMPT = """OCR this image to HTML, arranged as layout blocks. Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format. Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels: Caption, Footnote, Equation-Block, List-Group, Page-Header, Page-Footer, Image, Section-Header, Table, Text, Complex-Block, Code-Block, Form, Table-Of-Contents, Figure, Chemical-Block, Diagram, Bibliography, Blank-Page.

Only use these tags: math, br, i, b, u, del, sup, sub, table, tr, td, p, th, div, pre, h1, h2, h3, h4, h5, ul, ol, li, input, a, span, img, hr, tbody, small, caption, strong, thead, big, code, chem. Only use these attributes: class, colspan, rowspan, display, checked, type, border, value, style, href, alt, align, data-bbox, data-label.

Guidelines: Represent inline math with KaTeX-compatible LaTeX inside math tags. Use colspan and rowspan for tables. Preserve formatting, spacing, indentation, subscripts and superscripts. Describe images in alt text and do not provide an image src. Join paragraph lines naturally and use br only when needed. Preserve list indentation and markers. Return the simplest accurate HTML structure, with correct natural reading order. Output only HTML."""


def _file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 1_000_000
    except OSError:
        return False


def _module(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _indic_paths() -> tuple[Path, Path]:
    root = MODEL_ROOT / "bodhan-ai" / "indic-ocr"
    return root / "weights" / "ocr", root / "weights" / "layout"


def _surya_source() -> Path:
    configured = os.environ.get("OCR_SURYA_SOURCE")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[1] / "vendor" / "surya-0.23.0"
    )


def _benchmark_python() -> Path:
    configured = os.environ.get("OCR_BENCHMARK_PYTHON")
    default = Path(__file__).resolve().parents[1] / ".venv-benchmark/bin/python"
    return Path(configured).expanduser() if configured else default


def _benchmark_has(*packages: str) -> bool:
    python = _benchmark_python()
    return python.is_file() and all(
        any(python.parent.parent.glob(f"lib/python*/site-packages/{name}"))
        for name in packages
    )


@cache
def _system_tessdata(binary: str) -> set[str]:
    env = dict(os.environ)
    env.pop("TESSDATA_PREFIX", None)
    try:
        result = subprocess.run(
            [binary, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        return set(result.stdout.splitlines()) if result.returncode == 0 else set()
    except (OSError, subprocess.TimeoutExpired):
        return set()


def _indic_weights(directory: Path) -> bool:
    return (directory / "config.json").is_file() and any(
        _file(path) for path in directory.glob("model*.safetensors")
    )


def execution_timeout_seconds(pipeline: str) -> int:
    """Outer execution limit for one attempt, excluding time in the queue."""
    if (
        pipeline in _GGUF
        and llama_device(shutil.which("llama-mtmd-cli"), device_mode()) == "cpu"
    ):
        return TIMEOUT_SECONDS
    if pipeline in _GGUF or pipeline in {"tesseract", "anuvaad"}:
        return SHORT_TIMEOUT_SECONDS
    return TIMEOUT_SECONDS


def pipeline_catalog() -> dict[str, dict[str, Any]]:
    mode = device_mode()
    torch_backend = probe_torch_device(mode) if _module("torch") else None
    surya_backend = llama_device(shutil.which("llama-server"), mode)
    tess = shutil.which("tesseract")
    tessdata = MODEL_ROOT / "tesseract-ocr" / "tessdata"
    anuvaad = (
        MODEL_ROOT / "project-anuvaad" / "anuvaad-ocr-model" / "anuvaad_hin.traineddata"
    )
    ocr_w, layout_w = _indic_paths()
    indic_root = MODEL_ROOT / "bodhan-ai" / "indic-ocr"
    surya_model = (
        MODEL_ROOT / "datalab-to" / "surya-ocr-2.5-gguf" / "surya-2.5-Q8_0.gguf"
    )
    surya_mmproj = (
        MODEL_ROOT / "datalab-to" / "surya-ocr-2.5-gguf" / "surya-2.5-mmproj.gguf"
    )
    surya_source = _surya_source()
    surya_deps = _benchmark_has("openai", "platformdirs", "torch", "transformers")
    indic_deps = all(
        _module(name) for name in ("torch", "torchvision", "transformers", "PIL")
    )
    indic_ready = bool(indic_deps and torch_backend and _indic_weights(ocr_w))
    result: dict[str, dict[str, Any]] = {
        "docling": {
            "extensions": [".pdf"],
            "available": bool(
                tess
                and _module("docling")
                and torch_backend
                and {"eng", "script/Devanagari", "osd"} <= _system_tessdata(tess)
            ),
            "model": "Docling default PDF layout + region OCR",
            "version": "2.133.0 (configured)",
        },
        "tesseract": {
            "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "available": bool(tess and (tessdata / "eng.traineddata").is_file()),
            "model": "Tesseract",
            "version": _version(tess) if tess else None,
        },
        "anuvaad": {
            "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "available": bool(tess and _file(anuvaad)),
            "model": "Anuvaad Hindi Tesseract",
            "version": _version(tess) if tess else None,
        },
        "indicocr_block": {
            "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "available": indic_ready and (indic_root / "idp_recognizer.py").is_file(),
            "model": "Bodhan IndicOCR block recognizer",
            "version": "local checkpoint",
        },
        "indicocr_layout": {
            "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "available": indic_ready
            and (indic_root / "indic_ocr.py").is_file()
            and _indic_weights(layout_w),
            "model": "Bodhan IndicOCR layout",
            "version": "local checkpoint",
        },
        "surya25": {
            "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "available": bool(
                surya_backend
                and surya_deps
                and (surya_source / "surya" / "__init__.py").is_file()
                and _file(surya_model)
                and _file(surya_mmproj)
            ),
            "model": "Surya OCR 2.5 GGUF Q8_0",
            "version": "Surya SDK 0.23.0",
        },
        "easyocr": {
            "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "available": _benchmark_has("easyocr")
            and all(
                _file(MODEL_ROOT / "JaidedAI/EasyOCR" / f)
                for f in ("craft_mlt_25k.pth", "devanagari.pth")
            ),
            "model": "EasyOCR Hindi and English",
            "version": "1.7.2",
        },
        "ppocrv5": {
            "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "available": _benchmark_has("paddleocr", "paddle")
            and all(
                _file(MODEL_ROOT / "PaddlePaddle" / p / "inference.pdiparams")
                for p in ("PP-OCRv5_server_det", "devanagari_PP-OCRv5_mobile_rec")
            ),
            "model": "PaddleOCR PP-OCRv5 Devanagari",
            "version": "3.7.0",
        },
    }
    result["docling"]["recognition_languages"] = sorted(
        code
        for code, (trained, _) in _LANG.items()
        if tess and trained in _system_tessdata(tess)
    )
    result["tesseract"]["languages"] = sorted(
        code
        for code, (trained, _) in _LANG.items()
        if (tessdata / f"{trained}.traineddata").is_file()
    )
    result["anuvaad"]["languages"] = ["hi", "hindi"]
    result["easyocr"]["languages"] = ["en", "english", "hi", "hindi"]
    result["ppocrv5"]["languages"] = ["en", "english", "hi", "hindi", "mr", "marathi"]
    llama = shutil.which("llama-mtmd-cli")
    llama_backend = llama_device(llama, mode)
    for key, (model, mmproj, label) in _GGUF.items():
        result[key] = {
            "extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "available": bool(
                llama_backend
                and _file(MODEL_ROOT / model)
                and _file(MODEL_ROOT / mmproj)
            ),
            "model": label,
            "version": "local GGUF",
        }
    devices = {
        "docling": torch_backend,
        "indicocr_block": torch_backend,
        "indicocr_layout": torch_backend,
        "surya25": surya_backend,
        **dict.fromkeys(_GGUF, llama_backend),
    }
    for pipeline, entry in result.items():
        entry["device"] = devices.get(pipeline, "cpu")
        entry["execution_timeout_seconds"] = execution_timeout_seconds(pipeline)
    return result


def _version(binary: str | None) -> str | None:
    if not binary:
        return None
    try:
        run = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=5
        )
        return (run.stdout or run.stderr).strip().splitlines()[0][:120]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def _validate(source: Path, pipeline: str) -> Path:
    path = source.expanduser().resolve(strict=True)
    catalog = pipeline_catalog()
    if pipeline not in catalog:
        raise ValueError(f"unknown OCR pipeline: {pipeline}")
    if path.suffix.lower() not in catalog[pipeline]["extensions"]:
        raise ValueError("unsupported input extension")
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("input must be a regular file under 100 MiB")
    if not catalog[pipeline]["available"]:
        raise InputError("pipeline_unavailable")
    if path.suffix.lower() == ".pdf":
        import pymupdf

        if pipeline != "docling":
            raise InputError("invalid_pdf")
        try:
            with pymupdf.open(path) as pdf:
                if not pdf.is_pdf:
                    raise InputError("invalid_pdf")
                if pdf.needs_pass:
                    raise InputError("pdf_encrypted")
                if not 1 <= pdf.page_count <= MAX_PAGES:
                    raise InputError("pdf_page_limit")
                for page in pdf:
                    width, height = page.rect.width, page.rect.height
                    if not all(isfinite(side) and side > 0 for side in (width, height)):
                        raise InputError("invalid_pdf")
                    if max(width, height) > MAX_PDF_PAGE_POINTS:
                        raise InputError("pdf_page_too_large")
        except (pymupdf.FileDataError, OSError) as exc:
            raise InputError("invalid_pdf") from exc
    else:
        from PIL import Image

        try:
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG", "TIFF"}:
                    raise InputError("invalid_image")
                width, height = image.size
                if (
                    max(width, height) > MAX_IMAGE_SIDE
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise InputError("image_too_large")
                image.verify()
        except Image.DecompressionBombError as exc:
            raise InputError("image_too_large") from exc
        except (OSError, SyntaxError) as exc:
            raise InputError("invalid_image") from exc
    return path


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TESSDATA_PREFIX", None)
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(Path.home() / ".models" / "huggingface"),
        }
    )
    return env


_PROCESS_LOCK = threading.Lock()
_RUNNING: set[int] = set()
_STOPPED = False


def reset_cancellation() -> None:
    global _STOPPED
    with _PROCESS_LOCK:
        _STOPPED = False


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def cancel_running() -> None:
    global _STOPPED
    with _PROCESS_LOCK:
        _STOPPED = True
        for pid in _RUNNING:
            _kill_group(pid)


def _run(
    argv: list[str],
    *,
    timeout: int = TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    with _PROCESS_LOCK:
        if _STOPPED:
            raise RuntimeError("OCR service is stopping")
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env or _env(),
            start_new_session=True,
        )
        _RUNNING.add(proc.pid)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        proc.communicate()
        raise TimeoutError("OCR process exceeded its time limit") from None
    finally:
        with _PROCESS_LOCK:
            _RUNNING.discard(proc.pid)
    if proc.returncode:
        # Do not expose engine stderr, which may contain local paths or content.
        raise RuntimeError(f"OCR process failed (exit {proc.returncode})")
    return subprocess.CompletedProcess(argv, proc.returncode, out, b"")


def _flags(text: str, language: str | list[str] | None) -> list[str]:
    flags: list[str] = []
    if len(text) >= OUTPUT_CHAR_LIMIT:
        flags.append("output_truncated")
    letters = [c for c in text if c.isalpha()]
    hints = language if isinstance(language, list) else [language] if language else []
    ranges = {
        _SCRIPT_RANGES[hint.casefold()]
        for hint in hints
        if hint.casefold() in _SCRIPT_RANGES
    }
    if len(letters) > 100 and ranges:
        script_chars = sum(
            1 for c in letters if any(low <= ord(c) <= high for low, high in ranges)
        )
        if script_chars / len(letters) < 0.02:
            flags.append("script_mismatch")
    tokens = re.findall(r"\S+", text.casefold())
    if len(tokens) >= 30:
        most = max(Counter(tokens).values(), default=0)
        if most / len(tokens) >= 0.35:
            flags.append("repetition_suspected")
    lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3 and max(Counter(lines).values(), default=0) >= 3:
        flags.append("repeated_lines_suspected")
    if (
        re.search(r"list of the equations.*algebra diagram", text, re.I | re.S)
        or re.search(r"x\s*\^?2\s*\+\s*10\s*x\s*\+\s*20\s*=\s*0", text, re.I)
        and len(re.findall(r"x\s*\^?2\s*\+\s*10\s*x\s*\+\s*20\s*=\s*0", text, re.I))
        >= 2
    ):
        flags.append("known_boilerplate_suspected")
    if not text.strip():
        flags.append("empty_output")
    return flags


def _text_result(
    text: str,
    *,
    pipeline: str,
    model: str,
    version: str | None,
    language: str | list[str] | None,
    fmt: str = "text",
    metadata: dict | None = None,
) -> dict:
    text = text[:OUTPUT_CHAR_LIMIT]
    return {
        "text": text,
        "format": fmt,
        "flags": _flags(text, language),
        "model": model,
        "version": version,
        "pipeline": pipeline,
        **({"metadata": metadata} if metadata else {}),
    }


def _tesseract(path: Path, pipeline: str, language: str | list[str] | None) -> dict:
    binary = shutil.which("tesseract")
    assert binary
    tessdata = MODEL_ROOT / "tesseract-ocr" / "tessdata"
    hints = language if isinstance(language, list) else [language or "hi"]
    if any(hint not in _LANG for hint in hints):
        raise ValueError("Unsupported language for this OCR pipeline")
    trained_models = list(dict.fromkeys(_LANG[hint][0] for hint in hints))
    if pipeline == "anuvaad":
        if trained_models != ["hin"]:
            raise ValueError("Anuvaad supports Hindi")
        tessdata = MODEL_ROOT / "project-anuvaad" / "anuvaad-ocr-model"
        trained_models = ["anuvaad_hin"]
    elif not isinstance(language, list) and trained_models != ["eng"]:
        trained_models.insert(0, "eng")
    trained = "+".join(trained_models)
    if not all(
        (tessdata / f"{name}.traineddata").is_file() for name in trained.split("+")
    ):
        raise RuntimeError("Requested Tesseract language data is unavailable")
    completed = _run(
        [binary, str(path), "stdout", "-l", trained, "--tessdata-dir", str(tessdata)],
        timeout=execution_timeout_seconds(pipeline),
    )
    return _text_result(
        completed.stdout.decode("utf-8", "replace"),
        pipeline=pipeline,
        model="Anuvaad Hindi Tesseract"
        if pipeline == "anuvaad"
        else "Tesseract " + trained,
        version=_version(binary),
        language=language,
        metadata={"pages": 1, "device": "cpu"},
    )


def _docling(path: Path, language: str | list[str] | None = None) -> dict:
    script = Path(__file__).with_name("docling_ocr.py")
    with tempfile.TemporaryDirectory(prefix="ocr-docling-") as work:
        output = Path(work) / "result.md"
        command = [sys.executable, str(script), str(path), "--output", str(output)]
        if isinstance(language, list):
            trained_models = dict.fromkeys(_LANG[hint][0] for hint in language)
            for trained in trained_models:
                command.extend(["--ocr-language", trained])
        completed = _run(
            command,
            timeout=execution_timeout_seconds("docling"),
        )
        try:
            metadata = json.loads(completed.stdout.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError(
                "Docling returned an invalid extraction summary"
            ) from exc
        if not isinstance(metadata, dict):
            raise RuntimeError("Docling returned an invalid extraction summary")
        text = output.read_text(encoding="utf-8")
    return _text_result(
        text,
        pipeline="docling",
        model="Docling PDF default layout / region OCR",
        version="2.133.0",
        language=language or "hi",
        fmt="markdown",
        metadata={
            k: v
            for k, v in metadata.items()
            if k
            in {
                "status",
                "pages",
                "chars",
                "picture_text_pages",
                "page_image_pages",
                "ocr_languages",
                "device",
            }
        },
    )


def _indic(path: Path, pipeline: str, language: str | list[str] | None = None) -> dict:
    from PIL import Image

    sys.path.insert(0, str(MODEL_ROOT / "bodhan-ai" / "indic-ocr"))
    device = torch_device(device_mode())
    if device is None:
        raise RuntimeError("Configured PyTorch device is unavailable")
    if pipeline == "indicocr_block":
        from idp_contract import prompt_for
        from idp_recognizer import CropRequest, HfRecognizer
        from idp_types import RecognizerConfig

        model_path = MODEL_ROOT / "bodhan-ai" / "indic-ocr" / "weights" / "ocr"
        recognizer = HfRecognizer(
            ckpt=str(model_path),
            config=RecognizerConfig(max_tokens=2048),
            device=device,
            batch_size=1,
        )
        with Image.open(path) as image:
            result = recognizer.transcribe(
                [CropRequest(image.convert("RGB"), prompt_for("Text"))]
            )[0]
        text = getattr(result, "text", str(result))
    else:
        from indic_ocr import IndicOCR

        model_path = MODEL_ROOT / "bodhan-ai" / "indic-ocr"
        recognizer = IndicOCR.from_pretrained(
            str(model_path), device=device, max_new_tokens=2048
        )
        value = recognizer.parse(str(path))
        text = value.get("markdown", "") if isinstance(value, dict) else str(value)
    return _text_result(
        text.replace("▁", " ").strip(),
        pipeline=pipeline,
        model="Bodhan IndicOCR "
        + ("block" if pipeline == "indicocr_block" else "layout"),
        version="local checkpoint",
        language=language,
        fmt="markdown" if pipeline == "indicocr_layout" else "text",
        metadata={
            "device": device,
            "checkpoint": "bodhan-ai/indic-ocr/weights/ocr"
            if pipeline == "indicocr_block"
            else "bodhan-ai/indic-ocr",
        },
    )


def _worker(path: Path, pipeline: str, language: str | list[str] | None) -> dict:
    worker = Path(__file__).with_name("ocr_pipeline_worker.py")
    python = (
        _benchmark_python()
        if pipeline in {"surya25", "easyocr", "ppocrv5"}
        else sys.executable
    )
    completed = _run(
        [str(python), str(worker), pipeline, str(path), json.dumps(language)],
        timeout=execution_timeout_seconds(pipeline),
        env=_env(),
    )
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("OCR worker returned an invalid response") from exc
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise RuntimeError("OCR worker returned an invalid response")
    return value


def _gguf(path: Path, pipeline: str, language: str | list[str] | None) -> dict:
    model, mmproj, label = _GGUF[pipeline]
    binary = shutil.which("llama-mtmd-cli")
    device = llama_device(binary, device_mode())
    if not binary or device is None:
        raise RuntimeError("Configured llama.cpp device is unavailable")
    prompts = {
        "chandra2": _CHANDRA_LAYOUT_PROMPT,
        "lightonocr": "grounding",
        "paddleocr_vl": "OCR:",
        "glm_ocr": "Text Recognition:",
    }
    result = _run(
        [
            binary,
            "-m",
            str(MODEL_ROOT / model),
            "--mmproj",
            str(MODEL_ROOT / mmproj),
            "--image",
            str(path),
            "--jinja",
            "-p",
            prompts[pipeline],
            *llama_args(device),
            "-t",
            "2",
            "-tb",
            "2",
            "-c",
            "8192",
            "-n",
            "2048",
            "--temp",
            "0",
            "--top-p",
            "1",
            "--no-warmup",
            "--perf",
        ],
        timeout=execution_timeout_seconds(pipeline),
    )
    text = result.stdout.decode("utf-8", "replace").strip()
    if pipeline == "lightonocr":
        text = re.sub(
            r"!\[[^]]+\]\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", "", text
        )
    return _text_result(
        text,
        pipeline=pipeline,
        model=label,
        version=Path(model).name,
        language=language,
        fmt="html" if pipeline == "chandra2" else "text",
        metadata={"device": device, "projector_file": Path(mmproj).name},
    )


def run_pipeline(
    pipeline: str, source: Path, language: str | list[str] | None = None
) -> dict[str, Any]:
    """Run one locally available engine on a bounded image or PDF."""
    path = _validate(Path(source), pipeline)
    if pipeline == "docling":
        return _docling(path, language)
    from PIL import Image, ImageOps

    with Image.open(path) as original:
        width, height = original.size
        scale = bounded_raster_scale(width, height)
        if scale < 1:
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            try:
                original.thumbnail(size, Image.Resampling.LANCZOS)
                image = ImageOps.exif_transpose(original)
                if image.mode not in {"L", "RGB", "RGBA"}:
                    image = image.convert("RGB")
            except (OSError, SyntaxError) as exc:
                raise InputError("invalid_image") from exc
            with tempfile.NamedTemporaryFile(
                prefix="job-resized-", suffix=".png", dir=path.parent
            ) as resized:
                image.save(resized, format="PNG")
                resized.flush()
                result = _run_image(pipeline, Path(resized.name), language)
            result.setdefault("metadata", {})["image_resize"] = {
                "original_pixels": [width, height],
                "processed_pixels": list(image.size),
            }
            return result
    return _run_image(pipeline, path, language)


def _run_image(
    pipeline: str, path: Path, language: str | list[str] | None
) -> dict[str, Any]:
    if pipeline in {"tesseract", "anuvaad"}:
        return _tesseract(path, pipeline, language)
    if pipeline.startswith("indicocr_") or pipeline in {
        "surya25",
        "easyocr",
        "ppocrv5",
    }:
        return _worker(path, pipeline, language)
    if pipeline in _GGUF:
        return _gguf(path, pipeline, language)
    raise RuntimeError(f"OCR adapter not implemented for: {pipeline}")
