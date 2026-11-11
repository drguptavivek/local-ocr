"""Isolated inference workers for heavyweight local OCR runtimes."""

from __future__ import annotations

import contextlib
import io
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path


def _limit_resources(device: str | None) -> None:
    limits = [(resource.RLIMIT_CPU, 900), (resource.RLIMIT_FSIZE, 512 * 1024**2)]
    # CUDA can reserve more virtual address space than its physical memory use.
    if (
        not (device or "").lower().startswith("cuda")
        and sys.platform != "darwin"
        and hasattr(resource, "RLIMIT_AS")
    ):
        limits.append((resource.RLIMIT_AS, 12 * 1024**3))
    for kind, requested in limits:
        soft, hard = resource.getrlimit(kind)
        cap = requested if hard == resource.RLIM_INFINITY else min(hard, requested)
        resource.setrlimit(
            kind, (cap if soft == resource.RLIM_INFINITY else min(soft, cap), cap)
        )


def _indic(path: Path, pipeline: str, language: str | list[str] | None) -> dict:
    from ocr_pipelines import _indic

    return _indic(path, pipeline, language)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _surya(path: Path, language: str | list[str] | None) -> dict:
    from PIL import Image, ImageOps
    from ocr_device import device_mode, llama_args, llama_device
    from ocr_pipelines import _flags, _surya_source

    source = _surya_source()
    if not (source / "surya" / "__init__.py").is_file():
        raise RuntimeError("pinned local Surya SDK is unavailable")
    sys.path.insert(0, str(source))

    model_dir = Path.home() / ".models" / "ocr" / "datalab-to" / "surya-ocr-2.5-gguf"
    model = model_dir / "surya-2.5-Q8_0.gguf"
    mmproj = model_dir / "surya-2.5-mmproj.gguf"
    binary = shutil.which("llama-server")
    device = llama_device(binary, device_mode())
    if not binary or device is None or not model.is_file() or not mmproj.is_file():
        raise RuntimeError("local Surya GGUF runtime is unavailable")

    _limit_resources(device)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(Path.home() / ".models" / "huggingface"),
            "SURYA_MODEL_CHECKPOINT": "datalab-to/surya-ocr-2.5",
            "SURYA_GGUF_LOCAL_MODEL_PATH": str(model),
            "SURYA_GGUF_LOCAL_MMPROJ_PATH": str(mmproj),
            "SURYA_INFERENCE_BACKEND": "llamacpp",
            "SURYA_INFERENCE_URL": f"http://127.0.0.1:{port}/v1",
            "SURYA_INFERENCE_AUTOSTART": "false",
            "SURYA_INFERENCE_KEEP_ALIVE": "false",
            "SURYA_INFERENCE_TIMEOUT_SECONDS": "600" if device == "cpu" else "120",
            "SURYA_INFERENCE_STARTUP_TIMEOUT": "120",
            "SURYA_INFERENCE_PARALLEL": "1",
            "SURYA_INFERENCE_CTX_SIZE": "16384",
            "SURYA_INFERENCE_CTX_PER_SLOT": "8192",
            "SURYA_MAX_TOKENS_FULL_PAGE": "2500",
            "SURYA_MAX_TOKENS_BLOCK_CEILING": "2048",
            "SURYA_GUIDED_LAYOUT": "true",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
        }
    )
    os.environ.update(env)
    argv = [
        binary,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--alias",
        "datalab-to/surya-ocr-2.5",
        "--ctx-size",
        "16384",
        "--parallel",
        "1",
        *llama_args(device),
        "-t",
        "2",
        "-tb",
        "2",
        "--no-warmup",
        "--special",
    ]
    server = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    manager = None
    try:
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            manager = SuryaInferenceManager()
            manager.start()
            predictor = RecognitionPredictor(manager)
            predictor.disable_tqdm = True
        with Image.open(path) as incoming:
            image = ImageOps.exif_transpose(incoming).convert("RGB")
        image.thumbnail((1700, 1700), Image.Resampling.LANCZOS)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            (result,) = predictor([image])
        errors = any(getattr(block, "error", False) for block in result.blocks)
        blocks = sorted(
            result.blocks, key=lambda block: getattr(block, "reading_order", 0)
        )
        parts: list[str] = []
        for block in blocks:
            parser = _TextExtractor()
            parser.feed(getattr(block, "html", "") or "")
            text = " ".join(parser.parts)
            if text:
                parts.append(text)
        text = "\n".join(parts)
        flags = _flags(text, language)
        if errors:
            flags.append("recognition_block_error")
        return {
            "text": text[:2_000_000],
            "format": "text",
            "flags": flags,
            "model": "Surya OCR 2.5 Q8_0",
            "version": "0.23.0",
            "pipeline": "surya25",
            "metadata": {"blocks": len(blocks), "device": device},
        }
    finally:
        try:
            if manager is not None:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    manager.stop()
        finally:
            try:
                server.terminate()
                server.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    server.kill()
                except ProcessLookupError:
                    pass
                server.wait()


def _printed_image(path: Path, pipeline: str, language: str | list[str] | None) -> dict:
    from importlib.metadata import version
    from ocr_pipelines import MODEL_ROOT, _text_result

    if pipeline == "easyocr":
        import easyocr
        import torch

        torch.set_num_threads(2)
        reader = easyocr.Reader(
            ["hi", "en"],
            gpu=False,
            quantize=False,
            download_enabled=False,
            model_storage_directory=str(MODEL_ROOT / "JaidedAI/EasyOCR"),
            verbose=False,
        )
        lines = reader.readtext(str(path), detail=0, workers=0, batch_size=1)
        model = "EasyOCR Devanagari FP32"
        revision = version("easyocr")
    else:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR

        models = MODEL_ROOT / "PaddlePaddle"
        reader = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_detection_model_dir=str(models / "PP-OCRv5_server_det"),
            text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
            text_recognition_model_dir=str(models / "devanagari_PP-OCRv5_mobile_rec"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
            cpu_threads=1,
        )
        lines = next(iter(reader.predict(str(path))))["rec_texts"]
        model = "PP-OCRv5 Devanagari"
        revision = f"{version('paddleocr')} / PaddlePaddle {version('paddlepaddle')}"
    return _text_result(
        "\n".join(lines),
        pipeline=pipeline,
        model=model,
        version=revision,
        language=language,
        metadata={"device": "cpu"},
    )


def main() -> int:
    if len(sys.argv) != 4:
        return 2
    pipeline, source, language = sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3])
    try:
        if pipeline == "surya25":
            result = _surya(source, language)
        elif pipeline.startswith("indicocr_"):
            from ocr_device import device_mode, torch_device

            _limit_resources(torch_device(device_mode()))
            result = _indic(source, pipeline, language)
        elif pipeline in {"easyocr", "ppocrv5"}:
            _limit_resources("cpu")
            with (
                open(os.devnull, "w") as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                result = _printed_image(source, pipeline, language)
        else:
            return 2
        json.dump(result, sys.stdout, ensure_ascii=False)
        return 0
    except Exception as exc:
        # The exception message may include source text or private filesystem paths.
        print(json.dumps({"error": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
