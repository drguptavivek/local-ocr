#!/usr/bin/env python
"""Run Docling's PDF-aware layout extraction and Hindi+English OCR for one PDF."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

if __package__:
    from .ocr_device import device_mode, torch_device
    from .ocr_limits import bounded_raster_scale
else:
    from ocr_device import device_mode, torch_device
    from ocr_limits import bounded_raster_scale

for _name in (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

_MODEL_ROOT = Path.home() / ".models"
os.environ["HF_HOME"] = str(_MODEL_ROOT / "huggingface")
os.environ["DOCLING_CACHE_DIR"] = str(_MODEL_ROOT / "docling")
os.environ["TORCH_HOME"] = str(_MODEL_ROOT / "torch")


def build_pipeline_options(
    full_page: bool = False, languages: list[str] | None = None, *, device: str
):
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.pipeline_options import (
        OcrMode,
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.accelerator_options.device = AcceleratorDevice(device)
    options.accelerator_options.num_threads = 1
    ocr = TesseractCliOcrOptions(lang=languages or ["script/Devanagari", "eng"])
    # FULL_PAGE only for legacy-font PDFs whose embedded text layer is garbled
    # (tools/reocr-legacy-font.py); everything else keeps the PDF-aware mode.
    ocr.mode = OcrMode.FULL_PAGE if full_page else OcrMode.DEFAULT
    options.ocr_options = ocr
    return options


def _cap_resource_limit(resource_type: int, requested: int) -> None:
    current_soft, current_hard = resource.getrlimit(resource_type)
    hard_limit = (
        requested
        if current_hard == resource.RLIM_INFINITY
        else min(current_hard, requested)
    )
    soft_limit = (
        hard_limit
        if current_soft == resource.RLIM_INFINITY
        else min(current_soft, hard_limit)
    )
    resource.setrlimit(resource_type, (soft_limit, hard_limit))


def apply_resource_limits(device: str) -> None:
    _cap_resource_limit(resource.RLIMIT_CPU, 1200)
    _cap_resource_limit(resource.RLIMIT_FSIZE, 512 * 1024 * 1024)
    # CUDA can reserve more virtual address space than its physical memory use.
    if device != "cuda" and sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
        _cap_resource_limit(resource.RLIMIT_AS, 12 * 1024 * 1024 * 1024)


IMAGE_PLACEHOLDER = "<!-- image -->"


def _readable(markdown: str) -> bool:
    return bool(markdown.replace(IMAGE_PLACEHOLDER, "").strip())


def export_markdown(
    document, page_image_markdown=None
) -> tuple[str, dict[str, list[int]]]:
    """Markdown, plus which pages needed a fallback.

    Docling OCRs scanned pages in the default (PDF-aware) mode, but when the
    layout model labels a whole scan as a picture the normal export yields only
    a placeholder. For such pages only: (1) export with the OCR text inside the
    pictures; (2) if still empty, OCR the page rendered as an image
    (`page_image_markdown(page_no)`). Output for other documents is unchanged.
    """
    markdown = document.export_to_markdown()
    parts: list[str] = []
    fallbacks: dict[str, list[int]] = {"picture_text": [], "page_image": []}
    for page_no in sorted(document.pages):
        page_md = document.export_to_markdown(page_no=page_no)
        if not _readable(page_md):
            with_pictures = document.export_to_markdown(
                page_no=page_no, traverse_pictures=True
            )
            if _readable(with_pictures):
                page_md = with_pictures
                fallbacks["picture_text"].append(int(page_no))
            elif page_image_markdown is not None:
                from_image = page_image_markdown(int(page_no))
                if _readable(from_image):
                    page_md = from_image
                    fallbacks["page_image"].append(int(page_no))
        parts.append(page_md)
    if fallbacks["picture_text"] or fallbacks["page_image"]:
        markdown = "\n\n".join(part for part in parts if part.strip())
    return markdown, fallbacks


PAGE_IMAGE_DPI = 300


def page_image_ocr(pdf: Path, workdir: Path, pipeline_options):
    """Return page_no -> Markdown from Docling run on the page rendered as PNG."""
    import pypdfium2 as pdfium
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, ImageFormatOption

    converter = None

    def markdown_for(page_no: int) -> str:
        nonlocal converter
        if converter is None:
            converter = DocumentConverter(
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        document = pdfium.PdfDocument(str(pdf))
        try:
            page = document[page_no - 1]
            width, height = page.get_size()
            scale = bounded_raster_scale(width, height, PAGE_IMAGE_DPI / 72)
            image = page.render(scale=scale).to_pil()
        finally:
            document.close()
        path = workdir / f"page-{page_no}.png"
        image.save(path)
        try:
            result = converter.convert(path)
            return result.document.export_to_markdown(traverse_pictures=True)
        finally:
            path.unlink(missing_ok=True)

    return markdown_for


def summarize_result(result, markdown: str) -> dict:
    page_stats = {
        int(page_no): {
            "page": int(page_no),
            "status": "no_text_or_table",
            "text_items": 0,
            "table_items": 0,
            "text_chars": 0,
            "errors": 0,
        }
        for page_no in result.document.pages
    }
    for item in result.document.texts:
        provenances = getattr(item, "prov", ()) or ()
        text = getattr(item, "text", "")
        for provenance in provenances:
            page_no = int(provenance.page_no)
            stats = page_stats.get(page_no)
            if stats is None:
                continue
            stats["text_items"] += 1
            charspan = getattr(provenance, "charspan", None)
            if charspan and len(charspan) == 2:
                stats["text_chars"] += max(0, int(charspan[1]) - int(charspan[0]))
            elif len(provenances) == 1:
                stats["text_chars"] += len(text)

    for table in result.document.tables:
        for provenance in getattr(table, "prov", ()) or ():
            page_no = int(provenance.page_no)
            stats = page_stats.get(page_no)
            if stats is not None:
                stats["table_items"] += 1

    for error in result.errors:
        page_no = getattr(error, "page_no", None)
        if page_no is not None and int(page_no) in page_stats:
            page_stats[int(page_no)]["errors"] += 1

    for stats in page_stats.values():
        if stats["errors"]:
            stats["status"] = "warning"
        elif stats["text_items"] and stats["table_items"]:
            stats["status"] = "text+table"
        elif stats["text_items"]:
            stats["status"] = "text"
        elif stats["table_items"]:
            stats["status"] = "table"

    status = getattr(result.status, "value", str(result.status))
    return {
        "status": status,
        "pages": len(page_stats),
        "chars": len(markdown),
        "page_details": [page_stats[page_no] for page_no in sorted(page_stats)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-page", action="store_true")
    parser.add_argument("--ocr-language", action="append")
    args = parser.parse_args()

    pdf = Path(args.pdf).resolve(strict=True)
    output = Path(args.output)
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        parser.error("input must be a regular PDF file")
    if output.resolve() == pdf:
        parser.error("output must not overwrite the input PDF")

    device = torch_device(device_mode())
    if device is None:
        parser.error("Configured PyTorch device is unavailable")
    apply_resource_limits(device)
    # The API parses stdout as the JSON summary; Docling's loggers also write
    # to stdout, so route everything except the summary to stderr.
    summary_stream, sys.stdout = sys.stdout, sys.stderr

    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = build_pipeline_options(args.full_page, args.ocr_language, device=device)
    # Keep Docling's region OCR in PDF coordinates while bounding its raster scale.
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(str(pdf)) as document:
        opts.ocr_options.scale = min(
            bounded_raster_scale(*document.get_page_size(index), opts.ocr_options.scale)
            for index in range(len(document))
        )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts, backend=PyPdfiumDocumentBackend
            )
        }
    )
    started = time.time()
    result = converter.convert(pdf)
    with tempfile.TemporaryDirectory() as workdir:
        markdown, fallbacks = export_markdown(
            result.document, page_image_ocr(pdf, Path(workdir), opts)
        )
    output.write_text(markdown, encoding="utf-8")
    summary = summarize_result(result, markdown)
    summary["picture_text_pages"] = fallbacks["picture_text"]
    summary["page_image_pages"] = fallbacks["page_image"]
    summary["device"] = device
    summary["ocr_languages"] = opts.ocr_options.lang
    summary["filename"] = pdf.name
    summary["seconds"] = round(time.time() - started, 1)
    json.dump(summary, summary_stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
