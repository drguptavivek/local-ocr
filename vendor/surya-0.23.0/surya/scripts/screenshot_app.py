"""Screenshot-friendly Surya viewer.

Shows a PDF/image page on the left and full-page OCR output on the right, side
by side, for clean screenshots. You can scroll through pages and preview them
before running OCR, then export the side-by-side view as a PNG.

Run with `surya_screenshot`, then open http://localhost:8504.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import uuid
from typing import List, Optional

import pypdfium2
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

from surya.inference import SuryaInferenceManager
from surya.logging import configure_logging, get_logger
from surya.recognition import RecognitionPredictor
from surya.recognition.schema import PageOCRResult
from surya.settings import settings

configure_logging()
logger = get_logger()

app = Flask(__name__)

ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "surya_screenshot")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_rec: Optional[RecognitionPredictor] = None


def get_rec() -> RecognitionPredictor:
    """Lazily build the recognition predictor (shared inference manager)."""
    global _rec
    if _rec is None:
        _rec = RecognitionPredictor(SuryaInferenceManager())
    return _rec


# Datalab-flavored palette for layout block overlays, keyed by canonical label.
LABEL_COLORS = {
    "Text": "#2563eb",
    "SectionHeader": "#0ea5e9",
    "PageHeader": "#7c3aed",
    "PageFooter": "#7c3aed",
    "Caption": "#c026d3",
    "Footnote": "#64748b",
    "Equation": "#9333ea",
    "Table": "#f59e0b",
    "TableOfContents": "#f59e0b",
    "Form": "#ea580c",
    "ListGroup": "#10b981",
    "Picture": "#db2777",
    "Figure": "#db2777",
    "Diagram": "#db2777",
    "Code": "#0d9488",
    "default": "#ef4444",
}


def _logo_data_url() -> str:
    path = os.path.join(settings.BASE_DIR, "static", "datalab-logo.png")
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return (
        f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()
    )


def _is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def _page_count(path: str) -> int:
    if _is_pdf(path):
        doc = pypdfium2.PdfDocument(path)
        n = len(doc)
        doc.close()
        return n
    return 1


def _render_page(path: str, page: int, dpi: int) -> Image.Image:
    """Render a 0-indexed page of a PDF (or load an image file) as RGB."""
    if _is_pdf(path):
        doc = pypdfium2.PdfDocument(path)
        try:
            pil = doc[page].render(scale=dpi / 72).to_pil().convert("RGB")
        finally:
            doc.close()
        return pil
    return Image.open(path).convert("RGB")


def _assemble_page_html(page: PageOCRResult) -> str:
    """Whole-page HTML from a PageOCRResult (math stays in <math> tags)."""
    parts: List[str] = []
    for blk in page.blocks:
        if blk.skipped:
            continue
        x0, y0, x1, y1 = (int(c) for c in blk.bbox)
        parts.append(
            f'<div data-bbox="{x0} {y0} {x1} {y1}" '
            f'data-label="{blk.label}">{blk.html or ""}</div>'
        )
    return "\n".join(parts)


@app.route("/")
def index():
    return render_template("surya_screenshot.html", logo=_logo_data_url())


@app.route("/info", methods=["POST"])
def info():
    path = (request.json or {}).get("file_path", "").strip()
    if not path:
        return jsonify({"error": "file_path is required"}), 400
    if not os.path.exists(path):
        return jsonify({"error": f"File not found: {path}"}), 400
    try:
        return jsonify({"page_count": _page_count(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a drag/drop (or browsed) file, save to a temp path, return it."""
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file uploaded"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"unsupported file type: {ext or '(none)'}"}), 400
    safe = secure_filename(f.filename) or f"upload{ext}"
    dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{safe}")
    f.save(dest)
    try:
        return jsonify(
            {"file_path": dest, "page_count": _page_count(dest), "name": f.filename}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/page", methods=["POST"])
def page():
    """Render a single page for preview (no OCR)."""
    data = request.json or {}
    path = data.get("file_path", "").strip()
    page_num = int(data.get("page", 0))
    if not path or not os.path.exists(path):
        return jsonify({"error": "valid file_path is required"}), 400
    try:
        img = _render_page(path, page_num, settings.IMAGE_DPI_HIGHRES)
        return jsonify(
            {
                "image_base64": _pil_to_data_url(img),
                "width": img.size[0],
                "height": img.size[1],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process", methods=["POST"])
def process():
    """Run full-page OCR on one page; return the page image + OCR HTML + blocks."""
    data = request.json or {}
    path = data.get("file_path", "").strip()
    page_num = int(data.get("page", 0))
    if not path or not os.path.exists(path):
        return jsonify({"error": "valid file_path is required"}), 400
    try:
        img = _render_page(path, page_num, settings.IMAGE_DPI_HIGHRES)
        page_result = get_rec()([img], full_page=True)[0]
        blocks = [
            {
                "bbox": [int(c) for c in blk.bbox],
                "label": blk.label,
                "color": LABEL_COLORS.get(blk.label, LABEL_COLORS["default"]),
            }
            for blk in page_result.blocks
            if not blk.skipped
        ]
        return jsonify(
            {
                "image_base64": _pil_to_data_url(img),
                "width": img.size[0],
                "height": img.size[1],
                "html": _assemble_page_html(page_result),
                "blocks": blocks,
                "n_blocks": len(page_result.blocks),
            }
        )
    except Exception as e:
        logger.exception("Full-page OCR failed")
        return jsonify({"error": str(e)}), 500


def main():
    app.run(host="0.0.0.0", port=8504)


if __name__ == "__main__":
    main()
