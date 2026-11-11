"""Input bounds and safe API messages shared by the OCR adapters."""

from math import sqrt

MAX_PAGES = 20
MAX_PDF_PAGE_POINTS = 36 * 72
MAX_IMAGE_SIDE = 12_000
MAX_IMAGE_PIXELS = 50_000_000
RASTER_MAX_SIDE = 4096
RASTER_MAX_PIXELS = 12_000_000

ERROR_DETAILS = {
    "invalid_image": "The file could not be read as a JPEG, PNG, or TIFF image.",
    "image_too_large": "The image exceeds 12,000 pixels per side or 50 megapixels. Resize it and submit a new job.",
    "invalid_pdf": "The file could not be read as a PDF.",
    "pdf_encrypted": "The PDF requires a password. Submit an unlocked copy.",
    "pdf_page_limit": "The PDF must contain 1 to 20 pages. Split larger PDFs before submitting.",
    "pdf_page_too_large": "A PDF page exceeds 36 inches on one side. Resize or split that page before submitting.",
    "pipeline_unavailable": "The selected OCR pipeline is unavailable on this host.",
    "pipeline_timeout": "The OCR pipeline exceeded its execution time limit.",
    "storage_quota": "The service has insufficient job storage quota for the result.",
    "pipeline_failed": "The OCR engine could not process the input.",
}


class InputError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(ERROR_DETAILS[code])


def bounded_raster_scale(width: float, height: float, requested: float = 1.0) -> float:
    return min(
        requested,
        RASTER_MAX_SIDE / max(width, height),
        sqrt(RASTER_MAX_PIXELS / (width * height)),
    )
