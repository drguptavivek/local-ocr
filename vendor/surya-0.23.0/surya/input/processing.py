from typing import List

import pypdfium2

from surya.settings import settings


def open_pdf(pdf_filepath):
    return pypdfium2.PdfDocument(pdf_filepath)


def get_page_images(doc, indices: List, dpi=settings.IMAGE_DPI):
    images = [
        doc[i].render(scale=dpi / 72, draw_annots=False).to_pil() for i in indices
    ]
    images = [image.convert("RGB") for image in images]
    return images
