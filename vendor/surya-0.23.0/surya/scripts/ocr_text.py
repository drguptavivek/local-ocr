import os
import click
import json
import time
from collections import defaultdict

from surya.inference import SuryaInferenceManager
from surya.logging import configure_logging, get_logger
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader

configure_logging()
logger = get_logger()


@click.command(help="OCR text — full-page OCR (one VLM call per page).")
@CLILoader.common_options
def ocr_text_cli(input_path: str, **kwargs):
    # Full-page OCR is the default path: one VLM call per page returns layout
    # + content together. Pages whose full-page output fails to parse fall
    # back to layout + per-block OCR automatically (see RecognitionPredictor).
    loader = CLILoader(input_path, kwargs, highres=True)

    manager = SuryaInferenceManager()
    rec_predictor = RecognitionPredictor(manager)

    start = time.time()
    page_results = rec_predictor(loader.highres_images, full_page=True)

    if loader.debug:
        logger.debug(f"OCR took {time.time() - start:.2f} seconds")

    out_preds = defaultdict(list)
    for name, page in zip(loader.names, page_results):
        out_pred = page.model_dump()
        out_pred["page"] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)

    with open(
        os.path.join(loader.result_path, "results.json"), "w+", encoding="utf-8"
    ) as f:
        json.dump(out_preds, f, ensure_ascii=False)

    logger.info(f"Wrote results to {loader.result_path}")
