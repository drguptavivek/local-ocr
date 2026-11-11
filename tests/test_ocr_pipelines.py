"""Focused offline checks for OCR adapter contracts and safety bounds."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import ocr_device  # noqa: E402
import ocr_pipelines  # noqa: E402


class OcrPipelineTests(unittest.TestCase):
    def test_catalog_has_contract_fields_for_every_candidate(self):
        for entry in ocr_pipelines.pipeline_catalog().values():
            self.assertIsInstance(entry["available"], bool)
            self.assertIn(entry["execution_timeout_seconds"], {180, 900})
            self.assertTrue(all(ext.startswith(".") for ext in entry["extensions"]))

    def test_explicit_cpu_disables_model_and_projector_offload(self):
        with patch.object(ocr_device.subprocess, "run") as run:
            device = ocr_device.llama_device("cpu-test-llama", "cpu")
            self.assertEqual(ocr_device.probe_torch_device("cpu"), "cpu")
            run.assert_not_called()
        self.assertEqual(device, "cpu")
        self.assertEqual(
            ocr_device.llama_args(device),
            ["--device", "none", "-ngl", "0", "--no-mmproj-offload"],
        )

    def test_llama_selects_requested_cuda_and_reports_missing_gpu(self):
        from subprocess import CompletedProcess

        output = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4070 (12282 MiB)\n"
        with patch.object(
            ocr_device.subprocess,
            "run",
            return_value=CompletedProcess([], 0, output, ""),
        ):
            self.assertEqual(
                ocr_device.llama_device("cuda-test-llama", "cuda"), "CUDA0"
            )
            self.assertIsNone(ocr_device.llama_device("cuda-test-llama", "mps"))
        with patch.dict("os.environ", {"OCR_GPU_LAYERS": "20"}):
            self.assertEqual(
                ocr_device.llama_args("CUDA0"), ["--device", "CUDA0", "-ngl", "20"]
            )

    def test_unavailable_accelerator_keeps_cpu_baselines(self):
        with (
            patch.dict("os.environ", {"OCR_DEVICE": "cuda"}),
            patch.object(ocr_pipelines, "probe_torch_device", return_value=None),
            patch.object(ocr_pipelines, "llama_device", return_value=None),
        ):
            catalog = ocr_pipelines.pipeline_catalog()
        for pipeline in ("docling", "indicocr_layout", "surya25", "chandra2"):
            self.assertFalse(catalog[pipeline]["available"])
            self.assertIsNone(catalog[pipeline]["device"])
        for pipeline in ("tesseract", "anuvaad", "easyocr", "ppocrv5"):
            self.assertEqual(catalog[pipeline]["device"], "cpu")

    def test_paddle_cpu_uses_prepared_projector_and_reports_missing_file(self):
        with (
            patch.dict("os.environ", {"OCR_DEVICE": "cpu"}),
            patch.object(ocr_pipelines.shutil, "which", return_value="/test/llama"),
            patch.object(
                ocr_pipelines,
                "_file",
                side_effect=lambda path: not path.name.endswith("-q8_0.gguf"),
            ),
        ):
            paddle = ocr_pipelines.pipeline_catalog()["paddleocr_vl"]
        self.assertFalse(paddle["available"])
        self.assertIn("prepare-paddle-projector.py", paddle["unavailable_reason"])
        cpu = ocr_pipelines._gguf_projector("paddleocr_vl", "cpu")
        cuda = ocr_pipelines._gguf_projector("paddleocr_vl", "CUDA0")
        self.assertEqual(cpu.name, f"{cuda.stem}-q8_0.gguf")

    def test_tesseract_catalog_discovers_installed_codes_and_keeps_aliases(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            packs = root / "tesseract-ocr" / "tessdata"
            packs.mkdir(parents=True)
            for code in ("eng", "ben", "chi_sim", "osd"):
                (packs / f"{code}.traineddata").touch()
            with patch.object(ocr_pipelines, "MODEL_ROOT", root):
                hints = ocr_pipelines.pipeline_catalog()["tesseract"]["languages"]
            self.assertEqual(hints, ["ben", "chi_sim", "en", "eng", "english"])

    def test_process_timeout_surfaces_as_timeout_error(self):
        with self.assertRaises(TimeoutError):
            ocr_pipelines._run(
                [sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.1
            )

    def test_catalog_does_not_advertise_missing_runtime(self):
        with (
            patch.object(ocr_pipelines, "_module", return_value=False),
            patch.object(ocr_pipelines.shutil, "which", return_value=None),
        ):
            catalog = ocr_pipelines.pipeline_catalog()
        self.assertFalse(catalog["docling"]["available"])
        self.assertFalse(catalog["indicocr_block"]["available"])
        self.assertFalse(catalog["chandra2"]["available"])

    def test_unknown_unavailable_and_bad_extension_rejected(self):
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "file.txt"
            source.write_text("not an image")
            with self.assertRaises(ValueError):
                ocr_pipelines.run_pipeline("missing", source)
            with patch.object(
                ocr_pipelines,
                "pipeline_catalog",
                return_value={
                    "tesseract": {"available": False, "extensions": [".png"]}
                },
            ):
                with self.assertRaises(ValueError):
                    ocr_pipelines.run_pipeline("tesseract", source)
                image = Path(work) / "file.png"
                image.write_bytes(b"fake")
                with self.assertRaises(ocr_pipelines.InputError) as error:
                    ocr_pipelines.run_pipeline("tesseract", image)
                self.assertEqual(error.exception.code, "pipeline_unavailable")

    def test_pdf_page_count_and_dimensions_are_input_limits(self):
        import pymupdf

        catalog = {"docling": {"available": True, "extensions": [".pdf"]}}
        with (
            tempfile.TemporaryDirectory() as work,
            patch.object(ocr_pipelines, "pipeline_catalog", return_value=catalog),
        ):
            path = Path(work) / "sample.pdf"
            for count, width, code in [
                (21, 600, "pdf_page_limit"),
                (1, 3000, "pdf_page_too_large"),
            ]:
                with pymupdf.open() as document:
                    for _ in range(count):
                        document.new_page(width=width, height=800)
                    document.save(path)
                with self.assertRaises(ocr_pipelines.InputError) as error:
                    ocr_pipelines.run_pipeline("docling", path)
                self.assertEqual(error.exception.code, code)
                path.unlink()

    def test_large_image_is_downscaled_before_inference(self):
        from PIL import Image

        def inspect(pipeline, source, language):
            self.assertEqual(source.suffix, ".png")
            with Image.open(source) as image:
                self.assertEqual(image.size, (4050, 81))
            return {"text": "processed"}

        catalog = {"tesseract": {"available": True, "extensions": [".jpg"]}}
        with (
            tempfile.TemporaryDirectory() as work,
            patch.object(ocr_pipelines, "pipeline_catalog", return_value=catalog),
            patch.object(ocr_pipelines, "_run_image", side_effect=inspect),
        ):
            path = Path(work) / "photo.jpg"
            Image.new("CMYK", (5000, 100)).save(path)
            result = ocr_pipelines.run_pipeline("tesseract", path)
            self.assertEqual(
                result["metadata"]["image_resize"],
                {"original_pixels": [5000, 100], "processed_pixels": [4050, 81]},
            )
            self.assertEqual(list(Path(work).iterdir()), [path])

    def test_raster_area_limit_preserves_aspect_ratio(self):
        scale = ocr_pipelines.bounded_raster_scale(4000, 4000)
        self.assertAlmostEqual((4000 * scale) ** 2, 12_000_000)
        self.assertEqual(ocr_pipelines.bounded_raster_scale(600, 800, 3), 3)

    def test_multiple_hints_check_for_any_requested_script(self):
        english = "ordinary English text " * 10
        self.assertIn("script_mismatch", ocr_pipelines._flags(english, ["hi", "ta"]))
        self.assertNotIn("script_mismatch", ocr_pipelines._flags(english, ["en", "hi"]))

    def test_repetition_and_script_mismatch_flags(self):
        repeated = " ".join(["मകൻ്റെ"] * 35)
        self.assertIn("repetition_suspected", ocr_pipelines._flags(repeated, "ml"))
        self.assertIn(
            "script_mismatch", ocr_pipelines._flags("ordinary English text " * 10, "ta")
        )
        self.assertIn("empty_output", ocr_pipelines._flags(" \n", "hi"))

    def test_text_result_contract_and_truncation_flag(self):
        result = ocr_pipelines._text_result(
            "x" * (ocr_pipelines.OUTPUT_CHAR_LIMIT + 1),
            pipeline="mock",
            model="mock model",
            version="1",
            language="en",
        )
        self.assertEqual(result["format"], "text")
        self.assertEqual(result["model"], "mock model")
        self.assertEqual(len(result["text"]), ocr_pipelines.OUTPUT_CHAR_LIMIT)
        self.assertIn("output_truncated", result["flags"])


if __name__ == "__main__":
    unittest.main()
