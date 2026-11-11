"""Vendored, slimmed, detection-only copy of Roboflow's RF-DETR for surya's fast detectors.

Avoids a runtime dependency on the `rfdetr` package (and its heavy transitive deps:
roboflow, rf100vl, albumentations, supervision, peft). Pure PyTorch; runs on cpu/mps/cuda.
See `predictor.RFDetrDetector` for the inference entry point.
"""

from surya.common.rfdetr.predictor import RFDetrDetector

__all__ = ["RFDetrDetector"]
