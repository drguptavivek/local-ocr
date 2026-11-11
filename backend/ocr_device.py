"""Service-wide device policy for the PyTorch and llama.cpp OCR engines."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from functools import cache
from pathlib import Path


def device_mode() -> str:
    mode = os.environ.get("OCR_DEVICE", "auto").strip().lower()
    if mode not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("OCR_DEVICE must be auto, cpu, mps, or cuda")
    return mode


def gpu_layers() -> int:
    value = os.environ.get("OCR_GPU_LAYERS", "99")
    if not re.fullmatch(r"[0-9]{1,3}", value):
        raise ValueError("OCR_GPU_LAYERS must be an integer from 0 to 999")
    return int(value)


def torch_device(mode: str) -> str | None:
    if mode == "cpu":
        return "cpu"
    import torch

    if mode in {"auto", "mps"} and torch.backends.mps.is_available():
        return "mps"
    if mode in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda"
    return "cpu" if mode == "auto" else None


@cache
def probe_torch_device(mode: str) -> str | None:
    if mode == "cpu":
        return "cpu"
    try:
        probe = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    device = probe.stdout.strip()
    return (
        device if probe.returncode == 0 and device in {"cpu", "cuda", "mps"} else None
    )


@cache
def llama_device(binary: str | None, mode: str) -> str | None:
    if binary is None or mode == "cpu":
        return "cpu" if binary is not None else None
    try:
        probe = subprocess.run(
            [binary, "--list-devices"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode:
        return None
    devices = re.findall(
        r"^\s*((?:MTL|CUDA)\d+):", probe.stdout + "\n" + probe.stderr, re.MULTILINE
    )
    for prefix, name in (("MTL", "mps"), ("CUDA", "cuda")):
        if mode in {"auto", name}:
            match = next(
                (device for device in devices if device.startswith(prefix)), None
            )
            if match is not None:
                return match
    return "cpu" if mode == "auto" else None


def llama_args(device: str) -> list[str]:
    if device == "cpu":
        return ["--device", "none", "-ngl", "0", "--no-mmproj-offload"]
    return ["--device", device, "-ngl", str(gpu_layers())]


if __name__ == "__main__":
    print(torch_device(sys.argv[1]) or "")
