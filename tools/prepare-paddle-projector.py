#!/usr/bin/env python3
"""Prepare a reusable Q8 CPU projector from the official PaddleOCR-VL 1.6 GGUF."""

import os
import tempfile
from pathlib import Path

import gguf
import numpy as np

ROOT = Path.home() / ".models/ocr/PaddlePaddle/PaddleOCR-VL-1.6-GGUF"
SOURCE = ROOT / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
TARGET = ROOT / "PaddleOCR-VL-1.6-GGUF-mmproj-q8_0.gguf"


def convert(output: Path) -> None:
    reader = gguf.GGUFReader(SOURCE)
    architecture = reader.fields["general.architecture"].contents()
    if (
        architecture != "clip"
        or reader.endianess != gguf.GGUFEndian.LITTLE
        or reader.fields["clip.projector_type"].contents() != "paddleocr"
    ):
        raise ValueError("Expected the official little-endian PaddleOCR projector")
    writer = gguf.GGUFWriter(output, arch=architecture, endianess=reader.endianess)
    alignment = reader.fields.get("general.alignment")
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        value = field.contents()
        if name == "general.file_type":
            value = gguf.LlamaFileType.MOSTLY_Q8_0.value
        subtype = field.types[1] if len(field.types) > 1 else None
        writer.add_key_value(name, value, field.types[0], subtype)

    quantized = 0
    for tensor in reader.tensors:
        kind, data = tensor.tensor_type, tensor.data
        if kind == gguf.GGMLQuantizationType.BF16:
            values = gguf.dequantize(data, kind)
            if not np.isfinite(values).all() or np.abs(values).max() > 65504:
                raise ValueError("Projector contains weights outside the FP16 range")
            # Q8_0 uses 32-value blocks. Keep unaligned rows in FP16.
            if values.shape[-1] % 32 == 0:
                kind = gguf.GGMLQuantizationType.Q8_0
                data = gguf.quantize(values, kind)
                quantized += 1
            else:
                kind = gguf.GGMLQuantizationType.F16
                data = values.astype("<f2")
        writer.add_tensor(tensor.name, data, raw_dtype=kind)
    if not quantized:
        raise ValueError(
            "The source projector has no BF16 tensors compatible with Q8_0"
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    if not TARGET.is_file():
        with tempfile.TemporaryDirectory(prefix=".prepare-paddle-", dir=ROOT) as work:
            output = Path(work) / TARGET.name
            convert(output)
            try:
                os.link(output, TARGET)
            except FileExistsError:
                pass
    print(f"Q8 CPU projector ready at {TARGET}")


if __name__ == "__main__":
    main()
