#!/usr/bin/env python3
"""Install official Tesseract 5-compatible language data without replacing existing packs."""

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

SOURCE = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0"
DEFAULT = Path.home() / ".models/ocr/tesseract-ocr/tessdata"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "codes", nargs="+", help="Tesseract traineddata codes, e.g. ben guj pan tel"
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    for code in args.codes:
        if not re.fullmatch(r"[a-z]{3}(?:_[a-z]+)*|script/[A-Za-z][A-Za-z_]*", code):
            parser.error(f"Invalid traineddata code: {code}")

    root = args.directory.expanduser()
    for code in dict.fromkeys(args.codes):
        target = root / f"{code}.traineddata"
        if target.is_file():
            print(f"Already installed: {code}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".tessdata-", dir=target.parent
        ) as work:
            scratch = Path(work) / target.name
            with (
                urlopen(f"{SOURCE}/{code}.traineddata", timeout=30) as response,
                scratch.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            if scratch.stat().st_size < 1024:
                raise ValueError(f"Incomplete traineddata download: {code}")
            try:
                os.link(scratch, target)
            except FileExistsError:
                pass
        print(f"Installed: {code}")


if __name__ == "__main__":
    main()
