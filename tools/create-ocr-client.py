#!/usr/bin/env python3
"""Create an owner-only bearer credential for one local OCR client."""

import argparse
import os
import re
import secrets
import stat
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client", help="Application name, e.g. digitva")
    parser.add_argument("--clients-dir", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", args.client):
        parser.error("client must use lowercase letters, digits, or hyphens")

    os.umask(0o077)
    state = Path(
        os.environ.get("OCR_SERVICE_STATE", "~/.local/share/aiims-ocr")
    ).expanduser()
    directory = args.clients_dir or Path(
        os.environ.get("OCR_SERVICE_CLIENTS", state / "clients")
    )
    directory = directory.expanduser()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        parser.error("clients directory must be owned by this user and private")
    credential = directory / f"{args.client}.token"
    try:
        with credential.open("x", encoding="utf-8") as file:
            file.write(secrets.token_urlsafe(32) + "\n")
    except FileExistsError:
        parser.exit(1, f"Credential already exists at {credential}\n")
    print(f"Credential saved at {credential}")


if __name__ == "__main__":
    main()
