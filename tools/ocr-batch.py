#!/usr/bin/env python3
"""Submit local files one at a time and save OCR responses beside the inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from time import monotonic, sleep

import httpx

LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "tamil": "ta",
    "malayalam": "ml",
    "kannada": "kn",
    "odia": "or",
}


def save_sidecar(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".ocr-", delete=False
    ) as output:
        scratch = Path(output.name)
        try:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
            output.close()
            scratch.replace(path)
        finally:
            scratch.unlink(missing_ok=True)


def already_completed(path: Path, digest: str, device: str, model: str) -> bool:
    if not path.is_file():
        return False
    prior = json.loads(path.read_text(encoding="utf-8"))
    return (
        prior.get("status") == "completed"
        and prior.get("input_sha256") == digest
        and prior.get("requested_device") == device
        and prior.get("catalog_model") == model
    )


def process(api, path, pipeline, hints, *, wait_seconds):
    start = monotonic()
    with path.open("rb") as source:
        response = api.post(
            "/v1/jobs",
            data={"pipeline": pipeline, "languages": hints},
            files={"file": (path.name, source)},
        )
    if not response.is_success:
        return {
            "status": "rejected",
            "http_status": response.status_code,
            "error": response.json().get("detail"),
        }, monotonic() - start
    job = response.json()
    deadline = start + wait_seconds
    while job["status"] in {"queued", "processing"} and monotonic() < deadline:
        sleep(1)
        job = api.get(f"/v1/jobs/{job['job_id']}").raise_for_status().json()
    return job, monotonic() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--client", required=True, help="Local OCR credential name")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--port", type=int, default=8092)
    engines = parser.add_mutually_exclusive_group(required=True)
    engines.add_argument(
        "--pipeline", action="append", help="Repeat to compare engines"
    )
    engines.add_argument("--all-pipelines", action="store_true")
    language = parser.add_mutually_exclusive_group()
    language.add_argument("--languages", nargs="+", metavar="CODE")
    language.add_argument("--language-from-folder", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=3600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        not args.folder.is_dir()
        or not 1 <= args.port <= 65535
        or args.wait_seconds < 1
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.client)
        or len(args.languages or []) > 8
        or any(not re.fullmatch(r"[a-z][a-z_-]{0,31}", s) for s in args.languages or [])
    ):
        parser.error(
            "Check the folder, port, client name, wait limit and language codes"
        )
    token_path = args.token_file or (
        Path.home() / ".local/share/aiims-ocr/clients" / f"{args.client}.token"
    )
    token = token_path.expanduser().read_text().strip()
    counts = Counter()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
        trust_env=False,
    ) as api:
        catalog = api.get("/v1/pipelines").raise_for_status().json()["pipelines"]
        if args.pipeline:
            for pipeline in args.pipeline:
                if pipeline not in catalog or not catalog[pipeline]["available"]:
                    parser.error(f"Unavailable pipeline: {pipeline}")
        selected = args.pipeline or [
            name for name, cfg in catalog.items() if cfg["available"]
        ]
        for path in sorted(args.folder.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            hint = (
                LANGUAGES.get(path.parent.name.lower())
                if args.language_from_folder
                else None
            )
            hints = args.languages or ([hint] if hint else [])
            language_tag = "+".join(hints) or "auto"
            digest = None
            for pipeline in dict.fromkeys(selected):
                cfg = catalog[pipeline]
                if path.suffix.lower() not in cfg["extensions"]:
                    continue
                allowed = cfg.get("recognition_languages", cfg.get("languages"))
                if (
                    hints
                    and allowed is not None
                    and any(hint not in allowed for hint in hints)
                ):
                    counts["language_unsupported"] += 1
                    continue
                if digest is None:
                    with path.open("rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                sidecar = path.with_name(f"{path.name}.{pipeline}.{language_tag}.json")
                if not args.overwrite and already_completed(
                    sidecar, digest, cfg["device"], cfg["model"]
                ):
                    counts["already_completed"] += 1
                    continue
                job, seconds = process(
                    api, path, pipeline, hints, wait_seconds=args.wait_seconds
                )
                save_sidecar(
                    sidecar,
                    {
                        "pipeline": pipeline,
                        "languages": hints,
                        **job,
                        "input_sha256": digest,
                        "catalog_model": cfg["model"],
                        "requested_device": cfg["device"],
                        "elapsed_seconds": round(seconds, 3),
                    },
                )
                counts[job["status"]] += 1
                if job["status"] in {"completed", "failed"}:
                    api.delete(f"/v1/jobs/{job['job_id']}").raise_for_status()
                if job["status"] in {"queued", "processing"}:
                    print(json.dumps(dict(counts), sort_keys=True))
                    print(
                        "Wait limit reached. Rerun the command to resume.",
                        file=sys.stderr,
                    )
                    return 1
    print(json.dumps(dict(counts), sort_keys=True))
    return int(
        bool(
            counts["failed"]
            or counts["rejected"]
            or not (counts["completed"] or counts["already_completed"])
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
