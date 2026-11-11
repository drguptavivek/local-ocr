"""Expose the local OCR HTTP API to MCP clients over stdio."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

REPO = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path(os.environ.get("OCR_MCP_INPUT_ROOT", REPO / "private" / "inbox"))
TOKEN_FILE = Path(
    os.environ.get(
        "OCR_MCP_TOKEN_FILE",
        Path.home() / ".local" / "share" / "aiims-ocr" / "clients" / "mcp.token",
    )
)
PORT = int(os.environ.get("OCR_MCP_PORT", "8092"))
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
mcp = FastMCP("local-ocr", log_level="WARNING")


def request(method: str, route: str, **kwargs) -> dict[str, Any]:
    headers = {}
    if route != "/health":
        try:
            info = TOKEN_FILE.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise ValueError(
                    "The MCP credential must be an owner-only regular file."
                )
            token = TOKEN_FILE.read_text().strip()
        except OSError:
            raise ValueError(
                "OCR MCP credential unavailable. Run make ocr-client CLIENT=mcp."
            ) from None
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{PORT}",
            trust_env=False,
            follow_redirects=False,
            timeout=30,
        ) as client:
            response = client.request(method, route, headers=headers, **kwargs)
    except httpx.RequestError:
        raise ValueError(
            "Could not reach the local OCR service. Check /health before resubmitting."
        ) from None
    if response.status_code == 204:
        return {"deleted": True}
    if not response.is_success:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        message = (
            detail
            if isinstance(detail, str)
            else "Check the request against the OCR API guide"
        )
        raise ValueError(f"OCR service HTTP {response.status_code}: {message}")
    return response.json()


def job_summary(job: dict[str, Any], include_text: bool = False) -> dict[str, Any]:
    if not include_text and "result" in job:
        job["result"].pop("text", None)
    return job


@mcp.tool(annotations=READ_ONLY)
def ocr_health() -> dict[str, Any]:
    """Check whether the local OCR worker is running."""
    return request("GET", "/health")


@mcp.tool(annotations=READ_ONLY)
def ocr_pipelines() -> dict[str, Any]:
    """List pipelines, their availability and their input capabilities."""
    return request("GET", "/v1/pipelines")


@mcp.tool(annotations=READ_ONLY)
def ocr_workload() -> dict[str, Any]:
    """Get shared queue capacity and retained-job counts for this MCP credential."""
    return request("GET", "/v1/workload")


@mcp.tool()
def ocr_submit(
    path: str, pipeline: str = "docling", languages: list[str] | None = None
) -> dict[str, Any]:
    """Queue an approved local image or PDF from the configured OCR inbox."""
    root = INPUT_ROOT.expanduser().resolve()
    try:
        source = (root / path).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("The input file must be inside the OCR inbox.")
        if not stat.S_ISREG(source.stat().st_mode):
            raise ValueError("The OCR input must be a regular file.")
        with source.open("rb") as file:
            if os.fstat(file.fileno()).st_size > 32 * 1024**2:
                raise ValueError("The OCR input exceeds 32 MiB.")
            fields = [("pipeline", (None, pipeline))]
            fields.extend(("languages", (None, code)) for code in languages or [])
            fields.append(("file", (source.name, file)))
            return job_summary(request("POST", "/v1/jobs", files=fields))
    except OSError:
        raise ValueError("Could not open the input file in the OCR inbox.") from None


@mcp.tool(annotations=READ_ONLY)
def ocr_job_status(job_id: UUID, include_text: bool = False) -> dict[str, Any]:
    """Poll a job. Set include_text to true to return the extracted text to the MCP client."""
    return job_summary(request("GET", f"/v1/jobs/{job_id}"), include_text)


@mcp.tool()
def ocr_retry_job(job_id: UUID) -> dict[str, Any]:
    """Requeue a failed OCR job under the same job ID."""
    return job_summary(request("POST", f"/v1/jobs/{job_id}/retry"))


@mcp.tool()
def ocr_delete_job(job_id: UUID) -> dict[str, Any]:
    """Delete a completed or failed job and its retained data."""
    return request("DELETE", f"/v1/jobs/{job_id}")


if __name__ == "__main__":
    mcp.run(transport="stdio")
