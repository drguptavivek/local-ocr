---
name: local-ocr
description: Submit approved local PDFs and images to the shared OCR service from DigitVA, the AIIMS document library, or another local app. Use when choosing an OCR pipeline, supplying language hints, monitoring the queue, polling jobs, or recovering from a pipeline timeout.
---

# Local OCR

Read the [service and API guide](../../../docs/shared-local-ocr.md) for request
examples and the [MCP setup](../../../README.md#use-from-an-mcp-client) for
agent access. Use the calling app's own bearer credential for HTTP. The MCP
adapter has its own credential and can see jobs submitted under that credential.

1. Check health, then check the pipeline catalog for availability, supported
   file types and language hints. Use Docling for PDFs. For images, choose an
   engine appropriate for the script and whether the input is handwritten.
2. Submit the approved input. For MCP, pass a path in the configured inbox.
   For HTTP, send a multipart file and repeat the `languages` field when
   supplying multiple hints. Keep the returned job ID with the source record.
3. Poll the job status at a modest interval until `completed` or `failed`.
   The returned execution timeout starts when processing begins. For bulk
   runs, check workload before filling the shared queue or the app's storage.
4. If a job reports `pipeline_timeout`, use the retry endpoint for another
   attempt on that job ID. If the pipeline or input needs to change, submit
   a new request with the corrected settings or file.
5. Save the result, model provenance and source linkage in the owning app.
   Request MCP text with `include_text: true` only when the text is approved
   for the connected client. Review suspect output against the source.
   Delete a terminal job after the caller has saved what it needs.

Keep credentials, source files, OCR text and clinical linkage out of Git and
logs. DigitVA results and verified references belong in its private storage.
