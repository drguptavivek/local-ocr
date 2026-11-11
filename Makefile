OCR_PYTHON ?= .venv/bin/python
OCR_SERVICE_PORT ?= 8092

.PHONY: ocr-service ocr-client test
ocr-service:
	"$(OCR_PYTHON)" -m backend.ocr_service --port "$(OCR_SERVICE_PORT)"

ocr-client:
	"$(OCR_PYTHON)" tools/create-ocr-client.py "$(CLIENT)"

test:
	"$(OCR_PYTHON)" -m unittest discover -s tests
