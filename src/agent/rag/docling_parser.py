"""Unified document parsing via docling.

docling is a single, high-quality document-understanding engine that converts
Word, Excel, PowerPoint, PDF and EPUB into structured Markdown (with layout
awareness, tables, headings, and image OCR).  This module wraps it behind the
same ``extract_text``-style interface so the RAG engine can use one parser for
all office/document formats.

Design:
  * ``extract_markdown(filepath)`` — convert any supported document to Markdown
    via docling.  Returns ``None`` when docling is unavailable (callers fall
    back to the per-format parsers in :mod:`agent.rag.parsers`).
  * Image-based PDF pages are OCR'd by RapidOCR (docling's OCR backend uses
    RapidOCR under the hood; for pure-scan PDFs without a text layer we route
    through the shared RapidOCR singleton in :mod:`agent.rag.parsers`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Formats docling can handle natively.  Text/markdown/csv/html are handled by
# the lightweight parsers (docling is overkill and slower for them).
_DOCLING_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf", ".epub"}

# Converter singleton: docling models are heavy, so share one instance across
# all workers (access is guarded by a lock — DocumentConverter is not
# documented as thread-safe).
_converter = None
_converter_lock = None
_converter_unavailable = False


def is_docling_available() -> bool:
    """Return True if docling can be imported."""
    global _converter_unavailable
    if _converter_unavailable:
        return False
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        _converter_unavailable = True
        return False


def is_supported(ext: str) -> bool:
    return ext.lower() in _DOCLING_EXTENSIONS


def _get_converter():
    """Lazily build and cache the docling DocumentConverter singleton."""
    global _converter, _converter_lock, _converter_unavailable
    if _converter is not None:
        return _converter
    if _converter_unavailable:
        return None
    if _converter_lock is None:
        import threading
        _converter_lock = threading.Lock()
    with _converter_lock:
        if _converter is not None:
            return _converter
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            _converter = converter
            logger.info("docling DocumentConverter initialized (global singleton)")
            return converter
        except ImportError:
            logger.info("docling not installed; falling back to per-format parsers")
            _converter_unavailable = True
            return None
        except Exception as e:
            logger.warning("failed to init docling converter: %s", e)
            _converter_unavailable = True
            return None


def extract_markdown(filepath: str) -> Optional[str]:
    """Convert a document to Markdown using docling.

    Returns the Markdown text, or ``None`` if docling is unavailable or the
    format is not handled by docling (callers should fall back to the
    per-format parsers).
    """
    if not is_docling_available():
        return None

    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix not in _DOCLING_EXTENSIONS:
        return None

    # Image-based PDF handling: docling OCRs image pages via RapidOCR, but a
    # pure-scan PDF (no text layer at all) is better served by the dedicated
    # RapidOCR path in parsers.py which reuses the shared engine.  Detect that
    # case cheaply and delegate.
    if suffix == ".pdf" and _is_scan_pdf(path):
        return None

    converter = _get_converter()
    if converter is None:
        return None

    try:
        result = converter.convert(str(path))
        doc = result.document
        markdown = doc.export_to_markdown()
        if not markdown or not markdown.strip():
            logger.warning("docling extracted empty markdown from %s", path.name)
            return None
        return markdown
    except Exception as e:
        # Never let a docling failure abort ingestion — caller falls back.
        logger.warning("docling conversion failed for %s: %s", path.name, e)
        return None


def _is_scan_pdf(path: Path) -> bool:
    """Detect whether a PDF is a pure scan (no text layer on any page).

    Uses PyMuPDF for a cheap first-page probe; returns False if PyMuPDF is
    unavailable (let docling try).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False
    try:
        with fitz.open(str(path)) as doc:
            for page in doc:
                if page.get_text("text").strip():
                    return False
        return True
    except Exception:
        return False
