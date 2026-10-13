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
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Formats docling can handle natively.  Text/markdown/csv/html are handled by
# the lightweight parsers (docling is overkill and slower for them).
_DOCLING_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf", ".epub"}

# Converter singleton: docling models are heavy, so share one instance across
# all workers (access is guarded by a lock — DocumentConverter is not
# documented as thread-safe).
#
# 锁必须在模块导入时就创建好。旧实现在 _get_converter() 里「if _converter_lock
# is None: _converter_lock = threading.Lock()」，这在并发首次调用时是典型的
# check-then-act 竞态：两个线程各自新建一把锁，于是各自持自己的锁去创建
# DocumentConverter —— 同时加载两份数百 MB 的模型。
_converter = None
_converter_lock = threading.Lock()
_converter_unavailable = False


def release_converter() -> None:
    """Drop the shared docling converter so its models can be freed.

    Mirrors ``parsers.release_ocr_engine()``: ingestion runs in short bursts
    (a sync job), after which the converter's models would otherwise stay
    resident for the whole process lifetime.
    """
    global _converter
    with _converter_lock:
        _converter = None


def reset_unavailable() -> None:
    """清掉「docling 不可用」的永久禁用标志。

    ``_converter_unavailable`` 在任何异常（不只是 ImportError）下都会被置位
    且没有复位路径。典型后果：进程早期跑过一次 ingest（docling 未装 → 禁用），
    随后依赖检查自动装上了 docling，之后的同步仍然全程回退到按格式解析
    （PDF 表格/版面质量下降），且毫无提示。依赖装好后调用本函数即可恢复。
    """
    global _converter_unavailable
    with _converter_lock:
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
    global _converter, _converter_unavailable
    if _converter is not None:
        return _converter
    if _converter_unavailable:
        return None
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
            # 只在**缺失**时永久禁用。瞬时错误（模型下载中断、磁盘抖动、
            # OOM）绝不能置位 —— 那会让本进程后续所有同步都静默降级到轻量
            # 解析器，即使原因早已消失也不自愈。
            logger.warning("failed to init docling converter (will retry): %s", e)
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
        # DocumentConverter 没有声明线程安全，而 ingest 用 4 个 worker 线程
        # 并行解析。旧实现只对「创建单例」加锁，convert() 本身完全不设防，
        # 并发转换会互相踩踏（状态污染/崩溃）。这里用同一把锁串行化转换。
        with _converter_lock:
            result = converter.convert(str(path))
            doc = result.document
            markdown = doc.export_to_markdown()
        # 及时释放大对象引用，避免整棵文档树在切片前一直常驻。
        del result, doc
        if not markdown or not markdown.strip():
            logger.warning("docling extracted empty markdown from %s", path.name)
            return None
        return markdown
    except Exception as e:
        # Never let a docling failure abort ingestion — caller falls back.
        logger.warning("docling conversion failed for %s: %s", path.name, e)
        return None


def _is_scan_pdf(path: Path) -> bool:
    """Detect whether a PDF looks like a scan (no text layer at the front).

    只探测前几页：旧实现 ``for page in doc`` 把整本 PDF 的文本全提取一遍——
    而「判定为扫描件」后走的解析器又要逐页提取/OCR 一遍，大 PDF 的解析时间
    近乎翻倍。扫描件几乎不可能前 5 页都没有文本层，这里采样首页+中间页即可。

    Uses PyMuPDF; returns False if PyMuPDF is unavailable (let docling try).
    """
    _PROBE_PAGES = 5
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return False
    try:
        with fitz.open(str(path)) as doc:
            n = min(_PROBE_PAGES, doc.page_count)
            for i in range(n):
                if doc[i].get_text("text").strip():
                    return False
        return True
    except Exception:
        return False
