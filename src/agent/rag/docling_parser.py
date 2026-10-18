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
  * PDF 全量走 docling（2026-10 实验：撤掉「扫描件/图片重 → PyMuPDF 快速路径」
    拦截，改用 Apple Vision OCR 试试图片类 PDF 的速度）。OCR 引擎用 auto 探测：
    ocrmac 优先，未安装或请求语言不被 Vision 支持时自动回退 rapidocr。docling
    不可用/转换失败仍回退 parsers.py 的 PyMuPDF+RapidOCR 兜底路径。
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
    """Lazily build and cache the docling DocumentConverter singleton.

    管线降配（版本兼容，全部包在 try/except 中，任何失败都回退默认构造）：
    accelerator 强制 CPU + 限 4 线程。docling 2.x 在 macOS 上默认
    device='auto' → MPS（torch Metal 后端），其缓存分配器只增不放——
    实测同步图多的书时 RSS 可涨到 30GB+ 且不回落（#116 根因之一）。
    CPU 模式下布局/表格模型对文本为主的文档足够快，且内存有界。
    """
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

            converter = None
            try:
                from docling.datamodel.accelerator_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                )
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    OcrAutoOptions,
                    PdfPipelineOptions,
                )
                from docling.document_converter import PdfFormatOption

                pdf_opts = PdfPipelineOptions()
                pdf_opts.accelerator_options = AcceleratorOptions(
                    num_threads=4, device=AcceleratorDevice.CPU
                )
                # OCR 引擎用 auto 探测（2026-10 实验切换）：
                #   1. ocrmac（Apple Vision，系统级加速）优先；
                #   2. 未安装 ocrmac / 请求语言不被 Vision 支持时自动回退
                #      rapidocr（onnxruntime 后端），候选序见 docling
                #      auto_ocr_model.py。
                # lang 必须显式含 zh-Hans：OcrMacOptions 的字段默认值只有
                # 西语系（en/es/fr/de），中文书会被 Vision 拒识；而 rapidocr
                # 收到多语言时取第一个（zh-Hans → ch），回退路径不受影响。
                pdf_opts.ocr_options = OcrAutoOptions(lang=["zh-Hans", "en-US"])
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
                )
                logger.info("docling DocumentConverter initialized (PDF 管线强制 CPU 加速, OCR=auto: ocrmac→rapidocr)")
            except Exception as e:
                # docling 版本差异（2.13x 前后多次移动模块路径）：拿不到
                # 降配入口就用默认构造——正确性不受影响，只是 macOS 上
                # MPS 缓存可能再次推高内存。
                logger.info("docling 管线降配不可用（%s），使用默认配置", e)
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
