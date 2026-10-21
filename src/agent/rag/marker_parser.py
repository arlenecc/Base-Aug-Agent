"""PDF parsing via marker-pdf (parallel path to docling).

marker (2.x) is a layout-aware PDF→Markdown converter built on surya OCR.
Its 2.0 architecture runs the heavy models (VLM layout / full-page
recognition / fast-layout rf-detr) in shared surya *server* subprocesses,
so the in-process converter is a thin client — model RAM lives in those
children and is reclaimed via ``shutdown_models``.

Design (mirrors :mod:`agent.rag.docling_parser`):
  * ``extract_markdown(filepath)`` — convert a PDF to Markdown.
    Returns ``None`` when marker is unavailable or conversion fails
    (callers fall back to the next engine in the chain).
  * PDF only: marker's layout/table pipeline is tuned for PDFs; other
    formats stay on docling / the lightweight parsers.
  * Converter singleton guarded by a module-level lock. PdfConverter is
    not documented as thread-safe and surya servers batch per-request,
    so conversion is serialized exactly like docling's.
  * ``release_converter()`` — shuts down surya server subprocesses and
    drops the singleton (ingest runs in short bursts; without this the
    server processes would stay resident for the whole app lifetime).
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# surya 服务端（fast_layout / ocr_error / VLM 推理）由 marker 在首次转换时
# 自动拉起。它们的首启包含模型下载（ocr_error 的 DistilBert ~258MB），
# 默认 300s 启动超时在慢速网络下会直接 SpawnError（实测 500kB/s 需 ~9 分钟）。
# 这些 env 必须在 surya.settings 导入前设置（pydantic BaseSettings 在导入时
# 读 env），而 marker_parser 刻意不在模块级导入 marker，所以这里设置足够早。
for _var, _val in (
    ("OCR_ERROR_SERVER_STARTUP_TIMEOUT", "1800"),
    ("FAST_LAYOUT_SERVER_STARTUP_TIMEOUT", "1800"),
    ("SURYA_INFERENCE_STARTUP_TIMEOUT", "1800"),
):
    os.environ.setdefault(_var, _val)

# marker is PDF-focused: the converter pipeline (pdftext text-layer
# extraction + surya layout/OCR) only accepts PDFs (and bare images).
# Every other extension must return None so the engine chain routes it
# to docling / lightweight parsers without touching marker at all.
_MARKER_EXTENSIONS = {".pdf"}

_converter = None
# RLock：_get_converter 内部也拿这把锁，「取实例 + 转换」必须在同一把锁内
# 原子完成（持锁调用 _get_converter 需要同线程重入，Lock 会死锁）。
_converter_lock = threading.RLock()
_converter_unavailable = False


def is_marker_available() -> bool:
    """Return True if marker is installed (importable).

    用 ``find_spec`` 轻量探测，与 docling_parser.is_docling_available 同
    语义：只查包是否存在，不真正 import——否则一次可用性探测就把 torch /
    surya 全家桶拉进内存，而路由对每个文件都会调本函数。
    """
    global _converter_unavailable
    if _converter_unavailable:
        return False
    try:
        import importlib.util
        return importlib.util.find_spec("marker") is not None
    except Exception:
        return False


def is_supported(ext: str) -> bool:
    return ext.lower() in _MARKER_EXTENSIONS


def _get_converter():
    """Lazily build and cache the marker PdfConverter singleton.

    Models are NOT loaded here: create_model_dict() only constructs thin
    clients; the surya server subprocesses spawn on the first convert()
    call and stay alive until shutdown_models().  Any construction
    failure marks marker unavailable so the engine chain falls back to
    docling without retrying a broken import on every page.
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
            from marker.config.parser import ConfigParser
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            config_parser = ConfigParser({
                "output_format": "markdown",
                # RAG 管线只要文本：关闭图片裁剪/编码（每张图一次页面
                # 渲染 + PNG 编码，扫描件/幻灯片类 PDF 上开销显著），
                # markdown 中保留图片占位行为空。
                "extract_images": False,
                # 不用 LLM 后处理（需要 API key，且离线场景不可用）。
                "use_llm": False,
            })
            converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service(),
            )
            _converter = converter
            logger.info("marker PdfConverter initialized (global singleton)")
            return converter
        except ImportError:
            logger.info("marker not installed; engine chain falls back to docling")
            _converter_unavailable = True
            return None
        except Exception as e:
            logger.warning("failed to init marker converter: %s", e)
            _converter_unavailable = True
            return None


def release_converter() -> None:
    """Shut down surya server subprocesses and drop the converter singleton.

    Mirrors ``docling_parser.release_converter()``: ingestion runs in
    short bursts, and marker's server children (VLM layout + recognition)
    would otherwise stay resident — each potentially holding GB of model
    weights — for the whole process lifetime.
    """
    global _converter
    with _converter_lock:
        if _converter is None:
            return
        try:
            from marker.models import shutdown_models

            shutdown_models(_converter.artifact_dict)
        except Exception as e:
            logger.debug("marker: shutdown_models error: %s", e)
        _converter = None


def extract_markdown(filepath: str) -> Optional[str]:
    """Convert a PDF to Markdown using marker.

    Returns the Markdown text, or ``None`` if marker is unavailable, the
    format is not handled, or conversion failed (callers should fall
    back to the next engine in the chain).
    """
    if not is_marker_available():
        return None

    path = Path(filepath)
    if path.suffix.lower() not in _MARKER_EXTENSIONS:
        return None

    try:
        # 与 docling 同样的理由：PdfConverter 未声明线程安全，ingest 的
        # 4 个 worker 线程必须串行化转换；surya 服务端按请求批处理，
        # 串行调用不损失吞吐。
        # 取实例在锁内完成（2026-10 并发审查修复）：旧实现锁外取、锁内转，
        # 与 release_converter 竞态会出现「A 转换完回收 → C 新建转换器 →
        # B 仍用旧转换器」的两代并存——旧转换器的首次 convert 会再拉起一
        # 整套 surya 服务端子进程，两组子进程并存直到 B 结束。
        with _converter_lock:
            converter = _get_converter()
            if converter is None:
                return None
            rendered = converter(str(path))
            markdown = rendered.markdown
            del rendered
        if not markdown or not markdown.strip():
            logger.warning("marker extracted empty markdown from %s", path.name)
            return None
        return markdown
    except Exception as e:
        # Never let a marker failure abort ingestion — caller falls back.
        logger.warning("marker conversion failed for %s: %s", path.name, e)
        return None
