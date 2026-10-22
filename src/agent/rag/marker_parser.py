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

from .concurrency_gate import SizeAwareConversionGate

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

# 转换器空闲池（list）。marker 的 PdfConverter 未声明线程安全，存取/构建由
# _pool_lock 保护，并发执行转换由 _conv_gate（按文件大小限流）控制。
#
# 并发策略（见 _conv_gate 与阈值常量）：小文件（< 20MB）最多 2 个实例并发
# 转换（各拉起一套 surya 服务端子进程），大文件（>= 20MB）独占，防止大文件
# 并发把内存撑爆。空闲池上限 = _SMALL_FILE_CONCURRENCY。
_converters: list = []
_pool_lock = threading.Lock()
_converter_unavailable = False

# 按文件大小限流的并发闸门参数（与 docling_parser 同语义）。
_LARGE_FILE_THRESHOLD = 20 * 1024 * 1024   # 20MB：以上视为大文件，独占实例
_SMALL_FILE_CONCURRENCY = 2                # 小文件最多 2 路并发
_conv_gate = SizeAwareConversionGate(_LARGE_FILE_THRESHOLD, _SMALL_FILE_CONCURRENCY)


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


def _build_converter():
    """构建一个新的 marker PdfConverter（含版本兼容降级）。

    Models are NOT loaded here: create_model_dict() only constructs thin
    clients; the surya server subprocesses spawn on the first convert()
    call and stay alive until shutdown_models(). 构建失败抛异常，由调用方
    标记 unavailable。
    """
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
    return converter


def _get_converter():
    """从空闲池取一个转换器，没有则惰性新建（最多 _SMALL_FILE_CONCURRENCY 个
    并存实例）。返回 None 表示 marker 不可用（导入/构建失败，已标记
    _converter_unavailable）。

    空闲池的取/还在 _pool_lock 下完成；构建较重（拉起 surya 客户端），放到
    锁外，避免持池锁时阻塞其它 worker 取还实例。
    """
    global _converter_unavailable
    with _pool_lock:
        if _converters:
            return _converters.pop()
        if _converter_unavailable:
            return None
    # 锁外构建（拉起 surya 客户端，较重）。
    try:
        converter = _build_converter()
    except ImportError:
        logger.info("marker not installed; engine chain falls back to docling")
        _converter_unavailable = True
        return None
    except Exception as e:
        logger.warning("failed to init marker converter: %s", e)
        _converter_unavailable = True
        return None
    logger.info("marker PdfConverter initialized (pooled)")
    return converter


def _release_pooled_converter(converter) -> None:
    """把用过的转换器还回空闲池；只保留最多 _SMALL_FILE_CONCURRENCY 个空闲
    实例，避免无限增长（多余的丢弃引用，surya 子进程在 release_converter
    时统一关停）。
    """
    if converter is None:
        return
    with _pool_lock:
        if len(_converters) < _SMALL_FILE_CONCURRENCY:
            _converters.append(converter)


def release_converter() -> None:
    """Shut down surya server subprocesses and drop all pooled converters.

    Mirrors ``docling_parser.release_converter()``: ingestion runs in
    short bursts, and marker's server children (VLM layout + recognition)
    would otherwise stay resident — each potentially holding GB of model
    weights — for the whole process lifetime. 只清空闲池里的实例；正在被某
    路转换持有的实例会在该路 _release_pooled_converter 时归还（或随引用
    回收），不会被这里强行打断。
    """
    with _pool_lock:
        converters = list(_converters)
        _converters.clear()
    for c in converters:
        try:
            from marker.models import shutdown_models

            shutdown_models(c.artifact_dict)
        except Exception as e:
            logger.debug("marker: shutdown_models error: %s", e)


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
        # PdfConverter 未声明线程安全，并发执行转换由 _conv_gate 限流：
        #   * 小文件（< 20MB）最多 2 路并发（各一套 surya 子进程）；
        #   * 大文件（>= 20MB）独占，防止大文件并发把内存撑爆。
        # 取/还实例走 _pool_lock 保护的空闲池（最多 2 个并存实例）。闸门只
        # 控制「同时几路在跑」、池控制「最多建几个实例」，二者解耦；两代
        # 并存竞态由闸门限流（而非全局排他锁）规避，转换完即归还实例。
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        _conv_gate.acquire(size)
        converter = None
        try:
            converter = _get_converter()
            if converter is None:
                return None
            rendered = converter(str(path))
            markdown = rendered.markdown
            del rendered
        finally:
            _release_pooled_converter(converter)
            _conv_gate.release(size)
        if not markdown or not markdown.strip():
            logger.warning("marker extracted empty markdown from %s", path.name)
            return None
        return markdown
    except Exception as e:
        # Never let a marker failure abort ingestion — caller falls back.
        logger.warning("marker conversion failed for %s: %s", path.name, e)
        return None
