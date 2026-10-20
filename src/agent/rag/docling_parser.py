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
    不可用/转换失败仍回退 parsers.py 的 PyMuPDF 兜底路径（同样 ocrmac 优先、
    回退 rapidocr，见 parsers.ocr_recognize_array）。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Formats docling handles.  Text/markdown/csv/html 由轻量解析器处理；
# .epub 由专门的 ebook 路径（pandoc convert_ebook → ebook_parser）处理，不在此列
# —— 一旦进 docling，DocumentConverter 会加载布局模型 + TableFormer（+3.7GB / 2s），
# 对 epub 纯属浪费，故显式排除。若某天需要 docling 处理 epub，这里再加回。
_DOCLING_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf"}

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


# 转换篇数计数器 + 周期冲刷间隔。
#
# 为什么需要：docling 用 OcrAutoOptions **自己实例化**一份 OCR/ORT session，
# parsers.py 的 _OCR_RECYCLE_PAGES 周期重建只作用于 parsers._OCR_ENGINE 那个
# 模块级单例，管不到 docling 内部这份。整库同步几百页扫描件时，那份原生缓存
# 按输入形状在**同步过程中**单调增长，直到 engine.close() 才释放——中途就会被
# 系统杀掉（docling 是默认 pdf_engine，这是主路径）。按转换篇数主动置空单例，
# 下一份文档惰性重建，把峰值限制在一代内。
_DOCLING_RECYCLE_DOCS = 50
_convert_count = 0
_convert_count_lock = threading.Lock()


def _maybe_recycle_converter() -> None:
    """每 _DOCLING_RECYCLE_DOCS 篇冲刷一次 docling 原生缓存。

    必须在 **_converter_lock 之外**调用：threading.Lock 不可重入，在持有锁的
    转换流程里直接调 release_converter() 会死锁。
    """
    global _convert_count
    with _convert_count_lock:
        _convert_count += 1
        n = _convert_count
    if n % _DOCLING_RECYCLE_DOCS:
        return
    release_converter()
    logger.info("docling 转换器已周期重建（每 %d 篇，冲刷 OCR/ORT 原生缓存）",
                _DOCLING_RECYCLE_DOCS)


def is_docling_available() -> bool:
    """Return True if docling is installed (importable).

    用 ``importlib.util.find_spec`` 做**轻量探测**：只检查包是否可导入，绝不真正
    import docling——否则仅仅探测可用性就会把 torch 全家桶（docling 的依赖）加载
    进内存。torch 应只在真正需要转换的 PDF 走到 ``_get_converter()`` 时才加载。
    路由对每个文件都会调用本函数，轻量探测可避免对 .txt/.md 等根本用不到 docling
    的文件也顺带拉起 torch；pandoc 不可用时本该走 ebook 路径的 .epub 同理。
    """
    global _converter_unavailable
    if _converter_unavailable:
        return False
    try:
        import importlib.util
        return importlib.util.find_spec("docling") is not None
    except Exception:
        return False


def is_supported(ext: str) -> bool:
    return ext.lower() in _DOCLING_EXTENSIONS


# OCR 渲染分辨率上限（DPI）。docling 的 OcrOptions.scale 语义是「72 DPI ×
# scale」，默认 3.0 = 216 DPI；这里锁到 150 DPI，与 parsers.py 的 _OCR_DPI
# 保持一致，避免高分辨率扫描页把 OCR 原生缓存撑爆。
# 换算：scale = _DOCLING_OCR_DPI / 72.0 ≈ 2.083
_DOCLING_OCR_DPI = 150


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
                # 注意：这里**不要**动 pdf_opts.images_scale。
                # images_scale 管的是版面/表格/图片描述模型用的**整页图**
                # （page.image），与 OCR 渲染是两条独立通路；本机实测其默认
                # 值是 1.0（72 DPI），把它设成 150/72 只会**抬高**整页图
                # 分辨率、增加内存，与你想要的「限制」正好相反。
                # OCR 的渲染倍率由下面的 OcrAutoOptions.scale 控制。
                # OCR 引擎用 auto 探测（2026-10 实验切换）：
                #   1. ocrmac（Apple Vision，系统级加速）优先；
                #   2. 未安装 ocrmac / 请求语言不被 Vision 支持时自动回退
                #      rapidocr（onnxruntime 后端），候选序见 docling
                #      auto_ocr_model.py。
                # lang 必须显式含 zh-Hans：OcrMacOptions 的字段默认值只有
                # 西语系（en/es/fr/de），中文书会被 Vision 拒识；而 rapidocr
                # 收到多语言时取第一个（zh-Hans → ch），回退路径不受影响。
                # scale 才是 OCR 的渲染倍率：页面按「72 DPI × scale」渲染，
                # docling 默认 3.0 → **216 DPI**（远超 150 上限）。这里锁到
                # 150/72 ≈ 2.083 → 150 DPI，与 parsers.py 的 _OCR_DPI 对齐。
                # 且 docling 是对每个 OCR 区域**裁剪后**单独渲染
                # （get_page_image(scale=..., cropbox=ocr_rect)），降 scale
                # 直接减少送进 ocrmac/rapidocr 的像素量。
                pdf_opts.ocr_options = OcrAutoOptions(
                    lang=["zh-Hans", "en-US"],
                    scale=_DOCLING_OCR_DPI / 72.0,
                )
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

    # 预置为 None：finally 里无条件解引用，异常发生在赋值之前也不会
    # UnboundLocalError（旧实现把释放只放在成功路径）。
    result = None
    doc = None
    markdown = None
    try:
        # DocumentConverter 没有声明线程安全，而 ingest 用 4 个 worker 线程
        # 并行解析。旧实现只对「创建单例」加锁，convert() 本身完全不设防，
        # 并发转换会互相踩踏（状态污染/崩溃）。这里用同一把锁串行化转换。
        with _converter_lock:
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
    finally:
        # 及时释放大对象引用，避免整棵文档树在切片前一直常驻。
        # ConversionResult 持有整棵文档树 + 每页位图（数十~数百 MB）；
        # 旧实现只在成功路径 del，export_to_markdown() 抛异常时它活到函数
        # 返回才被回收，异常密集时是实打实的常驻。
        result = None
        doc = None
        # 周期性冲刷 docling 原生缓存（见 _maybe_recycle_converter 注释）。
        # 必须在 _converter_lock 之外：Lock 不可重入。
        try:
            _maybe_recycle_converter()
        except Exception as e:  # 绝不能让回收逻辑影响解析结果
            logger.debug("docling 周期回收失败（忽略）: %s", e)
