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
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Formats docling handles.  Text/markdown/csv/html 由轻量解析器处理；
# .epub 由专门的 ebook 路径（pandoc convert_ebook → ebook_parser）处理，不在此列
# —— 一旦进 docling，DocumentConverter 会加载布局模型 + TableFormer（+3.7GB / 2s），
# 对 epub 纯属浪费，故显式排除。若某天需要 docling 处理 epub，这里再加回。
_DOCLING_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf"}

# Converter 实例池（按 OCR scale 分桶，每桶一个 list）。docling 模型很重，跨
# worker 共享实例；DocumentConverter 未声明线程安全，存取/构建由 _pool_lock
# 保护，并发执行转换由 _conv_gate（按文件大小限流）控制。
#
# 并发策略（见 _conv_gate 与 _LARGE_FILE_THRESHOLD / _SMALL_FILE_CONCURRENCY）：
#   * 小文件（< 20MB）：最多 2 个转换器实例并发转换，加快小文件吞吐。
#   * 大文件（>= 20MB）：独占（1 个实例），防止大文件并发把内存撑爆。
# 实例池容量上限 = _SMALL_FILE_CONCURRENCY，避免无限制新建实例。
#
# 锁必须在模块导入时就创建好。旧实现在 _get_converter() 里「if 锁 is None:
# 新建锁」是典型 check-then-act 竞态：两线程各自建锁，各自持锁去创建
# DocumentConverter —— 同时加载两份数百 MB 模型。
#
# 为什么 _pool_lock 是 Lock 而非 RLock：取/还实例与构建不再需要「取实例」和
# 「执行转换」同锁重入——执行转换的并发约束已交给 _conv_gate，_pool_lock 只
# 负责池的取/还/构建这段临界区，纯互斥、无重入需求。
from .concurrency_gate import SizeAwareConversionGate

# 实例池（按 OCR scale 分桶）：scale -> [[converter, uses], ...]（**空闲**实例）。
# uses = 该实例已服务的转换次数。池里每个条目是 [实例, 已服务次数]，退役判定
# 挂在实例上，不挂全局计数——理由见 _release_pooled_converter 的注释。
_converters: dict = {}
# 借出中的实例登记表：[[converter, uses], ...]。取用时登记、归还时按 identity
# 找到条目累加 uses。没有这张表就无法知道「池里这个实例已经服过多少役」。
_checked_out: list = []
_pool_lock = threading.Lock()     # 保护 _converters / _checked_out 的取还/构建
_converter_unavailable = False

# 与永久闩锁（上面的 _converter_unavailable，仅用于 ImportError）互补的
# **瞬时失败负缓存**：构建转换器时的失败大多是瞬时的——MPS 显存瞬时不足、
# 模型权重正在被另一路加载、磁盘 I/O 抖动。旧实现把任何异常都打成永久闩锁，
# 一次抖动就让 docling 在整个进程生命周期内静默失效（全部 PDF 掉到轻量兜底，
# 用户只会觉得「这批同步质量变差了」却找不到原因）。
# 这里改为：瞬时失败只冷却 _CONV_FAIL_COOLDOWN_S 秒，之后允许重试自愈。
_converter_failed_until = 0.0
_CONV_FAIL_COOLDOWN_S = 60.0

# 按文件大小限流的并发闸门参数。
_LARGE_FILE_THRESHOLD = 20 * 1024 * 1024   # 20MB：以上视为大文件，独占实例
_SMALL_FILE_CONCURRENCY = 2                # 小文件最多 2 路并发
_conv_gate = SizeAwareConversionGate(_LARGE_FILE_THRESHOLD, _SMALL_FILE_CONCURRENCY)

# 闸门等待超过该秒数就打 INFO 日志——同步期「很久没输出」时，这条日志能直接
# 区分「卡在等闸门（大文件独占/并发已满）」与「卡在 convert 本身」。
_GATE_WAIT_LOG_S = 5.0


def release_converter() -> None:
    """Drop the shared docling converter(s) so their models can be freed.

    Mirrors ``parsers.release_ocr_engine()``: ingestion runs in short bursts
    (a sync job), after which the converters' models would otherwise stay
    resident for the whole process lifetime. 只清空闲池里的实例——正在被某路
    转换持有的实例会在该路 _release_pooled_converter 时自然归还（或随引用
    回收），不会被这里强行打断。
    """
    with _pool_lock:
        _converters.clear()


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


def _convert_with_scale(path: str, scale: float) -> Optional[str]:
    """取转换器 + 转换 + 导出，全程持大小闸门（按文件大小限流并发实例数）。

    并发约束不再靠「全局排他锁」串行化所有转换，而是交给 _conv_gate：
      * 小文件（< 20MB）最多 _SMALL_FILE_CONCURRENCY(2) 路并发转换；
      * 大文件（>= 20MB）独占（1 路），防止大文件并发把内存撑爆。

    取/还实例走 _pool_lock 保护的空闲池（最多 2 个并存实例），见
    _get_converter / _release_pooled_converter。闸门只控制「同一时刻有几路
    转换在跑」，实例池控制「最多建几个实例」，二者解耦。
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        size = 0
    t_wait = time.monotonic()
    permit = _conv_gate.acquire(size)
    waited = time.monotonic() - t_wait
    if waited >= _GATE_WAIT_LOG_S:
        # 同步期「很久没输出」时，这条日志直接区分：卡在等闸门（大文件独占中 /
        # 小文件并发已满）还是卡在 convert 本身（后者不会有本日志）。
        logger.info(
            "docling 转换闸门等待 %.1fs 后开始转换（%s）: %s",
            waited,
            "大文件独占转换中" if size >= _conv_gate.threshold else "小文件并发已满",
            getattr(path, "name", path),
        )
    converter = None
    try:
        converter = _get_converter(scale)
        if converter is None:
            return None
        result = converter.convert(str(path))
        md = result.document.export_to_markdown()
        result = None
        return md
    finally:
        _release_pooled_converter(scale, converter)
        _conv_gate.release(permit)


def _maybe_recycle_converter() -> None:
    """每 _DOCLING_RECYCLE_DOCS 篇全局冲刷一次：清空空闲池 + 周期重建日志。

    在 **_conv_gate / _pool_lock 之外**调用（转换流程的 finally 里，且
    _convert_with_scale 已 _conv_gate.release 之后）：只清空闲池里的实例
    （release_converter 取 _pool_lock），不会打断正在跑的转换。

    注意：**防缓存累积的主护栏是 _release_pooled_converter 的按实例退役**
    （每个实例服满 _DOCLING_RECYCLE_DOCS 篇即丢弃重建），与本函数互补——
    本函数负责「批与批之间把空闲实例的内存还掉」，按实例退役负责「任何实例
    都不允许无限期服役」。两者都只动空闲/已归还的实例。
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


# OCR 渲染倍率档位（OcrOptions.scale 语义 = 「72 DPI × scale」）。
# docling 的 scale 是**转换器级**参数，没有 per-page 接口，因此动态调整的
# 做法是：先用 PyMuPDF 逐页分析分类，再按档位切连续页段，每段各建一个
# 对应 scale 的转换器转换，最后按原页序拼接（见 _analyze_pdf_pages /
# _extract_pdf_by_scale）。分类规则（2026-10 需求）：
#   1. 纯文本文档（文本层达标）              → 默认档
#   2. 扫描件但原始分辨率 < 150 DPI          → 默认档（升采样无意义）
#   3. 其余扫描件：先全量扫描分类，检测到
#      表格的页面                              → 高保档
#   4. 其余页面                                → 默认档
# 注意：这条路径**默认不跑**（_dynamic_scale_enabled=False，全篇固定 2.0），
# 由 config.rag_docling_dynamic_scale 显式开启——实测代价/收益不稳定，
# 详见该配置项与开关处的注释。
_OCR_SCALE_DEFAULT = 2.0   # 144 DPI
_OCR_SCALE_TABLE = 2.5     # 180 DPI，表格页专用（边框/小字号对分辨率敏感）
_SOURCE_DPI_CEILING = 150  # 原始扫描分辨率阈值
_TEXT_LAYER_MIN_CHARS = 100  # 文本层字符数下限：达到即视为文本层页

# 动态 OCR 倍率总开关（2026-10 实测后默认关闭，见 config.py
# rag_docling_dynamic_scale 的代价/收益记录）。关闭时全篇固定 2.0，
# 上面的逐页分析与分段转换整条路径不跑（分析本身 0.2s/5 页不是问题，
# 问题是表格页升档后 OCR 像素 ×1.56 + 多建一个档位转换器）。
_dynamic_scale_enabled = False


def configure_dynamic_scale(enabled: bool) -> None:
    """RAGEngine.__init__ 调用：设置动态 OCR 倍率开关（默认关）。"""
    global _dynamic_scale_enabled
    _dynamic_scale_enabled = bool(enabled)
# 分类用的像素分析渲染 DPI。必须 >=150：扫描件通常按 300 DPI 栅格化，
# 100 DPI 的 3:1 盒式降采样会把目录页的点线（leader dots）平均成连续
# 灰带（(3×0+7×255)/10≈178 < 250 非白阈值），点线页被误判成表格；
# 150 DPI 的 2:1 降采样保留 1.5px 点 + 3.5px 间隙的断续结构，点线不再
# 成带。代价是分析渲染像素量 ×2.25（实测 3 页 1s，可忽略）。
_CLASSIFY_DPI = 150
# 印刷框线判别的灰度/厚度阈值（按 _CLASSIFY_DPI 标定，见 _rule_positions
# 的二要素：非白 + 连续）。
_NONWHITE_LEVEL = 250      # 非白：抓住被降采样成 ~246 浅灰的印刷框线
_MAX_LINE_THICK = 5        # 框线带最大厚度（像素）：文字行/照片通带排除


def _build_converter(scale: float):
    """构建一个新的 docling DocumentConverter（含版本兼容降级）。

    管线降配（版本兼容，全部包在 try/except 中，任何失败都回退默认构造）：
    accelerator 强制 MPS + 限 4 线程。docling 2.x 在 macOS 上默认
    device='auto' → MPS，其缓存分配器只增不放——旧实测同步图多的书时
    RSS 可涨到 30GB+ 且不回落（#116 根因之一）。护栏方案（2026-10 二次
    实验）：布局/表格模型跑 MPS，但每次 convert() 结束在 finally 里
    torch.mps.empty_cache() 归还缓存。实测 60 页扫描书两次运行峰值都是
    2.58GB 完全平坦，且比 CPU 版快 35%（106.5s vs 160.9s）；CPU 版不清
    缓存反而 3.94→4.80GB 持续涨。附带收益：布局挪到 GPU 后，CPU 核让给
    RapidOCR 与嵌入模型，消除同步期的 CPU 争抢（调优器「降档避让」）。

    ``scale`` 是该实例的 OCR 渲染倍率（72 DPI × scale），见 _OCR_SCALE_*
    常量与 _analyze_pdf_pages。构建失败抛异常，由调用方标记 unavailable。
    """
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
            num_threads=4, device=AcceleratorDevice.MPS
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
        # scale 是 OCR 的渲染倍率：页面按「72 DPI × scale」渲染
        # （2.0 → 144 DPI，2.5 → 180 DPI，按页分类动态选取，见
        # _analyze_pdf_pages）。docling 是对每个 OCR 区域**裁剪后**
        # 单独渲染（get_page_image(scale=..., cropbox=ocr_rect)），
        # 降 scale 直接减少送进 ocrmac/rapidocr 的像素量。
        pdf_opts.ocr_options = OcrAutoOptions(
            lang=["zh-Hans", "en-US"],
            scale=scale,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
        logger.info("docling DocumentConverter initialized (PDF 管线 MPS 加速 + 转换后归还缓存 + OCR scale=%.2f, OCR=auto: ocrmac→rapidocr)", scale)
    except Exception as e:
        # docling 版本差异（2.13x 前后多次移动模块路径）：拿不到
        # 降配入口就用默认构造——正确性不受影响，默认 device='auto'
        # 在 macOS 上同样走 MPS，只是少了 num_threads 限线程；
        # 缓存护栏在 extract_markdown 的 finally 里兜底生效。
        logger.info("docling 管线降配不可用（%s），使用默认配置", e)
        converter = DocumentConverter()
    return converter


def _get_converter(scale: float = _OCR_SCALE_DEFAULT):
    """从空闲池取一个转换器，没有则惰性新建（全局最多 _SMALL_FILE_CONCURRENCY
    个并存实例，跨 scale 桶合计也受此上限约束）。返回 None 表示 docling
    不可用（导入/构建失败，已标记 _converter_unavailable）。

    空闲池的取/还在 _pool_lock 下完成；构建较重（import torch + 加载模型），
    刻意放到锁外，避免持池锁时阻塞其它 worker 取还实例。取到的实例同时登记
    进 _checked_out（携带 uses 计数），归还时按这份计数决定退役还是回池。
    """
    global _converter_unavailable, _converter_failed_until
    key = round(float(scale), 3)
    with _pool_lock:
        pool = _converters.get(key)
        if pool:
            entry = pool.pop()
            _checked_out.append(entry)
            return entry[0]
        if _converter_unavailable:
            return None
        if _converter_failed_until > time.monotonic():
            # 瞬时失败冷却期内直接返回 None，不再反复触发重量级构建。
            return None
    # 锁外构建（耗时、且可能 import torch，不应持池锁）。
    try:
        converter = _build_converter(scale)
    except ImportError:
        logger.info("docling not installed; falling back to per-format parsers")
        _converter_unavailable = True
        return None
    except Exception as e:
        # 瞬时失败：只做负缓存，不永久闩锁（理由见 _converter_failed_until 注释）。
        _converter_failed_until = time.monotonic() + _CONV_FAIL_COOLDOWN_S
        logger.warning(
            "failed to init docling converter: %s（%.0fs 内不再重试，之后自动恢复）",
            e, _CONV_FAIL_COOLDOWN_S)
        return None
    # 构建成功：清掉任何残留的负缓存，让后续文件立刻可用。
    _converter_failed_until = 0.0
    with _pool_lock:
        _checked_out.append([converter, 0])
    logger.info("docling DocumentConverter initialized (pooled, scale=%.2f)", scale)
    return converter


def _release_pooled_converter(scale: float, converter) -> None:
    """把用过的转换器还回空闲池（按 scale 分桶），或按服役次数退役。

    **为什么按实例退役（2026-10 修复）**：docling 用 OcrAutoOptions 自行实例化
    的 OCR/ORT 原生缓存按输入形状在同步过程中单调增长，旧设计用「全局单例 +
    每 50 篇重建」把峰值限制在一代内。改成实例池后，池里的实例会被反复复用，
    若只靠全局 50 篇计数清空**空闲**池：正被借出（转换中）的实例躲过那次清理，
    回池后继续服役，缓存跨转换持续累积——同步越久内存越高，直到 swap 把整个
    机器拖慢（「很久没输出」的直接嫌疑）。因此退役判定必须挂在**每个实例**
    自己的 uses 上：服满 _DOCLING_RECYCLE_DOCS 篇就丢弃引用（模型内存随 GC
    回收），下次取用惰性重建新一代。

    只保留每桶最多 _SMALL_FILE_CONCURRENCY 个空闲实例，避免无限增长；key
    可能已被周期回收 pop 掉，用 setdefault 兜底。
    """
    if converter is None:
        return
    key = round(float(scale), 3)
    with _pool_lock:
        entry = None
        for i, e in enumerate(_checked_out):
            if e[0] is converter:      # identity：pydantic 的 __eq__ 又贵又险
                entry = _checked_out.pop(i)
                break
        if entry is None:
            # 未经本池借出（测试 stub 了 _get_converter / 直接构建）：按新实例计。
            entry = [converter, 0]
        entry[1] += 1
        if entry[1] >= _DOCLING_RECYCLE_DOCS:
            logger.info(
                "docling 转换器实例服役满 %d 篇，退役重建（冲刷 OCR/ORT 原生缓存）",
                entry[1],
            )
            return                     # 丢弃引用：不还池，下次惰性重建新一代
        pool = _converters.setdefault(key, [])
        if len(pool) < _SMALL_FILE_CONCURRENCY:
            pool.append(entry)


def configure_concurrency(small_capacity: Optional[int] = None,
                          large_threshold_bytes: Optional[int] = None) -> None:
    """RAGEngine.__init__ 调用：热更新大小闸门参数（不换闸门对象）。

    默认值来自模块常量（小文件 2 路并发 / 20MB 独占）；config 的
    rag_pdf_small_concurrency / rag_pdf_large_threshold_mb 经 engine 转发到
    这里。内存吃紧的机器把 rag_pdf_small_concurrency 设 1 即退回全串行。
    """
    capacity = _SMALL_FILE_CONCURRENCY if small_capacity is None else int(small_capacity)
    threshold = _LARGE_FILE_THRESHOLD if large_threshold_bytes is None \
        else int(large_threshold_bytes)
    capacity = max(1, capacity)
    _conv_gate.configure(threshold, capacity)
    logger.info(
        "docling 转换并发闸门：小文件(< %.1fMB) 最多 %d 路并发，大文件独占",
        threshold / (1024 * 1024), capacity,
    )


def _flush_mps_cache() -> None:
    """把 torch MPS 缓存分配器占着的统一内存归还系统。

    docling 的 MPS 后端（torch Metal）缓存只增不放：单次 convert 后缓存
    一直留着等复用，多本书累积可推到 30GB+（#116 的「逐次累积失控」）。
    每次转换结束调 torch.mps.empty_cache() 归还——实测两次运行峰值完全
    平坦（2.58GB），护栏把累积模式锁死。非 MPS 环境 / 版本差异下静默跳过。
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# 单次 convert 的硬超时（秒）：torch/docling 的转换线程无法被安全杀死，
# 超时后只能抓全线程栈定位卡点（卡在布局模型 / OCR / 表格识别一目了然），
# 由同步层决定放弃该文件。1h 对 60 页扫描书是 30 倍余量，纯兜底。
_CONVERT_TIMEOUT_S = 3600


def _convert_watchdog(path) -> None:
    """convert 硬超时兜底：dump 所有线程栈，便于事后定位卡死点。"""
    try:
        import faulthandler
        import io

        buf = io.StringIO()
        faulthandler.dump_traceback(file=buf)
        logger.error(
            "docling convert 超过 %d 秒未返回（%s），抓取线程栈定位卡点：\n%s",
            _CONVERT_TIMEOUT_S, getattr(path, "name", path), buf.getvalue(),
        )
    except Exception:
        logger.error("docling convert 超过 %d 秒未返回（%s）", _CONVERT_TIMEOUT_S, path)


# ---------------------------------------------------------------------------
# PDF 逐页分析：OCR 渲染倍率（scale）动态分类
# ---------------------------------------------------------------------------
# docling 的 OcrAutoOptions.scale 是转换器级参数，没有 per-page 接口。要实现
# 「表格页用 2.5、其余用 2.0」，只能在 convert 之前用 PyMuPDF 把每页分好类，
# 再按档位切连续页段，每段各用一个对应 scale 的转换器转换，按原页序拼接
# （见 _extract_pdf_by_scale）。本文件内的分析全部只读、线程安全。
#
# 分类规则（2026-10 需求）：
#   1. 文本层达标（>= _TEXT_LAYER_MIN_CHARS）→ 默认档
#   2. 扫描页原始分辨率 < _SOURCE_DPI_CEILING  → 默认档（升采样无意义）
#   3. 其余扫描页：检测到表格（矢量框线或扫描框线）→ 高保档
#   4. 其余                                  → 默认档
class _PageProfile:
    __slots__ = ("index", "has_text", "source_dpi", "has_table", "scale")

    def __init__(self, index: int, has_text: bool,
                 source_dpi: Optional[float], has_table: bool, scale: float):
        self.index = index
        self.has_text = has_text
        self.source_dpi = source_dpi
        self.has_table = has_table
        self.scale = scale


# 分析缓存：按 (路径, 大小, mtime) 键控，避免重复入库同一文件时重复分析。
_analysis_cache: dict = {}
_analysis_cache_lock = threading.Lock()
_ANALYSIS_CACHE_MAX = 32


def _dominant_image_dpi(page) -> Optional[float]:
    """页面主图（面积最大）的原生 DPI；无图像返回 None。

    扫描件的「原始分辨率」就是嵌入位图的分辨率：原生像素数 ÷ 显示尺寸
    （英寸）。取面积最大的那张——页面里混一个小 logo 不应把整页误判成
    低分辨率（旧实现用 min 就会被一枚 72 DPI 图标拖累）。
    """
    try:
        infos = page.get_image_info()
    except Exception:
        return None
    best = None
    for info in infos or ():
        bbox = info.get("bbox")
        width = info.get("width")
        if not bbox or not width:
            continue
        w_pt = float(bbox[2]) - float(bbox[0])
        h_pt = float(bbox[3]) - float(bbox[1])
        area = w_pt * h_pt
        if area <= 0:
            continue
        dpi = float(width) / (w_pt / 72.0)
        if best is None or area > best[0]:
            best = (area, dpi)
    return best[1] if best else None


def _vector_table_lines(page) -> int:
    """原生数字表格的长矢量线数量（数字 PDF 的表格是矢量线，不用渲染）。"""
    try:
        drawings = page.get_drawings()
    except Exception:
        return 0
    count = 0
    w, h = float(page.rect.width), float(page.rect.height)
    for d in drawings or ():
        for item in d.get("items", ()):
            if item and item[0] == "l":
                p1, p2 = item[1], item[2]
                if (abs(p1.x - p2.x) > w * 0.3
                        or abs(p1.y - p2.y) > h * 0.3):
                    count += 1
    return count


def _runs_of_true(mask):
    """布尔序列的连续 True 段 [(start, end), ...]（end 不含）。"""
    import numpy as np

    if not len(mask):
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _longest_run(line) -> int:
    """一维布尔数组的最长连续 True 段长度。"""
    import numpy as np

    if not line.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], line.astype(np.int8), [0]))))
    return int((idx[1::2] - idx[::2]).max())


def _regular_spacing(positions, tol: float = 0.35) -> bool:
    """是否构成「行框线式」等距结构：>=5 条线且 >=4 个间距一致。

    抗两类误报：
      * 装饰性横线（目录页的 Part 分隔线）：数量够但间距双峰杂乱，模态
        间距最多重复 3 次——真实表格行框线是均匀重复的（实测 8 条线 7 个
        一致间距），装饰线做不到；
      * 表格下方无关分隔线：只数一致间距，不要求全部一致。
    """
    if len(positions) < 5:
        return False
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    med = sorted(gaps)[len(gaps) // 2]
    if med <= 0:
        return False
    consistent = sum(1 for g in gaps if abs(g - med) <= tol * med)
    return consistent >= 4


def _rule_positions(arr, nonwhite, bands, axis: int, min_run: float) -> list:
    """候选带中过滤出真正的「印刷框线」位置（带中线）。

    判别维度是「是不是线」而不是「多暗」：
      1. 非白(250) 占比达标 —— 扫描书的印刷框线常被降采样成 246 左右的
         浅灰，按墨迹定的阈值（Otsu/128）会整条漏掉；
      2. 最长连续非白段达标 —— 线是通的，文字行有词间/字间断裂。
    加上调用方的「细带（<=5px）」预过滤，文字行（150 DPI 下 10pt 行高
    ~14px）、照片通带（太厚）、目录点线（断续）全部排除。
    注意**不按墨核深浅判**：黑色印刷框线本身就是墨核，按「无墨核」滤会
    把黑框线表格整页漏掉（实测踩坑）。
    """
    positions = []
    for s, e in bands:
        mid = (s + e) // 2
        if axis == 0:  # 水平带 → 检查该行的横向连续段
            frac = _longest_run(nonwhite[mid, :]) / arr.shape[1]
        else:          # 垂直带 → 检查该列的纵向连续段
            frac = _longest_run(nonwhite[:, mid]) / arr.shape[0]
        if frac >= min_run:
            positions.append(mid)
    return positions


def _page_has_ruled_table(doc, index: int) -> bool:
    """扫描页表格检测：先矢量线（数字表格），再像素投影（扫描表格）。

    像素法判定「印刷框线」（阈值按 _CLASSIFY_DPI(150) 渲染标定）：
    非白占比 >25%（水平）/ >8%（垂直）的细带（<=5 像素厚），且带中线
    连续——细则见 _rule_positions。两种表格形态都认：
      1. 网格型：>=2 条水平框线 + >=1 条垂直框线；
      2. 横线型（中文书常见：只有水平框线无竖线）：>=5 条等距实线横框。

    已知行为（有意接受）：少数书籍的目录页用通页宽分隔线做版式，像素层面
    与表格行框线同构，会被判为表格页 → 该页用 2.5 渲染。代价有界（每本书
    1-3 页、慢 ~25%），且对目录条目 OCR 质量有益，不值得为它加更多启发式。
    """
    import fitz  # noqa: F401  （doc 已由调用方打开）

    page = doc[index]
    # 1) 矢量线：>=4 条长直线即视为表格（原生数字 PDF）。
    if _vector_table_lines(page) >= 4:
        return True
    # 2) 像素投影：只对无文本层的扫描页有意义。
    import numpy as np

    pix = page.get_pixmap(dpi=_CLASSIFY_DPI, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width).astype(np.int16)
    nonwhite = arr < _NONWHITE_LEVEL  # 250：抓住被降采样成浅灰的印刷框线
    h_frac = nonwhite.mean(axis=1)
    v_frac = nonwhite.mean(axis=0)
    h_bands = [b for b in _runs_of_true(h_frac > 0.25) if b[1] - b[0] <= _MAX_LINE_THICK]
    v_bands = [b for b in _runs_of_true(v_frac > 0.08) if b[1] - b[0] <= _MAX_LINE_THICK]
    h_rules = _rule_positions(arr, nonwhite, h_bands, axis=0, min_run=0.25)
    v_rules = _rule_positions(arr, nonwhite, v_bands, axis=1, min_run=0.08)
    # 形态 1：网格型表格。
    if len(h_rules) >= 2 and len(v_rules) >= 1:
        return True
    # 形态 2：横线型表格（>=5 条等距实线横框，无竖线也认；阈值见
    # _regular_spacing——装饰性横线数量够但间距不 uniform）。
    if _regular_spacing(h_rules):
        return True
    return False


def _analyze_pdf_pages(path: Path) -> Optional[list]:
    """逐页分类，返回 _PageProfile 列表；分析失败返回 None（调用方回退整篇）。

    失败即回退：任何异常（加密 PDF、损坏、fitz 缺失）都返回 None，由
    extract_markdown 走「整篇 + 默认档」的老路径，绝不让分析成为新故障点。
    """
    import os

    try:
        st = os.stat(str(path))
        key = (str(path), st.st_size, st.st_mtime)
    except Exception:
        return None
    with _analysis_cache_lock:
        cached = _analysis_cache.get(key)
    if cached is not None:
        return cached

    try:
        import fitz

        src = fitz.open(str(path))
        try:
            if src.is_encrypted and not src.authenticate(""):
                return None
            profiles = []
            for i in range(len(src)):
                page = src[i]
                text_len = len(page.get_text().strip())
                has_text = text_len >= _TEXT_LAYER_MIN_CHARS
                dpi = _dominant_image_dpi(page)
                has_table = False
                if not has_text:
                    # 规则 2：低分辨率扫描件不升采样；其余做表格检测。
                    if dpi is None or dpi >= _SOURCE_DPI_CEILING:
                        has_table = _page_has_ruled_table(src, i)
                scale = _OCR_SCALE_TABLE if has_table else _OCR_SCALE_DEFAULT
                profiles.append(_PageProfile(i, has_text, dpi, has_table, scale))
        finally:
            src.close()
    except Exception as e:
        logger.warning("PDF 逐页分析失败，回退整篇默认档转换（%s）: %s", path.name, e)
        return None

    if not profiles:
        return None
    with _analysis_cache_lock:
        if len(_analysis_cache) >= _ANALYSIS_CACHE_MAX:
            # 简单淘汰：清掉最早一批（同步任务按序处理，LRU 收益有限）。
            for k in list(_analysis_cache.keys())[:_ANALYSIS_CACHE_MAX // 2]:
                _analysis_cache.pop(k, None)
        _analysis_cache[key] = profiles
    n_table = sum(1 for p in profiles if p.has_table)
    logger.info(
        "PDF 逐页分析完成: %s（%d 页，文本层 %d 页，表格页 %d 个 → scale %.1f，"
        "其余 %.1f）",
        path.name, len(profiles),
        sum(1 for p in profiles if p.has_text), n_table,
        _OCR_SCALE_TABLE, _OCR_SCALE_DEFAULT,
    )
    return profiles


def _extract_pdf_by_scale(path: Path, profiles: list) -> Optional[str]:
    """按 scale 档位切连续页段，分段转换后按原页序拼接。

    连续段（contiguous runs）而非逐页调用：同档位的相邻页合成一个子 PDF 一次
    转换，绝大多数文档只有 1-2 个段，调用次数与整篇转换持平。

    按档位**分组顺序处理、转完即释放该档转换器**：每档一个 DocumentConverter
    = 一整套布局 + TableFormer 模型（两档并存白占 ~1.9GB 统一内存）。分组不
    改变拼接顺序——各段 markdown 按段首页码重排，结果仍是原页序。
    """
    runs: list = []
    for p in profiles:
        if runs and runs[-1][0] == p.scale:
            runs[-1][1].append(p.index)
        else:
            runs.append((p.scale, [p.index]))

    parts: dict = {}  # 段首页码 -> markdown（按段首页码排序即原页序）
    result = None
    try:
        import os
        import fitz
        import tempfile

        src = fitz.open(str(path))
        try:
            tmpdir = tempfile.mkdtemp(prefix="docling_seg_")
            try:
                for scale in sorted({s for s, _ in runs}, reverse=True):
                    for s, idxs in [r for r in runs if r[0] == scale]:
                        sub = fitz.open()
                        try:
                            sub.insert_pdf(src, from_page=idxs[0], to_page=idxs[-1])
                            tmp = os.path.join(
                                tmpdir, f"seg_{scale:.2f}_{idxs[0]}_{idxs[-1]}.pdf")
                            sub.save(tmp)
                        finally:
                            # 必须 try/finally：insert_pdf / save 在加密、损坏或
                            # 超大 PDF 上会直接抛，旧实现把 sub.close() 放在
                            # 正常路径末尾 → 每个失败文件都泄漏一个 fitz.Document
                            # （文件句柄 + 整段子文档的解码缓存）。
                            sub.close()
                        md = _convert_with_scale(tmp, scale)
                        if md and md.strip():
                            parts[idxs[0]] = md
                    # 该档全部转完，释放其转换器（只放这一把钥匙），把布局/
                    # TableFormer 模型占的统一内存还给系统；下一本书用到同档
                    # 位时惰性重建（~2s 模型加载 + ~10s 管线初始化）。
                    with _pool_lock:
                        _converters.pop(round(float(scale), 3), None)
            finally:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
        finally:
            src.close()
    except Exception as e:
        logger.warning("PDF 分段转换失败（%s），回退整篇转换: %s", path.name, e)
        return None

    if not parts:
        return None
    if len(runs) > 1:
        logger.info("PDF 分段转换完成: %s → %d 段（scale 分布 %s），转换器已按档释放",
                    path.name, len(runs),
                    "+".join(f"{s:.1f}×{len(ix)}" for s, ix in runs))
    return "\n\n".join(parts[k] for k in sorted(parts))


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

    # 预置为 None：finally 里无条件解引用，异常发生在赋值之前也不会
    # UnboundLocalError（旧实现把释放只放在成功路径）。
    result = None
    doc = None
    markdown = None
    watchdog = threading.Timer(_CONVERT_TIMEOUT_S, _convert_watchdog, args=(path,))
    watchdog.daemon = True
    watchdog.start()
    try:
        # PDF 走动态 scale（**需 config.rag_docling_dynamic_scale 显式开启**，
        # 默认关闭全篇固定 2.0）：先逐页分析分类，再按档位切连续段分段
        # 转换、按序拼接。分析或分段任一失败都回退下面的「整篇 + 默认档」
        # 老路径——动态调整永远是优化，绝不让它成为新故障点。
        if suffix == ".pdf" and _dynamic_scale_enabled:
            profiles = _analyze_pdf_pages(path)
            if profiles is not None:
                markdown = _extract_pdf_by_scale(path, profiles)
                if markdown and markdown.strip():
                    return markdown
                logger.warning("PDF 分段转换无输出，回退整篇转换: %s", path.name)
            markdown = None

        # DocumentConverter 没有声明线程安全，而 ingest 用 4 个 worker 线程
        # 并行解析。旧实现只对「创建单例」加锁，convert() 本身完全不设防，
        # 并发转换会互相踩踏（状态污染/崩溃）；后来改成「锁外取实例、锁内
        # 转换」，又与周期回收藏出两代并存的竞态（见 _convert_with_scale）。
        # 现在取实例 + 转换 + 导出在同一把锁内原子完成。
        markdown = _convert_with_scale(str(path), _OCR_SCALE_DEFAULT)
        if not markdown or not markdown.strip():
            logger.warning("docling extracted empty markdown from %s", path.name)
            return None
        return markdown
    except Exception as e:
        # Never let a docling failure abort ingestion — caller falls back.
        logger.warning("docling conversion failed for %s: %s", path.name, e)
        return None
    finally:
        watchdog.cancel()
        # 及时释放大对象引用，避免整棵文档树在切片前一直常驻。
        # ConversionResult 持有整棵文档树 + 每页位图（数十~数百 MB）；
        # 旧实现只在成功路径 del，export_to_markdown() 抛异常时它活到函数
        # 返回才被回收，异常密集时是实打实的常驻。
        result = None
        doc = None
        # 每次转换后归还 MPS 缓存（护栏，见 _flush_mps_cache 注释）。
        # 放在 finally：成功与异常路径都清，防止异常密集时缓存照样累积。
        # 有其它转换在飞时跳过：empty_cache 只释放分配器里「未占用」的缓存块，
        # 此时另一路转换可能正要重新分配——清了也是立刻再分配，纯属抖动，且
        # 与另一路的分配动作争抢分配器锁；顺序执行（大文件独占的常见形态，
        # 也是唯一真正需要归还的时机）时才清。最后一班离开时 active()==0，
        # 缓存照样会被归还，护栏不漏。
        if _conv_gate.active() == 0:
            _flush_mps_cache()
        # 周期性冲刷 docling 原生缓存（见 _maybe_recycle_converter 注释）。
        # _convert_with_scale 的 finally 已先 _conv_gate.release，这里不再持
        # 任何闸门/池锁，回收只清空闲池。
        try:
            _maybe_recycle_converter()
        except Exception as e:  # 绝不能让回收逻辑影响解析结果
            logger.debug("docling 周期回收失败（忽略）: %s", e)
