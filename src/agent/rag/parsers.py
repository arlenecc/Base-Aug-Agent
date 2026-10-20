"""Document parsers for Word, Excel, PPT, PDF, EPUB, MOBI, AZW3 and more.

Each parser accepts a file path and returns extracted text. Lazy imports keep
optional dependencies optional — only the formats you actually use need to be
installed.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局 OCR 引擎（单例）+ 并发信号量
# ---------------------------------------------------------------------------
# RapidOCR ONNX Runtime 单实例约 500MB 内存。如果每个 worker 各加载一份，
# 4 worker 同时遇到图片 PDF 时会占用 ~2GB 仅仅用于 OCR 模型。
# 改为全局单例：所有 worker 共享同一个引擎实例，通过信号量串行执行
# （RapidOCR 可能非线程安全，串行调用 engine(img) 避免竞态）。
_OCR_SEMAPHORE = threading.Semaphore(1)
_OCR_ENGINE = None  # 全局单例；None 表示尚未尝试加载
_OCR_ENGINE_LOCK = threading.Lock()

# OCR 引擎周期性重建页数。CoreML/ORT 的原生缓存按输入形状保留（实测真实
# 扫描页 ~35MB/页，#119：400 页扫描书滞留 ~14GB，进程退出前不释放，最终
# 被系统压缩到 17GB 直至 50GB 崩溃）。每 N 页强制重建引擎冲刷全部原生
# 缓存：#130 系统扫描（N=10..100，单线程真实 200dpi 页）证实热重建仅
# 1-2s/次、峰值足迹在各 N 下持平 ~5.2GB（压缩器吸收缓存增长），N 对速度
# 影响仅 1-3%/400 页书。取 50：单代水位 ~1.8GB（#123 后单代存活，4 worker
# 聚合约同量级），重建开销摊薄到 ~0.5%。合成小图触发不了该积累，只有真实
# 200dpi 大页 + 密集文字框才会——不要因为「合成实验没复现」而移除。
_OCR_RECYCLE_PAGES = 50

# OCR 输入归一化：渲染 DPI 上限 + 长边像素上限（两道独立约束，缺一不可）。
#   1. _OCR_DPI：渲染分辨率不超过 150 DPI。200+ DPI 扫描页渲染出的大位图会
#      让 RapidOCR 的 CoreML/ORT 原生缓存按形状暴涨（见 _OCR_RECYCLE_PAGES
#      注释），且对识别率提升边际递减；150 DPI 已足以覆盖常规印刷/扫描文字。
#   2. _OCR_MAX_LONG_EDGE：长边不超过 1500px。A4 页在 150 DPI 下长边约 1754px，
#      超出部分对识别无增益，却成倍推高 OCR 计算量与原生缓存占用，等比
#      缩放回 1500px 即可（见 _ocr_downscale_long_edge）。
_OCR_DPI = 150
_OCR_MAX_LONG_EDGE = 1500

# 锐化 / 二值化开关（默认全开）。放在这里是为了留退路：二值化会丢掉反锯齿
# 灰边，个别印刷质量差的扫描件若识别率反而下降，把 _OCR_BINARIZE 置 False
# 即可退回灰度管线，不必改代码结构。
_OCR_SHARPEN = True
_OCR_BINARIZE = True

# MuPDF store 的页级收缩间隔。store 缓存解码后的嵌入图（每张满页扫描图解码
# 后 ~10MB+），旧实现只在**全书结束**时收缩一次（_extract_pdf 的 finally），
# 数百页扫描书在单文件内的峰值就冲到 GB 级，再 ×并发 worker 数——「同步跑
# 一会就崩」的来源之一。每 N 页冲刷一次把峰值压在 ~N×10MB。
# store_shrink(percent) 的语义是「释放百分之多少」（fitz.TOOLS.store_shrink
# 的 docstring: "Free 'percent' of current store size"），100 = 全部释放。
_STORE_SHRINK_PAGES = 25

# ---------------------------------------------------------------------------
# Format → extension mapping
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".docx",
    ".xlsx",
    ".pptx",
    ".pdf",
    ".epub",
    ".mobi", ".azw3", ".azw",
    ".txt", ".md", ".markdown", ".rst",
    ".csv", ".tsv",
    ".html", ".htm",
    ".tex", ".latex",
}
# 旧版二进制格式（.doc/.xls/.ppt）不在支持列表：python-docx/openpyxl/
# python-pptx 与 docling 均只支持 OOXML，解析必然失败（extract_text 对它们
# 给出明确的「另存为」提示）。保留在支持列表里只会让用户误以为可用。

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_text(filepath: str) -> str:
    """Extract text from a document. Returns the full text as a string."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    suffix = path.suffix.lower()

    # 文件类型→解析器映射（用于日志）
    _PARSER_LABELS = {
        ".docx": "Word 文档", ".doc": "Word 文档",
        ".xlsx": "Excel 表格", ".xls": "Excel 表格",
        ".pptx": "PPT 幻灯片", ".ppt": "PPT 幻灯片",
        ".pdf": "PDF 文档",
        ".epub": "EPUB 电子书",
        ".mobi": "MOBI 电子书", ".azw3": "AZW3 电子书", ".azw": "AZW 电子书",
        ".txt": "纯文本", ".md": "Markdown", ".markdown": "Markdown", ".rst": "reStructuredText",
        ".csv": "CSV 表格", ".tsv": "TSV 表格",
        ".html": "HTML 网页", ".htm": "HTML 网页",
        ".tex": "LaTeX 文档", ".latex": "LaTeX 文档",
    }
    label = _PARSER_LABELS.get(suffix, suffix)
    logger.debug("    解析文件类型: %s (%s)", path.name, label)

    if suffix in (".docx", ".doc"):
        if suffix == ".doc":
            # python-docx 只支持 OOXML（.docx）；旧版 OLE2 二进制 .doc 传进来
            # 100% 抛异常（docling 同样不支持 legacy 格式），之前表现为一条
            # 难懂的库报错。直接给出可行动的提示。
            raise ValueError(
                "旧版 .doc 二进制格式不受支持（python-docx/docling 均只支持 .docx），"
                "请先用 Word/WPS 另存为 .docx 后重新同步"
            )
        return _extract_docx(path)
    elif suffix in (".xlsx", ".xls"):
        if suffix == ".xls":
            # openpyxl 明确不支持旧版 .xls（BIFF），必然抛
            # "does not support the old .xls file format"。
            raise ValueError(
                "旧版 .xls 二进制格式不受支持（openpyxl 只支持 .xlsx），"
                "请先用 Excel/WPS 另存为 .xlsx 后重新同步"
            )
        return _extract_xlsx(path)
    elif suffix in (".pptx", ".ppt"):
        if suffix == ".ppt":
            # python-pptx 只支持 OOXML（.pptx），同上。
            raise ValueError(
                "旧版 .ppt 二进制格式不受支持（python-pptx 只支持 .pptx），"
                "请先用 PowerPoint/WPS 另存为 .pptx 后重新同步"
            )
        return _extract_pptx(path)
    elif suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".epub":
        return _extract_epub(path)
    elif suffix in (".mobi", ".azw3", ".azw"):
        return _extract_mobi(path)
    elif suffix in (".txt", ".md", ".markdown", ".rst"):
        return _extract_text(path)
    elif suffix in (".csv", ".tsv"):
        return _extract_csv(path)
    elif suffix in (".html", ".htm"):
        return _extract_html(path)
    elif suffix in (".tex", ".latex"):
        # LaTeX 没有 Python 侧轻量解析器，只走 pandoc（engine 管线在 pandoc
        # 可用时已优先路由到这里；extract_text 直调是 pandoc 缺失的兜底）。
        from .pandoc_parser import convert_document, is_pandoc_available
        md = convert_document(str(path))
        if md and md.strip():
            return md
        if is_pandoc_available():
            raise ValueError(
                "LaTeX → Markdown 转换失败（pandoc 已安装但未产出文本），"
                "请确认文件是完整的 LaTeX 源文档"
            )
        raise ValueError(
            "LaTeX 解析需要 pandoc（未检测到）。请安装：brew install pandoc"
        )
    else:
        raise ValueError(f"Unsupported format: {suffix}")


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for Word files. pip install python-docx")

    doc = Document(str(path))
    parts: List[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    # Also extract tables
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _extract_xlsx(path: Path) -> str:
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for Excel files. pip install openpyxl")

    wb = openpyxl.load_workbook(str(path), data_only=True)
    parts: List[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"# Sheet: {sheet_name}")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(c.strip() for c in cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("python-pptx is required for PPT files. pip install python-pptx")

    prs = Presentation(str(path))
    parts: List[str] = []
    for i, slide in enumerate(prs.slides, 1):
        slide_parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        slide_parts.append(t)
            if shape.has_table:
                table = shape.table
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    slide_parts.append(" | ".join(cells))
        if slide_parts:
            parts.append(f"## Slide {i}\n\n" + "\n".join(slide_parts))
    return "\n\n".join(parts)


def _pixmap_is_blank(pix) -> bool:
    """检测渲染后的整页位图是否为纯色/近纯色（无有效内容）。

    均匀采样至多 8192 字节取 min/max：极差 ≤ 12 视为近纯色。覆盖两类
    场景：页面上只有纯色占位图/装饰条；扫描件里的空白纸页（整页白纸
    带轻微压缩噪点）。这些页送 OCR 必然返回 "text detection result is
    empty"，每页白白消耗 2-5 秒。极差阈值 12 足以容纳 JPEG/扫描噪点，
    而任何含文字的页都有大量暗像素，极差远超此值。
    """
    samples = pix.samples
    if not samples:
        return True
    step = max(1, len(samples) // 8192)
    data = samples[::step]
    return (max(data) - min(data)) <= 12


def _ocr_normalize(img):
    """归一化核心（可独立测试）：暗色主题自动反色 + 对比度拉伸。

    RapidOCR 的 det/rec 模型在白底黑字、高对比度样本上训练，对两类页面
    检测率骤降（真实电子书里「一大片页面检测不到文字」的根因）：
      1. 暗色主题页（黑底白字/灰字）——模型几乎检不出任何文本框；
      2. 低对比度/模糊文字页（低分辨率扫描放大、JPEG 压缩、浅灰字）——
         文本框被当成背景。

    管线（两步独立计算分位数——反色会反转明暗分布，若复用反色前的
    p2/p98 做拉伸，暗主题页的 p98 仍落在黑底范围，拉伸会把背景噪声
    放大十余倍、文字被噪声海淹没）：
      a) 自动反色：页面整体偏暗（p50 < 100 且 p95 < 160，即 95% 像素都
         是暗色调，符合「深色背景 + 少量浅色文字」的特征）时反色为白底
         深字。插图占比大的正常页 p95 会远超 160，不会误触发。
      b) 对比度拉伸：用**当前图像**的 p2-p98 分位线性映射到 [0, 255]，
         恢复低对比度页面的文字-背景差异。极小极差（< 8，纯色/噪点页）
         跳过，避免放大噪点。
    """
    import numpy as np

    p50, p95 = np.percentile(img, (50, 95))
    if p50 < 100 and p95 < 160:
        img = 255 - img
    p2, p98 = np.percentile(img, (2, 98))
    if p98 - p2 > 8:
        lo, hi = p2, p98
        img = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo),
                      0, 255).astype(np.uint8)
    return img


def _ocr_downscale_long_edge(img):
    """长边超过 _OCR_MAX_LONG_EDGE(1500px) 时等比缩放回 1500px。

    A4 页在 150 DPI 下长边约 1754px，超出部分对识别无增益，却成倍推高
    OCR 计算量与 CoreML/ORT 原生缓存占用。用 LANCZOS 高质量下采样，
    仅在确实超限时触发（绝大多数页面长边 < 1500px，零开销直通）。
    """
    import numpy as np
    from PIL import Image

    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= _OCR_MAX_LONG_EDGE:
        return img
    scale = _OCR_MAX_LONG_EDGE / float(long_edge)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return np.asarray(
        Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.uint8,
    )


def _ocr_sharpen(img):
    """锐化：轻度 USM（Unsharp Mask），让文字边缘更利落。

    参数刻意保守（radius=1 / percent=60 / threshold=3）：锐化过猛会把 JPEG 与
    扫描噪点放大成伪笔画，反而拉低识别率——与 _ocr_normalize 里「极差过小就
    不做对比度拉伸」是同一条原则。threshold=3 让纯色背景等平滑区域不参与
    锐化，只作用于真正的边缘。
    """
    import numpy as np
    from PIL import Image, ImageFilter

    return np.asarray(
        Image.fromarray(img).filter(
            ImageFilter.UnsharpMask(radius=1, percent=60, threshold=3)
        ),
        dtype=np.uint8,
    )


def _ocr_binarize(img):
    """二值化：按**当前图像**的灰度直方图用 Otsu 自动定阈值。

    为什么不用固定阈值：不同扫描件的黑白场差异极大，固定阈值会把浅色页整页
    判成背景（全白、一个字都没有），深色页整页判成前景（全黑）。Otsu 最大化
    类间方差自动取直方图谷底，对归一化后的文档页稳定。

    为什么此时二值化是安全的：_ocr_normalize 已做过 p2-p98 对比度拉伸，前景与
    背景双峰已经分开，阈值落在谷底。这一步的收益是去掉压缩噪点与反锯齿灰边，
    让 det 模型的文本框更紧。

    退化保护：阈值极端（≤8 或 ≥247，整页几乎全黑/全白）说明直方图没有有效
    双峰，二值化只会把整页抹平——此时保留灰度原样返回。
    """
    import numpy as np

    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return img
    omega = np.cumsum(hist) / total
    levels = np.arange(256, dtype=np.float64)
    mu = np.cumsum(hist * levels) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    thr = int(np.argmax(sigma_b))
    if thr <= 8 or thr >= 247:
        return img
    return np.where(img > thr, np.uint8(255), np.uint8(0))


def _ocr_preprocess(pix):
    """OCR 输入预处理。管线顺序即代码顺序：

        灰度解码 → 长边 Lanczos 缩放到 ≤1500px → 归一化（暗色反色 + 对比度
        拉伸）→ 锐化 → 二值化

    DPI 与灰度在**渲染侧**保证（_extract_pdf 用 _OCR_DPI=150 +
    colorspace=fitz.csGRAY 渲染），这里只做像素级处理。

    返回 ndarray（不编码 PNG）：RapidOCR 接受 ndarray 输入，省掉 PNG
    编码 + 引擎内部解码两步开销。
    """
    import numpy as np

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n != 1:  # 防御：非灰度渲染时按行内多通道扁平处理
        img = img.reshape(-1, pix.n).mean(axis=1).astype(np.uint8)
    img = img.reshape(pix.height, pix.width)
    return _ocr_preprocess_array(img)


def _ocr_preprocess_array(img):
    """对**灰度 ndarray** 执行完整 OCR 预处理管线。

    与 PDF 图片页共用同一套归一化：长边 Lanczos 缩放 → 归一化（暗色反色 +
    对比度拉伸）→ 锐化 → 二值化。扫描版 epub 的图片 OCR（pandoc_parser）
    复用本函数，保证两条通路的预处理完全一致。
    """
    img = _ocr_downscale_long_edge(img)
    img = _ocr_normalize(img)
    if _OCR_SHARPEN:
        img = _ocr_sharpen(img)
    if _OCR_BINARIZE:
        img = _ocr_binarize(img)
    return img


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF. Auto-detects text vs image-based pages.

    Strategy:
    1. Text-based pages: extract text directly via PyMuPDF get_text("text").
    2. Image-based pages (no text layer, has embedded images): OCR —— 走
       :func:`ocr_recognize_array`，ocrmac(Apple Vision) 优先、不可用时
       回退 RapidOCR，与主引擎 docling 的 OcrAutoOptions 同策略。
    3. Mixed PDFs: text pages extracted directly, image pages OCR'd.

    If no OCR backend is available and the PDF contains image-based pages,
    a warning is logged and the image pages are skipped — but the text
    pages are still extracted.
    """
    path = Path(path)  # 兼容 str 和 Path 两种入参
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF files. pip install PyMuPDF")

    doc = fitz.open(str(path))
    try:
        # 加密 PDF：没有密码就取不到任何文本，旧行为是整份文件静默产出空
        # 内容并计入「提取失败」。用空密码尝试解锁，能解就继续正常解析。
        if doc.needs_pass:
            if not doc.authenticate(""):
                raise ValueError("PDF 已加密且无法用空密码解锁")

        total_pages = len(doc)
        parts: List[str] = []
        # ocrmac / RapidOCR 都不可用标记：触发后整份文件的剩余图页都不再
        # 渲染，避免逐页白做。纯文本书不进 OCR 分支时为 False。
        ocr_unavailable = False

        # 统计各类型页面数量，用于最终日志
        text_page_count = 0
        image_page_count = 0
        ocr_page_count = 0
        ocr_skipped_count = 0
        blank_page_count = 0

        for page_num in range(total_pages):
            # 页级 store 收缩：解码缓存随页数累积，不能等全书结束才冲刷
            # （见 _STORE_SHRINK_PAGES 注释）。放在循环开头而非末尾：循环体
            # 内有多处 continue，放在末尾会被跳过。
            if page_num and page_num % _STORE_SHRINK_PAGES == 0:
                try:
                    fitz.TOOLS.store_shrink(100)
                except Exception:
                    pass
            page = doc[page_num]
            text = page.get_text("text").strip()

            if text:
                # 文字版页面：直接提取文本，不需要 OCR
                parts.append(text)
                text_page_count += 1
                continue

            # 无文字层 → 可能是图片版或空白页。检查是否含嵌入图片。
            images = page.get_images(full=True)
            if not images:
                # 既无文字也无图片 → 空白页，跳过
                blank_page_count += 1
                continue

            # 图片版页面 → 需要 OCR
            image_page_count += 1

            if ocr_unavailable:
                ocr_skipped_count += 1
                continue

            with _OCR_SEMAPHORE:
                # 后端探测放在渲染之前：ocrmac 与 RapidOCR 都没有时整份文件
                # 的图页都识别不了，早退可以省掉全部渲染开销。
                if not _ocr_backend_available():
                    ocr_unavailable = True
                    ocr_skipped_count += 1
                    continue
                # 引擎单例在 ocr_recognize_array 内部按需取用（走 ocrmac 时
                # 根本不会加载 RapidOCR，省掉 ~500MB）。每页从进程级单例取
                # 引擎，页末即释放局部引用（#123）。
                # 旧实现把引擎缓存在 worker 的循环级局部变量里：周期回收
                # （release + 全局置 None）只影响触发回收的 worker，其它
                # worker 仍抱着旧引擎继续用——最多 4 代引擎的原生缓存并存
                # （每代 ~1-2GB），回收形同虚设，实测内存仍涨到 16GB。
                # 每页取用后，任意时刻至多一代引擎存活，回收立即全局生效。
                # 灰度渲染：OCR 模型输入本来就是灰度语义，且体积减半、
                # 便于后续的暗页反色/对比度拉伸归一化（_ocr_preprocess）。
                # DPI 上限 _OCR_DPI(150)；若原页在 150 DPI 下长边仍 > 1500px，
                # _ocr_downscale_long_edge 会等比缩回到 1500px。
                pix = page.get_pixmap(dpi=_OCR_DPI, colorspace=fitz.csGRAY)
                try:
                    if _pixmap_is_blank(pix):
                        # 渲染后确认的纯色/近纯色页（纯色占位图、扫描件的
                        # 空白纸页）：OCR 检测必然为空，直接跳过，省下每页
                        # 2-5 秒的 OCR（真实日志中单文档可浪费 1 分钟以上）。
                        blank_page_count += 1
                        continue
                    img = _ocr_preprocess(pix)
                finally:
                    # Pixmap 持有整页位图（数 MB），显式释放避免
                    # 多页循环时峰值内存持续累积。
                    pix = None

                ocr_text = ocr_recognize_array(img, page_num + 1, path.name)

            if ocr_text:
                parts.append(ocr_text)
                ocr_page_count += 1
            else:
                # OCR 返回空结果（可能是图片质量太差或引擎异常）
                ocr_skipped_count += 1

            # 周期性重建引擎：冲刷 CoreML/ORT 按形状保留的原生缓存（见
            # _OCR_RECYCLE_PAGES 注释）。全局置 None 后，所有 worker 的
            # 下一页都会重新取用/重建——旧引擎随各 worker 页末的引用释放
            # 而销毁，原生缓存真正被冲刷。
            if (ocr_page_count + ocr_skipped_count) % _OCR_RECYCLE_PAGES == 0:
                release_ocr_engine()
                logger.info("PDF '%s': OCR 引擎已周期重建（每 %d 页，冲刷原生缓存）",
                            path.name, _OCR_RECYCLE_PAGES)
    finally:
        # 必须放在 finally：中途抛异常（加密 PDF、损坏文件、OCR 崩溃）时
        # 旧实现直接跳过 doc.close()，泄漏文件句柄与整份文档的解码缓存。
        doc.close()
        # MuPDF 的 store 会缓存解码后的嵌入图（每张满页扫描图解码后
        # ~10MB+，数百页累计可观且进程内常驻）；全书解析完立即收缩并
        # 回收 Python 侧临时对象。store 是进程级全局的，收缩会影响其它
        # worker 正在打开的文档——只是缓存失效（重解码），不影响正确性。
        try:
            fitz.TOOLS.store_shrink(100)
        except Exception:
            pass
        gc.collect()

    # 汇总日志：清晰报告各类型页面的处理情况
    detail_parts = [f"text={text_page_count}"]
    if image_page_count > 0:
        detail_parts.append(f"image={image_page_count}(OCR={ocr_page_count})")
    if ocr_skipped_count > 0:
        detail_parts.append(f"OCR-skipped={ocr_skipped_count}")
    if blank_page_count > 0:
        detail_parts.append(f"blank={blank_page_count}")

    # ---- 全图片版 PDF 且 OCR 不可用 → 抛出明确错误 ----
    if not parts and image_page_count > 0 and ocr_unavailable:
        msg = (
            f"PDF '{path.name}' 是图片版（{total_pages} 页全部为扫描图片），"
            f"但没有任何可用的 OCR 后端（ocrmac / rapidocr），无法进行识别。"
            f"请安装: pip install ocrmac（或 pip install rapidocr-onnxruntime）"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if not parts:
        logger.warning(
            "PDF '%s': no text extracted from %d pages "
            "(text=%d, image=%d, blank=%d)",
            path.name, total_pages, text_page_count, image_page_count, blank_page_count,
        )
        return ""

    if ocr_skipped_count > 0 and ocr_unavailable:
        # 混合 PDF：部分图片页因 OCR 不可用被跳过
        logger.warning(
            "PDF '%s': %d/%d pages are image-based but no OCR backend "
            "(ocrmac / rapidocr) is available. "
            "Image pages were SKIPPED — only text pages were extracted (%d chars). "
            "Install with: pip install ocrmac",
            path.name, ocr_skipped_count, total_pages, sum(len(p) for p in parts),
        )
    elif image_page_count > 0:
        logger.info(
            "PDF '%s': %d pages processed (%s)",
            path.name, total_pages, ", ".join(detail_parts),
        )
    else:
        logger.info(
            "PDF '%s': %d pages, all text-based (%d chars extracted)",
            path.name, total_pages, sum(len(p) for p in parts),
        )

    return "\n\n".join(parts)


def _is_apple_silicon() -> bool:
    """当前是否为 Apple Silicon（M 系列）Mac。

    arm64 macOS 即 Apple Silicon；Intel Mac 返回 False，继续用 CPU EP。
    """
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _get_ocr_engine():
    """Lazy-load RapidOCR engine as a global singleton.

    Returns None if not installed. All workers share the same engine instance
    (~500MB) — without this, 4 workers loading 4 instances would OOM. The
    _OCR_SEMAPHORE serializes actual OCR calls so the shared engine is only
    invoked by one thread at a time.

    引擎选择：优先 rapidocr 3.x（支持 CoreML EP、模型目录与 docling 共享、
    懒加载省内存），未安装再回退旧版 rapidocr-onnxruntime 1.x（CPU EP）。
    Apple Silicon（M 系列）上 3.x 启用 onnxruntime 的 CoreML EP（ANE+GPU
    推理，实测比 CPU EP 快约 1.5 倍且输出一致）；CoreML 初始化失败自动
    回退 CPU EP。
    """
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE
        try:
            params = None
            try:
                from rapidocr import RapidOCR  # 3.x
                if _is_apple_silicon():
                    params = {"EngineConfig.onnxruntime.use_coreml": True}
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR  # 1.x 回退
            if params:
                try:
                    _OCR_ENGINE = RapidOCR(params=params)
                except Exception as e:
                    logger.info(
                        "RapidOCR CoreML EP init failed (%s); using CPU EP",
                        type(e).__name__)
                    _OCR_ENGINE = RapidOCR()
            else:
                _OCR_ENGINE = RapidOCR()
            # 预热一次（空图）：提前加载模型并探测实际生效的 EP。
            # 模型本来马上就会被用到，预热同时消除首页 OCR 的加载延迟。
            try:
                import numpy as np
                _OCR_ENGINE(np.full((64, 64), 255, dtype=np.uint8))
                inner = getattr(getattr(_OCR_ENGINE, "text_det", None),
                                "session", None)
                inner = getattr(inner, "session", inner)
                logger.info("RapidOCR engine loaded, EP: %s",
                            inner.get_providers())
            except Exception:
                logger.info("RapidOCR engine loaded (global singleton)")
            return _OCR_ENGINE
        except ImportError:
            logger.warning(
                "rapidocr is required for OCR on image-based PDFs. "
                "pip install rapidocr"
            )
            return None
        except Exception as e:
            logger.warning("Failed to init RapidOCR: %s", e)
            return None


class _OcrmacUnavailable(Exception):
    """ocrmac 不可用（未安装 / 运行期异常）→ 调用方据此回退 RapidOCR。

    注意：与「ocrmac 跑通了但这一页没识别到文字」**区分开**——后者是正常
    情况（整页版画、插图、纯空白），不应触发 RapidOCR 的 500MB 引擎加载。
    """


def _ocrmac_recognize(img) -> str:
    """用 ocrmac（Apple Vision）识别**已预处理**的灰度 ndarray。

    返回空串表示「跑通了但本页没识别到文字」（插图/空白页属于此类，直接
    跳过，不回退）。ocrmac 未安装或运行期异常则抛 :class:`_OcrmacUnavailable`，
    由 :func:`ocr_recognize_array` 回退 RapidOCR —— 与 docling 的
    ``AutoOcrModel`` 同策略：Vision 是系统级加速，中文识别的质量与速度都
    优于 RapidOCR。
    """
    from PIL import Image

    pil = Image.fromarray(img)
    try:
        from ocrmac import ocrmac

        results = ocrmac.OCR(
            pil,
            framework="vision",
            recognition_level="accurate",
            language_preference=["zh-Hans", "en-US"],
            detail=False,
        ).recognize()
    except Exception as e:
        raise _OcrmacUnavailable(e)

    parts: List[str] = []
    for item in results or []:
        # detail=False → (text, confidence)；个别版本仍返回三元组。
        text = item[0] if isinstance(item, (tuple, list)) else item
        if text and str(text).strip():
            parts.append(str(text).strip())
    return "\n".join(parts)


def ocr_recognize_array(img, page_num: int = 0, filename: str = "") -> str:
    """识别**已归一化**的灰度 ndarray：ocrmac(Apple Vision) 优先，回退 RapidOCR。

    全项目唯一的「图 → 文字」入口：PyMuPDF 兜底 PDF 页（:func:`_extract_pdf`）
    与扫描版 epub 页（``pandoc_parser.ocr_epub_images``）都走这里，保证两条
    通路的预处理（:func:`_ocr_preprocess_array`）与引擎选择策略完全一致。

    回退触发条件严格限于「ocrmac 不可用」——即 ocrmac 未安装或运行期异常
    （:class:`_OcrmacUnavailable`）。ocrmac 跑通但本页无文字（插图/空白页）
    直接返回空，绝不为了这几页拉起 RapidOCR 的 500MB 引擎。这也符合「优先
    ocrmac，不可用再走 rapidocr」的语义：页上没字 ≠ ocrmac 不可用。

    ``page_num`` / ``filename`` 仅用于 RapidOCR 失败时的日志定位。
    """
    try:
        return _ocrmac_recognize(img)
    except _OcrmacUnavailable as e:
        logger.debug("ocrmac 不可用，回退 rapidocr: %s", e)
        ocr_engine = _get_ocr_engine()
        if ocr_engine is None:
            return ""
        return _ocr_image_cached(ocr_engine, img, page_num, filename)


def _ocr_backend_available() -> bool:
    """是否存在任一可用 OCR 后端（ocrmac 优先，其次 RapidOCR）。

    用于「整份文件都不必渲染」的早退判断：两个后端都没有时，提前标记
    ``ocr_unavailable``，省掉逐页渲染的 CPU 与内存。
    """
    try:
        import ocrmac  # noqa: F401  # 存在即视为可用（识别期异常会回退）
        return True
    except Exception:
        pass
    return _get_ocr_engine() is not None


def release_ocr_engine() -> None:
    """Release the global OCR singleton to reclaim its ~500MB memory.

    The RapidOCR engine holds an ONNX Runtime inference session (~500MB of
    native heap) that Python GC cannot reclaim promptly.  GUI apps that OCR
    image-based PDFs would otherwise hold this memory until process exit.
    Call this from RAGEngine.close() after a sync finishes; the engine is
    lazily re-created on the next OCR-requiring ingestion.
    """
    global _OCR_ENGINE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            # RapidOCR wraps an onnxruntime InferenceSession.  Drop the
            # reference so the session's C++ heap is freed on GC.
            _OCR_ENGINE = None
            logger.info("RapidOCR engine released (memory reclaimed)")


def _ocr_image_cached(engine, img, page_num: int, filename: str) -> str:
    """Run OCR using a pre-loaded engine instance.

    ``img`` 是归一化后的灰度 ndarray（见 _ocr_preprocess）；RapidOCR 的
    LoadImage 对 ndarray 直接走 cv2 转换，无需再编码/解码。
    兼容两代 API 的返回：3.x 返回 RapidOCROutput（行文本在 .txts）；
    1.x 返回 (result, elapse)，result 是 [box, text, score] 列表。
    """
    try:
        out = engine(img)
        if isinstance(out, tuple):
            out = out[0]
        txts = getattr(out, "txts", None)
        if txts is not None:
            return "\n".join(t for t in txts if t and t.strip())
        if not out:
            return ""
        lines = []
        for item in out:
            text = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else str(item)
            if text.strip():
                lines.append(text)
        return "\n".join(lines)
    except Exception as e:
        logger.warning("OCR failed on page %d of '%s': %s", page_num, filename, e)
        return ""


def _extract_epub(path: Path) -> str:
    try:
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("ebooklib and beautifulsoup4 are required for EPUB. pip install ebooklib beautifulsoup4")

    book = epub.read_epub(str(path))
    parts: List[str] = []
    for item in book.get_items_of_type(9):  # ITEM_DOCUMENT = 9
        content = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_mobi(path: Path) -> str:
    """Extract text from MOBI/AZW3 files.

    Strategy: try to convert via calibre ebook-convert CLI first (most reliable),
    then fall back to a basic raw-text extraction.
    """
    # Try calibre's ebook-convert first
    import subprocess

    # 不能用 shutil.which("ebook-convert")：macOS 上 calibre 装在
    # /Applications/calibre.app/Contents/MacOS/ 里，**不在 PATH**，which 必然
    # 返回 None —— 明明装了 calibre 却直接掉到下面的裸解析（正则扫二进制字节，
    # 效果很差，>100MB 还直接报错返回空）。这里复用 pandoc_parser 的探测
    # （which + 常见安装路径），与电子书主路径用同一个可执行文件。
    from .pandoc_parser import find_ebook_convert

    conv = find_ebook_convert()
    if conv:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            txt_path = tmp.name
        try:
            result = subprocess.run(
                [conv, str(path), txt_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
        except subprocess.TimeoutExpired:
            logger.warning("ebook-convert timed out for '%s'", path.name)
        except Exception as e:
            logger.warning("ebook-convert failed for '%s': %s", path.name, e)
        finally:
            try:
                os.unlink(txt_path)
            except Exception:
                pass

    # Fallback: basic raw extraction
    try:
        return _extract_mobi_raw(path)
    except Exception as e:
        logger.warning("Raw MOBI extraction failed for '%s': %s", path.name, e)
        return ""


def _extract_mobi_raw(path: Path) -> str:
    """Basic MOBI text extraction without external tools.

    对二进制原始字节做全文正则扫描，内存峰值 ≈ 原始字节 + 解码副本 + 匹配
    列表 ≈ 3× 文件大小——数百 MB 的电子书会把 GUI 进程打向 OOM，且 extract
    的效果本来就差（正则捞可读块）。calibre 缺失时的裸解析只对小文件有意义，
    超过上限直接报错让用户装 calibre（正路）。
    """
    _MAX_RAW_MOBI_BYTES = 100 * 1024 * 1024  # 100MB
    size = path.stat().st_size
    if size > _MAX_RAW_MOBI_BYTES:
        raise ValueError(
            f"MOBI/AZW3 文件过大（{size // (1024 * 1024)}MB），裸解析需全量载入内存。"
            f"请安装 calibre 后重试（会走正规转换）：brew install --cask calibre"
        )
    with open(path, "rb") as f:
        data = f.read()

    # Try to find text content in the raw bytes
    text_parts: List[str] = []
    # Look for UTF-8 text segments
    try:
        decoded = data.decode("utf-8", errors="ignore")
        # Extract readable text blocks (sequences of printable chars)
        blocks = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\w\s.,!?;:()\[\]{}\'\"\-–—]{30,}', decoded)
        text_parts.extend(blocks)
    except Exception:
        pass

    # Also try Latin-1
    if not text_parts:
        try:
            decoded = data.decode("latin-1", errors="ignore")
            blocks = re.findall(r'[\w\s.,!?;:()\[\]{}\'\"\-–—]{30,}', decoded)
            text_parts.extend(blocks)
        except Exception:
            pass

    return "\n\n".join(text_parts)


def _extract_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_csv(path: Path) -> str:
    import csv
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # default: comma-delimited
        reader = csv.reader(f, dialect)
        rows = list(reader)
    if not rows:
        return ""
    lines = [" | ".join(r) for r in rows]
    return "\n".join(lines)


def _extract_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: basic regex stripping
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # Remove script/style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


# ---------------------------------------------------------------------------
# Directory-level extraction
# ---------------------------------------------------------------------------


def extract_directory(
    dirpath: str,
    recursive: bool = True,
    extensions: Optional[set] = None,
    errors: Optional[List[str]] = None,
) -> List[tuple]:
    """Walk a directory and extract text from all supported files.

    Returns a list of (filepath, extracted_text) tuples.
    If *errors* is provided, extraction failure messages are appended to it.
    """
    return list(iter_directory(dirpath, recursive=recursive,
                               extensions=extensions, errors=errors))


def iter_directory(
    dirpath: str,
    recursive: bool = True,
    extensions: Optional[set] = None,
    errors: Optional[List[str]] = None,
) -> Iterator[Tuple[str, str]]:
    """流式版本的 extract_directory：逐文件 yield (filepath, text)。

    避免大知识库一次性把全部文件全文累积在内存（H1 隐患）。
    调用方消费完一个文件后即可释放该文件的全文字符串。
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    root = Path(dirpath)
    iterator = root.rglob("*") if recursive else root.glob("*")
    for filepath in iterator:
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in extensions:
            continue
        # Skip hidden files
        if filepath.name.startswith("."):
            continue
        try:
            text = extract_text(str(filepath))
            if text.strip():
                yield (str(filepath), text)
                logger.info("Extracted: %s (%d chars)", filepath.name, len(text))
            else:
                msg = f"{filepath.name}: extracted text is empty"
                logger.warning(msg)
                if errors is not None:
                    errors.append(msg)
        except Exception as e:
            msg = f"{filepath.name}: {e}"
            logger.warning("Failed to extract '%s': %s", filepath.name, e)
            if errors is not None:
                errors.append(msg)