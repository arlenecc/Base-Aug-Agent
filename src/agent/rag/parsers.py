"""Document parsers for Word, Excel, PPT, PDF, EPUB, MOBI, AZW3 and more.

Each parser accepts a file path and returns extracted text. Lazy imports keep
optional dependencies optional — only the formats you actually use need to be
installed.
"""
from __future__ import annotations

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

# AgentConfig（可选注入）。未提供时按「全部开启」处理 —— config 的默认值
# 本身就是 True，不注入不会改变行为；注入后 UI 的开关才生效。
_OCR_CONFIG = None


def set_ocr_config(config) -> None:
    """让解析层感知 AgentConfig（决定是否启用 CoreML EP / 预处理）。

    引擎是进程级单例且可能已被创建，所以这里只影响**后续**创建的引擎；
    调用方若需要立刻生效，应先 :func:`release_ocr_engine`。
    """
    global _OCR_CONFIG
    _OCR_CONFIG = config

# ---------------------------------------------------------------------------
# Format → extension mapping
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".docx", ".doc",
    ".xlsx", ".xls",
    ".pptx", ".ppt",
    ".pdf",
    ".epub",
    ".mobi", ".azw3", ".azw",
    ".txt", ".md", ".markdown", ".rst",
    ".csv", ".tsv",
    ".html", ".htm",
}

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
    }
    label = _PARSER_LABELS.get(suffix, suffix)
    logger.debug("    解析文件类型: %s (%s)", path.name, label)

    if suffix in (".docx", ".doc"):
        return _extract_docx(path)
    elif suffix in (".xlsx", ".xls"):
        return _extract_xlsx(path)
    elif suffix in (".pptx", ".ppt"):
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

    # python-docx/openpyxl/python-pptx 都会持有打开的文件句柄（openpyxl
    # 非 read_only 时还留着 zip 句柄），只靠 GC 回收在 4 worker × 数千文件
    # 的同步里会累积句柄（Windows 上还会锁住文件，导致后续无法删除/覆盖）。
    doc = Document(str(path))
    try:
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
    finally:
        try:
            doc.part.package.stream.close()
        except Exception:
            pass


def _extract_xlsx(path: Path) -> str:
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for Excel files. pip install openpyxl")

    wb = openpyxl.load_workbook(str(path), data_only=True)
    try:
        parts: List[str] = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            parts.append(f"# Sheet: {sheet_name}")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    finally:
        # 关闭底层 zip 句柄（非 read_only 模式会一直持有）
        try:
            wb.close()
        except Exception:
            pass


def _extract_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("python-pptx is required for PPT files. pip install python-pptx")

    prs = Presentation(str(path))
    try:
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
    finally:
        try:
            prs.part.package.stream.close()
        except Exception:
            pass


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF. Auto-detects text vs image-based pages.

    Strategy:
    1. Text-based pages: extract text directly via PyMuPDF get_text("text").
    2. Image-based pages (no text layer, has embedded images): OCR via RapidOCR.
    3. Mixed PDFs: text pages extracted directly, image pages OCR'd.

    If RapidOCR is not installed and the PDF contains image-based pages,
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
        ocr_engine = None       # lazy-init, 复用同一个引擎实例
        ocr_unavailable = False  # OCR 引擎加载失败标记，避免每页重复尝试

        # OCR 图像预处理（自适应 DPI + 尺寸归一化 + CLAHE + 锐化）：默认开启，
        # 可由 AgentConfig.rag_ocr_preprocess 关闭。放在循环外只解析一次，
        # 且导入失败（缺 cv2/numpy）时降级为原始的固定 200dpi + 裸位图。
        def _fallback_dpi(_page) -> int:
            return 200

        render_dpi_for_page = _fallback_dpi
        decode_image = None
        preprocess_for_ocr = None
        preprocess_enabled = bool(getattr(_OCR_CONFIG, "rag_ocr_preprocess", True))
        if preprocess_enabled:
            try:
                from .ocr_engine import (
                    decode_image as _decode,
                    preprocess_for_ocr as _preprocess,
                    render_dpi_for_page as _dpi,
                )
                decode_image, preprocess_for_ocr, render_dpi_for_page = (
                    _decode, _preprocess, _dpi,
                )
            except Exception as e:  # pragma: no cover - 缺少 cv2/numpy
                logger.debug("OCR 预处理不可用，回退原始位图: %s", e)
                preprocess_enabled = False

        # 统计各类型页面数量，用于最终日志
        text_page_count = 0
        image_page_count = 0
        ocr_page_count = 0
        ocr_skipped_count = 0
        blank_page_count = 0

        for page_num in range(total_pages):
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

            if ocr_engine is None:
                ocr_engine = _get_ocr_engine()
                if ocr_engine is None:
                    ocr_unavailable = True
                    ocr_skipped_count += 1
                    continue

            with _OCR_SEMAPHORE:
                # 双重检查：可能在等信号量期间其它 worker 已判定不可用
                if ocr_engine is None:
                    ocr_unavailable = True
                    ocr_skipped_count += 1
                    continue
                # 自适应 DPI：页面物理尺寸差异很大（A4 / letter / 裁剪的小图），
                # 固定 200dpi 会让检测器看到完全不同的字形大小。先按页面尺寸
                # 解出「短边落在目标像素」的 DPI，再做归一化。
                dpi = render_dpi_for_page(page)
                pix = page.get_pixmap(dpi=dpi)
                try:
                    img_bytes = pix.tobytes("png")
                finally:
                    # Pixmap 持有整页位图（数 MB），显式释放避免多页循环时
                    # 峰值内存持续累积。
                    pix = None

                image_for_ocr = img_bytes
                if preprocess_enabled:
                    decoded = decode_image(img_bytes)
                    if decoded is not None:
                        image_for_ocr = preprocess_for_ocr(decoded)
                    # 解码/预处理失败 → 把原始 PNG bytes 交给引擎（旧路径）

                ocr_text = _ocr_image_cached(ocr_engine, image_for_ocr,
                                             page_num + 1, path.name)

            if ocr_text:
                parts.append(ocr_text)
                ocr_page_count += 1
            else:
                # OCR 返回空结果（可能是图片质量太差或引擎异常）
                ocr_skipped_count += 1
    finally:
        # 必须放在 finally：中途抛异常（加密 PDF、损坏文件、OCR 崩溃）时
        # 旧实现直接跳过 doc.close()，泄漏文件句柄与整份文档的解码缓存。
        doc.close()

    # 汇总日志：清晰报告各类型页面的处理情况
    detail_parts = [f"text={text_page_count}"]
    if image_page_count > 0:
        detail_parts.append(f"image={image_page_count}(OCR={ocr_page_count})")
    if ocr_skipped_count > 0:
        detail_parts.append(f"OCR-skipped={ocr_skipped_count}")
    if blank_page_count > 0:
        detail_parts.append(f"blank={blank_page_count}")

    # ---- 全图片版 PDF 且 OCR 不可用 → 抛出明确错误 ----
    if not parts and image_page_count > 0 and ocr_engine is None:
        msg = (
            f"PDF '{path.name}' 是图片版（{total_pages} 页全部为扫描图片），"
            f"但 RapidOCR 未安装，无法进行 OCR 识别。"
            f"请安装: pip install rapidocr-onnxruntime"
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

    if ocr_skipped_count > 0 and ocr_engine is None:
        # 混合 PDF：部分图片页因 OCR 不可用被跳过
        logger.warning(
            "PDF '%s': %d/%d pages are image-based but RapidOCR is not installed. "
            "Image pages were SKIPPED — only text pages were extracted (%d chars). "
            "Install with: pip install rapidocr-onnxruntime",
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


def _get_ocr_engine():
    """Lazy-load RapidOCR engine as a global singleton.

    Returns None if not installed. All workers share the same engine instance
    (~500MB) — without this, 4 workers loading 4 instances would OOM. The
    _OCR_SEMAPHORE serializes actual OCR calls so the shared engine is only
    invoked by one thread at a time.

    引擎构造交给 :mod:`agent.rag.ocr_engine`：Apple Silicon 上会注入 CoreML EP
    （CPU+GPU+ANE），非 Apple 平台或 onnxruntime 过旧时自动回退默认 EP。
    """
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE
        try:
            from .ocr_engine import create_ocr_engine, coreml_reason
            use_coreml = bool(getattr(_OCR_CONFIG, "rag_ocr_coreml", True))
            if not use_coreml:
                logger.info("OCR: CoreML EP 按配置关闭")
            else:
                _, reason = coreml_reason()
                logger.info("OCR: CoreML EP 检测 — %s", reason)
            _OCR_ENGINE = create_ocr_engine(use_coreml=use_coreml)
            logger.info("RapidOCR engine loaded (global singleton)")
            return _OCR_ENGINE
        except ImportError:
            logger.warning(
                "rapidocr-onnxruntime is required for OCR on image-based PDFs. "
                "pip install rapidocr-onnxruntime"
            )
            return None
        except Exception as e:
            logger.warning("Failed to init RapidOCR: %s", e)
            return None


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


def _ocr_image_cached(engine, image, page_num: int, filename: str) -> str:
    """Run OCR using a pre-loaded engine instance.

    ``image`` 可以是 PNG bytes（旧路径）或已预处理的 BGR ndarray——RapidOCR
    的 LoadImage 两者都接受，ndarray 走零拷贝路径，省掉一次编解码往返。
    """
    try:
        result, _ = engine(image)
        if not result:
            return ""
        lines = []
        for item in result:
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
    import shutil

    if shutil.which("ebook-convert"):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            txt_path = tmp.name
        try:
            result = subprocess.run(
                ["ebook-convert", str(path), txt_path],
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
    """Basic MOBI text extraction without external tools."""
    # 只处理前 N MB：MOBI 常有几十 MB，整份读入 + 整份 decode + findall
    # 物化会同时存在 3-4 份副本（latin-1 回退时再翻一倍）。
    _MOBI_MAX_BYTES = 8 * 1024 * 1024
    with open(path, "rb") as f:
        data = f.read(_MOBI_MAX_BYTES)

    # Try to find text content in the raw bytes
    text_parts: List[str] = []
    # Look for UTF-8 text segments
    try:
        decoded = data.decode("utf-8", errors="ignore")
        # finditer 边匹配边消费，避免把所有匹配块一次物化成列表
        text_parts.extend(
            m.group(0) for m in re.finditer(
                r'[\u4e00-\u9fff\u3000-\u303f\w\s.,!?;:()\[\]{}\'\"\-–—]{30,}', decoded)
        )
        del decoded
    except Exception:
        pass

    # Also try Latin-1
    # 注意 data 不能在 utf-8 分支里 del：utf-8 的 errors="ignore" 几乎不会抛
    # 异常，但 text_parts 为空时 latin-1 回退需要原始字节。旧实现在 utf-8
    # 分支 del data，回退触发 NameError 被 except 吞掉 → 非 utf-8 的 MOBI
    # 静默返回空。这里用显式置 None 释放引用，回退分支判空跳过。
    if not text_parts:
        if data is None:
            return "\n\n".join(text_parts)
        try:
            decoded = data.decode("latin-1", errors="ignore")
            text_parts.extend(
                m.group(0) for m in re.finditer(
                    r'[\w\s.,!?;:()\[\]{}\'\"\-–—]{30,}', decoded)
            )
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
        # 单次流式消费：旧实现 rows=list(reader) → lines=[...] → join，
        # 三个全量副本同时存在（rows 与 lines 长期共存），200MB CSV 峰值 1GB+。
        parts: List[str] = []
        append = parts.append
        for row in reader:
            append(" | ".join(row))
    if not parts:
        return ""
    return "\n".join(parts)


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