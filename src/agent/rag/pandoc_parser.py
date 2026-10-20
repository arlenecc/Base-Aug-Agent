"""Pandoc-based conversion — ebooks (epub/mobi/azw3/azw), docx, html, LaTeX → Markdown.

与 docling/marker（PDF 主引擎）平行的轻量转换路线：epub 电子书用 pandoc 直接
转 GFM Markdown（--wrap=none 保证段落单行），docx/html/LaTeX 同理；mobi/azw3/
azw 先用本地 calibre 的 ``ebook-convert`` 转成临时 epub 再走 epub 流程。

epub 还会用 ebooklib 读取书的目录树（TOC），两处用途：
  1. 作为该书在 documents 表里的目录缩略版（digest）；
  2. 指导「按章节切块」——章节边界 = TOC 顶级条目标题对应的 Markdown 标题。

pandoc / ebook-convert 都是系统二进制（非 pip 包）：pandoc 缺失时本模块全部
返回 None，engine 回退 docling/marker/parsers 旧路径；ebook-convert 缺失时
mobi 系转换失败同样回退。
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

from .document_outline import Chapter, _find_headings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 格式集合
# ---------------------------------------------------------------------------

# 电子书格式：pandoc 直转（epub）或经 calibre 中转（mobi 系），按章节切块。
PANDOC_EBOOK_EXTS = {".epub", ".mobi", ".azw3", ".azw"}
# 普通文档：pandoc → Markdown → 常规管线（语义切片、标题 digest）。
PANDOC_DOC_EXTS = {".docx", ".html", ".htm", ".tex", ".latex"}
PANDOC_SUPPORTED_EXTENSIONS = PANDOC_EBOOK_EXTS | PANDOC_DOC_EXTS

# 源格式 → pandoc -f 参数（epub/mobi 系单独处理，不在表内）。
_PANDOC_FROM = {
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".tex": "latex",
    ".latex": "latex",
}

# pandoc 单次转换超时（秒）。整本 epub 实测 < 5s，取宽裕值防大书卡死。
_PANDOC_TIMEOUT = 300
_EBOOK_CONVERT_TIMEOUT = 600

# pandoc 常见安装位置（macOS Homebrew / Linux / Intel brew）。
_PANDOC_CANDIDATES = (
    "/opt/homebrew/bin/pandoc",
    "/usr/local/bin/pandoc",
)
# calibre 的 ebook-convert 常见位置（macOS app bundle 不在 PATH 里）。
_EBOOK_CONVERT_CANDIDATES = (
    "/Applications/calibre.app/Contents/MacOS/ebook-convert",
    "/usr/local/bin/ebook-convert",
    "/opt/homebrew/bin/ebook-convert",
)

_PANDOC_EXE: Optional[str] = None
_EBOOK_CONVERT_EXE: Optional[str] = None


def _resolve_binary(candidates: Sequence[str], name: str) -> Optional[str]:
    """which() 优先，再探测常见安装路径（首次结果缓存）。"""
    found = shutil.which(name)
    if found:
        return found
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def find_pandoc() -> Optional[str]:
    global _PANDOC_EXE
    if _PANDOC_EXE is None:
        _PANDOC_EXE = _resolve_binary(_PANDOC_CANDIDATES, "pandoc")
    return _PANDOC_EXE


def is_pandoc_available() -> bool:
    return find_pandoc() is not None


def find_ebook_convert() -> Optional[str]:
    global _EBOOK_CONVERT_EXE
    if _EBOOK_CONVERT_EXE is None:
        _EBOOK_CONVERT_EXE = _resolve_binary(_EBOOK_CONVERT_CANDIDATES, "ebook-convert")
    return _EBOOK_CONVERT_EXE


# ---------------------------------------------------------------------------
# pandoc 转换
# ---------------------------------------------------------------------------


def _run(cmd: List[str], timeout: int) -> Optional[str]:
    """执行子进程，stdout 为 UTF-8 文本；失败返回 None（不抛异常）。"""
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError:
        logger.warning("%s 不存在或不可执行", cmd[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("%s 超时 (> %ds): %s", cmd[0], timeout, cmd[-1] if len(cmd) > 1 else "")
        return None
    except OSError as e:
        logger.warning("%s 执行失败: %s", cmd[0], e)
        return None
    if res.returncode != 0:
        logger.warning(
            "%s 返回 %d: %s",
            cmd[0], res.returncode, (res.stderr or "").strip()[:300],
        )
        return None
    return res.stdout


# pandoc 从 epub 转出的 spine 锚点：每个内容文档起点一个空 span，
# id 即文档文件名（如 <span id="ch1.xhtml"></span>）——是「直接读到准确
# 章节」的关键：按锚点位置切分正文，边界精确到 epub 的内容文档，TOC 只
# 负责提供权威标题。convert_ebook 用完后再统一剥掉，不进缓存/digest。
_EPUB_ANCHOR_SPAN_RE = re.compile(r'<span id="[^"]*"></span>\s*\n?')
_EPUB_ANCHOR_FIND_RE = re.compile(r'<span id="([^"]+)"></span>')
# 文档级锚点的判定：id 以内容文档扩展名结尾（titlepage.xhtml /
# index_split_000.html）。带 "_filepos<数字>" 的是内部链接目标，不是文档边界。
_EPUB_DOC_ANCHOR_RE = re.compile(r'\.(?:xhtml|html|htm)$', re.IGNORECASE)

# pandoc 转出的空 HTML 注释残留。
_HTML_COMMENT_RE = re.compile(r'<!--[\s\S]*?-->\s*\n?')


def _pandoc_to_markdown(src: str, from_fmt: str) -> Optional[str]:
    exe = find_pandoc()
    if exe is None:
        return None
    # 注意：保留 epub 锚点 span（split_chapters_by_anchors 依赖它定位
    # 内容文档边界），由 convert_ebook 在切块后统一剥离。
    md = _run(
        [exe, str(src), "-f", from_fmt, "-t", "gfm", "--wrap=none", "--strip-comments"],
        timeout=_PANDOC_TIMEOUT,
    )
    if md:
        md = _HTML_COMMENT_RE.sub('', md)
    return md


def convert_document(filepath: str) -> Optional[str]:
    """docx/html/LaTeX → Markdown（pandoc）。失败返回 None（调用方回退旧路径）。"""
    suffix = Path(filepath).suffix.lower()
    fmt = _PANDOC_FROM.get(suffix)
    if fmt is None:
        return None
    md = _pandoc_to_markdown(filepath, fmt)
    if md and md.strip():
        # docx/html/LaTeX 源同样会透传 raw HTML（html 源尤其严重），与电子书
        # 路径保持一致：文档转换无锚点，出口直接剥。图片与内部锚点链接同理
        # 一并剥掉（html 源常带 <img> 与页内锚点）。
        return strip_markdown_noise(strip_raw_html(md))
    return None


def extract_epub_toc(filepath: str) -> List[Chapter]:
    """用 ebooklib 读取 epub 目录树，扁平化为 Chapter 列表。

    ``Chapter.text`` 存 href（如 "ch1.xhtml" / "ch1.xhtml#sec1"），供
    ``split_chapters_by_anchors`` 与 pandoc 锚点做精确对齐；digest 场景
    只用 level/title，text 字段无副作用。Section（无 href 的目录节点）
    text 为空。

    ``book.toc`` 是嵌套结构：``Link``（叶子）与 ``(Section, [children])``
    （子树）混合；Section 层级即目录层级。解析失败返回 []（digest/切块
    回退到原有的启发式实现）。
    """
    try:
        from ebooklib import epub
    except ImportError:
        logger.warning("ebooklib 未安装，无法读取 EPUB 目录树")
        return []
    try:
        book = epub.read_epub(str(filepath))
        toc = book.toc or []
    except Exception as e:
        logger.warning("读取 EPUB 目录树失败 %s: %s", os.path.basename(filepath), e)
        return []

    chapters: List[Chapter] = []

    def _walk(items, level: int) -> None:
        for it in items:
            if isinstance(it, epub.Link):
                title = (it.title or "").strip()
                if title:
                    chapters.append(Chapter(level, title, it.href or ""))
            elif isinstance(it, (tuple, list)):
                sec = it[0] if it else None
                children = it[1] if (isinstance(it, (tuple, list)) and len(it) > 1) else []
                sec_title = getattr(sec, "title", "") or ""
                if sec_title.strip():
                    chapters.append(Chapter(level, sec_title.strip(), getattr(sec, "href", "") or ""))
                    _walk(children, level + 1)
                else:
                    _walk(children, level)

    _walk(list(toc), 1)
    # 逐级去重（同一标题在嵌套 TOC 中偶见重复），保序。
    seen: Set[str] = set()
    unique: List[Chapter] = []
    for c in chapters:
        key = f"{c.level}\x00{c.title}\x00{c.text}"
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def strip_anchor_spans(markdown: str) -> str:
    """剥掉 pandoc 留下的 epub 锚点 span（切块定位完成后清理）。"""
    if not markdown:
        return markdown
    return _EPUB_ANCHOR_SPAN_RE.sub('', markdown)


# 残留 HTML 标签：以 </ 开头的闭合标签，或 <字母 开头的开/自闭合标签。
# 要求标签首字符是字母——避开散文里的比较运算（"if a < b and c > d" 中
# "< b and c >" 不是标签）；pandoc 的 markdown writer 本身会把正文里的
# 字面 "<" 转义成 "\<"，所以能匹配到的都是源文档透传的真 HTML。
_HTML_TAG_RE = re.compile(r'</?[a-zA-Z][^>]*>')
# script/style 整块删除（含内容）——标签剥除只会留下正文，CSS/JS 文本
# 反而会变成噪音入库。
_HTML_BLOCK_RE = re.compile(
    r'<(script|style|head)[^>]*>[\s\S]*?</\1\s*>', re.IGNORECASE
)
_FENCE_OPEN_RE = re.compile(r'^\s{0,3}(```|~~~)')
# 缩进代码块（4 空格 / tab 起）——markdown 字面代码，< > 是内容不是标签。
_INDENT_CODE_RE = re.compile(r'^(?:\t| {4,})\S')
# pandoc 会把正文里的字面反引号/尖括号转义成 \` \< \>（行内代码 span 整体被
# 转义时，`` `[^`]+` `` 的掩码匹配不上），这些转义序列必须先掩掉。
_ESCAPED_RE = re.compile(r'\\[`<>]')


def _strip_html_line(line: str, code_spans: list) -> str:
    """单行剥标签：先掩掉行内代码 span 与转义序列（里面的 < > 不能动）。"""
    def _mask(m: "re.Match") -> str:
        code_spans.append(m.group(0))
        return '\x00%d\x00' % (len(code_spans) - 1)

    protected = re.sub(r'`[^`]+`', _mask, line)
    protected = _ESCAPED_RE.sub(_mask, protected)
    protected = _HTML_BLOCK_RE.sub(' ', protected)
    protected = _HTML_TAG_RE.sub(' ', protected)
    return re.sub(r'\x00(\d+)\x00', lambda m: code_spans[int(m.group(1))], protected)


def _strip_outside_code(markdown: str, fn: "Callable[[str], str]") -> str:
    """逐行应用 ``fn``，跳过围栏代码块与缩进代码块（其中的内容是字面量）。

    代码里的 ``![](x.png)`` / ``<div>`` 是示例内容，不是噪音，不能剥。
    """
    out: List[str] = []
    in_fence = False
    fence_marker = ""
    for line in markdown.split('\n'):
        stripped = line.strip()
        if not in_fence and _FENCE_OPEN_RE.match(line):
            in_fence = True
            fence_marker = stripped[:3]
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue
        if _INDENT_CODE_RE.match(line):
            out.append(line)
            continue
        out.append(fn(line))
    return '\n'.join(out)


# pandoc 从 epub 转出的 markdown 图片：``![](images/xxx.jpg)``。
# 图片本身不入库（多模态检索不在范围内），留着只占切片字数、污染嵌入。
_MD_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
# 内部锚点链接：``[序一](#index_split_003.html_filepos6604)``。
# epub 的目录页与交叉引用会产出成百上千条这类链接，全部是导航噪音；
# 剥掉链接语法但**保留链接文字**（目录条目名本身仍有语义）。
_MD_INTERNAL_LINK_RE = re.compile(r'\[([^\]]*)\]\([^)]*#[^)]*\)')


def strip_markdown_noise(markdown: str) -> str:
    """剥掉 Markdown 图片语法与内部锚点链接（链接文字保留）。

    ``strip_raw_html`` 只管 HTML 标签，管不到这两种 Markdown 原生语法；而
    pandoc 输出的 markdown 在 engine 里被当作「规范 Markdown」直接入库
    （跳过 clean_text），所以必须在这里显式剥。
    """
    if not markdown:
        return markdown

    def _fn(line: str) -> str:
        line = _MD_IMAGE_RE.sub('', line)
        return _MD_INTERNAL_LINK_RE.sub(r'\1', line)

    return _strip_outside_code(markdown, _fn)


def strip_raw_html(markdown: str) -> str:
    """剥掉 pandoc 输出里残留的 raw HTML 标签，保留标签内正文。

    背景：pandoc 转成 GFM 时 ``raw_html`` 扩展默认开启，epub/docx/html 源里
    的 ``<div>``/``<span class=...>``/``<a id=...>`` 等会原样透传进 markdown。
    而 engine 对 pandoc 路径直接把文本当「规范 Markdown」用（跳过 clean_text
    的 HTML 剥离），这些标签就大量进入切片与缓存。

    不能改用 ``-t gfm-raw_html`` 一刀切：epub 锚点 span 本身就是 raw HTML，
    split_chapters_by_anchors 依赖它按内容文档边界精确切章。所以只在锚点
    切分完成后调用本函数。

    保护措施：
      - 围栏代码块（``` / ~~~）内部逐字保留——代码里的 ``vector<int>`` 等
        不是标签；
      - 行内代码 span（`` `...` ``）先掩后剥——`` `<div>` `` 的尖括号是内容；
      - script/style/head 整块连内容一起删（标签剥除只会留下 CSS/JS 噪音）；
      - 标签必须以字母开头，避免误伤散文里的比较运算。
    """
    if not markdown:
        return markdown
    # 围栏/缩进代码块的跳过逻辑统一由 _strip_outside_code 处理（图片与内部
    # 链接的剥离共用同一套保护）。
    stripped_md = _strip_outside_code(markdown, lambda line: _strip_html_line(line, []))
    return re.sub(r'[ \t]{2,}', ' ', stripped_md)


def split_chapters_by_anchors(
    markdown: str,
    toc_chapters: List[Chapter],
) -> Optional[List[str]]:
    """按 epub 内容文档边界精确切分章节（不需要启发式）。

    pandoc 在每个 spine 内容文档的起点输出锚点
    ``<span id="文件名"></span>``；TOC 条目的 href 也指向内容文档。两者
    对齐后：边界 = 锚点位置（epub 物理结构，准确），标题 = TOC 中该文档
    最浅层级条目的 title（权威目录）。

    返回章节文本列表（未做超长切分，块内子标题行保留）；以下任一情况
    返回 None，由调用方回退到原有的启发式实现：
      - TOC 为空（无法提取准确目录信息）；
      - pandoc 输出中锚点 < 2（单文档书无法按文档分章）；
      - 切出的章节全部为空。
    """
    if not toc_chapters or not markdown:
        return None

    # 只保留**文档级**锚点。pandoc（3.12 实测）除了每个 spine 内容文档的
    # 起点，还会为**每个内部链接目标**输出一个锚点，id 形如
    #   index_split_008.html_filepos46532
    # 即末段是 "_filepos<数字>" 而不是文档扩展名。实测某本 54 个内容文档的
    # epub 共输出 982 个锚点，其中 928 个是这类链接/脚注目标。把它们一律当
    # 成章节边界，会把正文在每个脚注处切碎——产生数百个几字符的碎片块，
    # 检索时全是噪音，观感就是「只转了目录，没有内容」。
    anchors = [
        (m.start(), os.path.basename(m.group(1)).lower())
        for m in _EPUB_ANCHOR_FIND_RE.finditer(markdown)
        if _EPUB_DOC_ANCHOR_RE.search(m.group(1))
    ]
    if len(anchors) < 2:
        return None

    # 文件名 → 最浅层级 TOC 标题（章名取权威目录；同名取最先出现的更浅者）。
    title_by_file: dict = {}
    for c in toc_chapters:
        if not c.text:
            continue
        fname = os.path.basename(c.text.split('#', 1)[0]).lower()
        if not fname:
            continue
        cur = title_by_file.get(fname)
        if cur is None or c.level < cur[0]:
            title_by_file[fname] = (c.level, c.title)

    # 段落边界：锚点 i 的段 = [pos_i, pos_{i+1})；首个锚点前的前缀单独成块。
    chunks: List[str] = []
    prefix = markdown[: anchors[0][0]].strip()
    if prefix:
        chunks.append(strip_anchor_spans(prefix).strip())

    for i, (_pos, fname) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(markdown)
        seg = strip_anchor_spans(markdown[_pos:end]).strip()
        if not seg:
            continue
        entry = title_by_file.get(fname)
        if entry and not seg.startswith('#'):
            # 内容文档没有自己的标题行 → 用 TOC 标题补上。
            seg = f"# {entry[1]}\n\n{seg}"
        chunks.append(seg)

    return chunks if chunks else None


# ---------------------------------------------------------------------------
# 扫描版 epub：图片 OCR
# ---------------------------------------------------------------------------

# 有些 epub 是「扫描图片版」：正文每页都是一张图，文字层只剩目录/页眉。实测某
# 241.9MB 的统计教材：535 张图占 241.8MB（99.96%），文字层仅 5474 字符。这类
# 书走 pandoc/docling 只能拿到目录，正文一个字都没有——必须对图片做 OCR。
#
# 判定：图片总字节 ≥ _EPUB_SCAN_MIN_IMAGE_BYTES 且「文字层字符 × 200 < 图片字节」
# （图片在体积上压倒性占优）。纯文字书不会误判：正文几万字 × 200 远超其插图体积。
_EPUB_SCAN_MIN_IMAGE_BYTES = 5 * 1024 * 1024
_EPUB_SCAN_TEXT_WEIGHT = 200
# pandoc 转出的正文少于这个字符数才考虑「可能是扫描版」（够长就不折腾）。
_EPUB_SCAN_MD_MIN_CHARS = 20000
# 太小的图是装饰（图标、分隔线、角标），不值得 OCR。
_EPUB_OCR_MIN_IMAGE_BYTES = 20 * 1024

_TAG_RE = re.compile(r'<[^>]+>')


def _epub_scan_profile(filepath: str) -> Optional[Tuple[int, int, list]]:
    """读取 epub 结构 → (图片总字节, 文字层字符数, 图片 item 列表)。

    图片 item 按文件名排序：扫描版通常命名为 page001.jpg…page535.jpg，文件名
    序即页序。读取失败返回 None（调用方按「非扫描版」处理，行为与改动前一致）。
    """
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        return None
    try:
        book = epub.read_epub(str(filepath))
    except Exception as e:
        logger.debug("读取 epub 结构失败 %s: %s", os.path.basename(filepath), e)
        return None
    try:
        docs = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        imgs = [i for i in book.get_items() if i.get_type() == ebooklib.ITEM_IMAGE]
    except Exception as e:
        logger.debug("遍历 epub 条目失败 %s: %s", os.path.basename(filepath), e)
        return None

    img_bytes = 0
    for im in imgs:
        try:
            img_bytes += len(im.get_content() or b"")
        except Exception:
            continue
    text_chars = 0
    for d in docs:
        try:
            raw = (d.get_content() or b"").decode("utf-8", "replace")
        except Exception:
            continue
        text_chars += len(_WS_RE.sub("", _TAG_RE.sub("", raw)))
    imgs.sort(key=lambda i: getattr(i, "file_name", "") or "")
    return img_bytes, text_chars, imgs


def _is_scanned_epub(filepath: str) -> bool:
    """是否为扫描图片版 epub（正文全是图，文字层只有目录/页眉）。"""
    prof = _epub_scan_profile(filepath)
    if prof is None:
        return False
    img_bytes, text_chars, _imgs = prof
    return (img_bytes >= _EPUB_SCAN_MIN_IMAGE_BYTES
            and text_chars * _EPUB_SCAN_TEXT_WEIGHT < img_bytes)


def _ocr_image_array(img) -> str:
    """对单张**已预处理**的灰度 ndarray 做识别。

    **ocrmac（Apple Vision）优先**，不可用/失败时回退 RapidOCR —— 与 PDF 主
    引擎（docling 的 OcrAutoOptions）的策略一致：Vision 是系统级加速，中文
    识别的质量与速度都更好。
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
        parts: List[str] = []
        for item in results or []:
            # detail=False → (text, confidence)；个别版本仍返回三元组。
            text = item[0] if isinstance(item, (tuple, list)) else item
            if text and str(text).strip():
                parts.append(str(text).strip())
        if parts:
            return "\n".join(parts)
        logger.debug("ocrmac 返回空结果，回退 rapidocr")
    except Exception as e:
        logger.debug("ocrmac 不可用/识别失败，回退 rapidocr: %s", e)

    # 回退 RapidOCR：复用 parsers 的引擎单例与结果缓存（与 PDF 图片页同一实例）。
    try:
        from .parsers import _get_ocr_engine, _ocr_image_cached

        engine = _get_ocr_engine()
        if engine is None:
            return ""
        return _ocr_image_cached(engine, img, 0, "epub-page")
    except Exception as e:
        logger.debug("rapidocr 回退失败: %s", e)
        return ""


def ocr_epub_images(filepath: str, log=None) -> Optional[str]:
    """对扫描版 epub 的图片逐张 OCR，返回拼接正文（无可识别内容返回 None）。

    预处理复用 ``parsers._ocr_preprocess_array``——与 PDF 图片页**完全一致**的
    归一化：长边 Lanczos ≤1500px、灰度、USM 锐化、Otsu 二值化。

    内存：逐张读取、识别后立即释放，不会把整本书的图片（可达数百 MB）常驻。
    """
    prof = _epub_scan_profile(filepath)
    if prof is None:
        return None
    _img_bytes, _text_chars, imgs = prof
    if not imgs:
        return None

    import io

    import numpy as np
    from PIL import Image

    from .parsers import _ocr_preprocess_array

    parts: List[str] = []
    done = 0
    for idx, item in enumerate(imgs):
        try:
            raw = item.get_content() or b""
        except Exception:
            continue
        if len(raw) < _EPUB_OCR_MIN_IMAGE_BYTES:
            continue  # 装饰图，跳过
        try:
            pil = Image.open(io.BytesIO(raw))
            pil.load()
            gray = pil.convert("L")
            del pil
            arr = np.asarray(gray, dtype=np.uint8)
            del gray
            arr = _ocr_preprocess_array(arr)
            text = _ocr_image_array(arr)
            del arr
            if text and text.strip():
                parts.append(text.strip())
                done += 1
        except Exception as e:
            logger.debug("epub 图片 OCR 失败 %s 第 %d 张: %s",
                         os.path.basename(filepath), idx, e)
            continue
        if log and done and done % 20 == 0:
            log("    ├─ 扫描版 OCR 进度: %d/%d 张", done, len(imgs))

    if not parts:
        return None
    return "\n\n".join(parts)


def convert_ebook(filepath: str) -> Optional[Tuple[str, List[Chapter], Optional[List[str]]]]:
    """epub/mobi/azw3/azw → (Markdown, TOC Chapters, 预切章节)。

    - epub：pandoc 直转 + ebooklib 读 TOC + 按锚点精确切章。
    - mobi/azw3/azw：calibre ``ebook-convert`` → 临时 epub → pandoc + TOC
      + 切章，临时文件在读取后即删除。

    第三项是 ``split_chapters_by_anchors`` 的结果（章节文本列表，未做超长
    切分）；None 表示无法精确切分（TOC 读不到 / 锚点不足），调用方回退到
    原有启发式实现。返回的 Markdown 已剥离锚点 span，可直接清洗入库。
    任一步骤失败整体返回 None（调用方回退 docling/parsers 旧路径）。
    """
    suffix = Path(filepath).suffix.lower()
    tmp_epub: Optional[str] = None
    try:
        if suffix == ".epub":
            src = filepath
        else:
            conv = find_ebook_convert()
            if conv is None:
                logger.warning(
                    "未找到 calibre ebook-convert，无法转换 %s（请安装 calibre）",
                    os.path.basename(filepath),
                )
                return None
            fd, tmp_epub = tempfile.mkstemp(suffix=".epub", prefix="rag_pandoc_")
            os.close(fd)
            os.unlink(tmp_epub)  # ebook-convert 要求目标不存在
            res = _run([conv, str(filepath), tmp_epub], timeout=_EBOOK_CONVERT_TIMEOUT)
            if res is None or not os.path.isfile(tmp_epub):
                _cleanup(tmp_epub)
                tmp_epub = None
                return None
            src = tmp_epub

        toc = extract_epub_toc(src)
        md = _pandoc_to_markdown(src, "epub")
        if not md or not md.strip():
            return None
        # 清洗后的正文（锚点 span 要留给 split_chapters_by_anchors，所以只对
        # 副本做剥离）。
        body_md = strip_markdown_noise(strip_raw_html(strip_anchor_spans(md)))
        # 扫描图片版：正文全是图，pandoc 只能抽出目录/页眉那点文字 → 改用图片
        # OCR 取正文；TOC 仍然保留（缩略版与目录信息是准的）。
        #
        # 判定必须用**清洗后**的正文量，不能拿原始 md：扫描版的原始 md 里塞满
        # <img>/<svg> 标记（实测 82266 字符，剥完只剩 9986），拿原始值判断会
        # 被误判成「内容很足」而跳过 OCR。
        if (len(_WS_RE.sub("", body_md)) < _EPUB_SCAN_MD_MIN_CHARS
                and _is_scanned_epub(src)):
            logger.info("%s: 判定为扫描图片版 epub，改用图片 OCR 取正文",
                        os.path.basename(filepath))
            ocr_md = ocr_epub_images(src)
            if ocr_md:
                return ocr_md, toc, None
            logger.warning("%s: 扫描版图片 OCR 未取到内容，回退 pandoc 结果",
                           os.path.basename(filepath))
        chapter_texts = split_chapters_by_anchors(md, toc)
        md = body_md
        if chapter_texts:
            chapter_texts = [strip_markdown_noise(strip_raw_html(t)) for t in chapter_texts]
        return md, toc, chapter_texts
    finally:
        if tmp_epub:
            _cleanup(tmp_epub)


def _cleanup(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 章节切块
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm_title(title: str) -> str:
    return _WS_RE.sub("", title or "").lower()


def _split_sections(markdown: str) -> List[Tuple[int, str, str]]:
    """按 ATX 标题把 Markdown 切成 (level, title, body) 列表（跳过围栏内 #）。

    首个标题前的内容作为 (0, "", prefix)。围栏代码块由
    document_outline._find_headings 统一处理。
    """
    if not markdown or not markdown.strip():
        return []
    headings = _find_headings(markdown)
    sections: List[Tuple[int, str, str]] = []
    if headings:
        prefix = markdown[: headings[0][2]].strip()
        if prefix:
            sections.append((0, "", prefix))
        for i, (level, title, _start, end) in enumerate(headings):
            nxt = headings[i + 1][2] if i + 1 < len(headings) else len(markdown)
            sections.append((level, title, markdown[end:nxt].strip()))
    else:
        sections.append((0, "", markdown.strip()))
    return sections


def _group_by_boundary(
    sections: List[Tuple[int, str, str]],
    is_boundary: Callable[[int, str], bool],
) -> List[List[Tuple[int, str, str]]]:
    """把 section 序列聚成组：边界条件命中的 section 开新组。"""
    groups: List[List[Tuple[int, str, str]]] = []
    for sec in sections:
        if not groups or sec[0] == 0 or (sec[1] and is_boundary(sec[0], sec[1])):
            groups.append([sec])
        else:
            groups[-1].append(sec)
    return groups


def _count_contentful(groups: List[List[Tuple[int, str, str]]]) -> int:
    return sum(1 for g in groups if any(body for _lv, _t, body in g))


def build_chapter_chunks(
    markdown: str,
    toc_chapters: Optional[List[Chapter]],
    max_chars: int = 1200,
    oversized_split: Optional[Callable[[str], List[str]]] = None,
) -> List[str]:
    """按章节把整书 Markdown 切成 chunk 文本列表（不经语义切片）。

    边界判定优先级：
      1. TOC 顶级条目标题（ebooklib 目录树的最小层级）与 Markdown 标题匹配；
      2. 匹配失败/无 TOC → 取「能产生 ≥2 个非空组的最小标题层级」
         （书名通常是唯一的 H1，自动下沉到 H2 章节级）。

    章节内更深的标题行会保留在 chunk 文本里（作为上下文）；单个章节超过
    ``max_chars`` 时交给 ``oversized_split``（引擎传入基于 chunk_documents
    的兜底切片，保证不超嵌入模型上下文）。
    """
    sections = _split_sections(markdown)
    if not sections:
        return []

    boundary_titles: Optional[Set[str]] = None
    if toc_chapters:
        min_level = min(c.level for c in toc_chapters)
        titles = {_norm_title(c.title) for c in toc_chapters
                  if c.level == min_level and c.title}
        if titles:
            boundary_titles = titles

    if boundary_titles is not None:
        groups = _group_by_boundary(
            sections, lambda _lv, title: _norm_title(title) in boundary_titles)
        # TOC 标题与正文标题完全对不上（<2 个非空组）→ 层级兜底。
        if _count_contentful(groups) < 2 and len(sections) > 1:
            groups = None  # type: ignore[assignment]
    else:
        groups = None  # type: ignore[assignment]

    if groups is None:
        levels = sorted({lv for lv, _t, _b in sections if lv > 0})
        groups = [list(sections)]
        for lvl in levels:
            candidate = _group_by_boundary(sections, lambda _lv, _t, _lvl=lvl: _lv <= _lvl)
            if _count_contentful(candidate) >= 2:
                groups = candidate
                break

    chunks: List[str] = []
    for group in groups:
        head_level, head_title, head_body = group[0]
        pieces: List[str] = []
        if head_body:
            pieces.append(head_body)
        for level, title, body in group[1:]:
            if not body:
                continue
            if title and level > 0:
                pieces.append(f"{'#' * level} {title}\n\n{body}")
            else:
                pieces.append(body)
        text = "\n\n".join(pieces).strip()
        if not text:
            continue
        # 章节标题行放回 chunk 顶部（extract 切分时标题行不进 body）。
        if head_title:
            text = f"{'#' * max(head_level, 1)} {head_title}\n\n{text}"
        if len(text) <= max_chars or oversized_split is None:
            chunks.append(text)
        else:
            chunks.extend(oversized_split(text))
    return chunks
