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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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

# 扫描版 epub 的**整本 OCR 时间预算**（秒）。
#
# 为什么需要：mobi/azw3 转出的 epub 与原文件等大（实测 86MB→85MB、
# 75MB→71MB），图多的书（漫画/绘本/扫描件）文字层为 0 字符 → 判定扫描版 →
# 全部正文只能逐张 OCR。实测 75MB 漫画 = 202 张图 ≈ 2~7 分钟；KB 里最大的
# 半小时漫画套装 417MB（16 册）会到上千张 ≈ 9~37 分钟。而这段既没有超时
# 也没有进度输出，一个文件就能把 worker 占满数十分钟且界面零输出 —— 正是
# 「同步卡住了」的直接观感。给一个宽裕但有界的预算，超了就带着已识别的部分
# 收尾，绝不无限期占住 worker。
_EPUB_OCR_TIME_BUDGET_S = 900

# ---------------------------------------------------------------------------
# 扫描版 epub 的并发闸门：**同一时刻只允许 1 本扫描书在做图片 OCR**
# ---------------------------------------------------------------------------
# 为什么只拦扫描版、不拦整本 epub：扫描版的峰值内存不比 PDF 小——`epub.read_epub`
# 把整本书（含所有图片字节）载入进程，还要逐张解码图片送 RapidOCR（2 实例常驻
# ~1GB）。若 2 个 worker 同时各扛一本大扫描书就是数 GB 瞬时占用，在统一内存的
# Apple Silicon 上会卡顿 / 被系统杀进程。
#
# 而**文字版 epub**（pandoc 直抽正文，轻量）保持 2 路并行、不进这道闸——按用户
# 要求「扫描版串行、文字版 2 并行」：判定「可能是扫描版」的便宜门槛是清洗后
# 正文极短（< ``_EPUB_SCAN_MD_MIN_CHARS``），只有这种书才进入扫描闸去读全本 +
# OCR；正文够长的书根本不读 profile、不进闸，全程 2 并行。
_SCANNED_EPUB_SEM = threading.Semaphore(1)   # 1 = 扫描版 OCR 严格串行
# 闸门等待超过该秒数就打 INFO：同步期「很久没输出」时，这条日志能直接区分
# 「卡在等扫描闸（另一本扫描书独占中）」与「卡在 OCR 本身」。
_SCANNED_EPUB_GATE_WAIT_LOG_S = 5.0
# 单本扫描书内部并行 OCR 的线程数。= 2 才能让 RapidOCR 的 2 实例真正同时开工
# （扫描版整体串行，所以这本内部并行不会突破「1 本在 OCR」的内存上限）。
_EPUB_OCR_MAX_WORKERS = 2


@contextmanager
def _scanned_epub_gate(filepath: str):
    """进入/退出扫描版 OCR 串行闸；等待过久打 INFO（区分卡顿与串行排队）。"""
    name = os.path.basename(filepath)
    logged = False
    start = time.time()
    acquired = False
    try:
        while True:
            if _SCANNED_EPUB_SEM.acquire(blocking=False):
                acquired = True
                if logged:
                    logger.info("%s: 扫描版 OCR 闸门已获取，继续处理", name)
                break
            if (not logged
                    and time.time() - start > _SCANNED_EPUB_GATE_WAIT_LOG_S):
                logger.info(
                    "%s: 等待扫描版 OCR 闸门——当前已有另一本扫描书在 OCR"
                    "（为防内存峰值刻意串行），稍候…", name)
                logged = True
            time.sleep(0.5)
        yield
    finally:
        if acquired:
            _SCANNED_EPUB_SEM.release()


# mobi/azw3 转换失败的路径负缓存：path -> 失败时刻（monotonic）。
#
# 为什么需要：engine 对 mobi/azw3 先走本模块的 convert_ebook()（calibre
# 转临时 epub，超时 600s），失败后回退 parsers._extract_mobi()——后者又会用
# **同一个 calibre** 把同一个文件再转一次（输出 .txt，超时 120s）。于是每个
# 转换失败的文件要付两次完整的 calibre 转换：最坏 720s 才走到裸解析。KB 里
# mobi/azw3 有 6585 个，DRM/损坏文件一多，这两段叠加就是「同步长时间不动」。
# 记下本模块已判失败的路径，让兜底跳过它自己的那次 calibre 尝试。
_EBOOK_CONVERT_FAILED: dict = {}
_EBOOK_CONVERT_FAILED_TTL_S = 3600.0


def mark_ebook_convert_failed(filepath: str) -> None:
    """记录「本模块的 calibre 转换已失败」，供兜底路径跳过重复转换。"""
    try:
        _EBOOK_CONVERT_FAILED[str(filepath)] = time.monotonic()
        # 顺手清理过期条目，避免长期运行的进程里这张表无限增长。
        now = time.monotonic()
        for k in [k for k, t in _EBOOK_CONVERT_FAILED.items()
                  if now - t > _EBOOK_CONVERT_FAILED_TTL_S]:
            _EBOOK_CONVERT_FAILED.pop(k, None)
    except Exception:
        pass


def ebook_convert_failed_recently(filepath: str) -> bool:
    """该文件是否刚在本模块转换失败过（兜底路径据此跳过重复的 calibre 调用）。"""
    t = _EBOOK_CONVERT_FAILED.get(str(filepath))
    if t is None:
        return False
    if time.monotonic() - t > _EBOOK_CONVERT_FAILED_TTL_S:
        _EBOOK_CONVERT_FAILED.pop(str(filepath), None)
        return False
    return True

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


def _kill_proc_tree(proc) -> None:
    """回收超时子进程及其整棵子树（进程组 kill，见 _run 的注释）。"""
    if proc is None:
        return
    try:
        import signal

        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        # 必须再 communicate 一次回收僵尸，否则子进程变 <defunct> 常驻。
        proc.communicate(timeout=10)
    except Exception:
        pass


class _CompletedProc:
    """把 Popen.communicate() 的结果包装成 subprocess.run() 的返回形态。"""
    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, stdout, stderr, returncode) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


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
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            # 自成进程组：ebook-convert/pandoc 超时时要能**整组**回收。
            # subprocess.run(timeout=) 只 kill 顶层进程，派生的子进程会成为
            # 孤儿继续占 CPU/内存（本项目在 shell_run / MCP 已用同一纪律，见
            # 设计文档 6.4.1），这里补齐。
            start_new_session=True,
        )
        # returncode **必须在 communicate() 之后**读取（那时进程才退出、码才有效）。
        stdout, stderr = proc.communicate(timeout=timeout)
        res = _CompletedProc(stdout, stderr, proc.returncode)
    except FileNotFoundError:
        logger.warning("%s 不存在或不可执行", cmd[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("%s 超时 (> %ds): %s", cmd[0], timeout, cmd[-1] if len(cmd) > 1 else "")
        _kill_proc_tree(proc)
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

# HTML 表格块：复杂表格 markdown 无法无损表示，须保留原始 <table> 块（含行列
# 结构），整体掩成占位符、剥离其余 raw HTML 后再还原——避免逐行剥标签时把
# <table>/<tr>/<td> 拆散成散落文字、丢失表格结构。
_TABLE_BLOCK_RE = re.compile(r'<table\b[^>]*>[\s\S]*?</table>', re.IGNORECASE)


def _mask_table_blocks(text: str):
    """把 ``<table>...</table>`` 块掩成占位符，返回 (掩码文本, 还原函数)。"""
    blocks: list = []

    def _repl(m: "re.Match") -> str:
        blocks.append(m.group(0))
        return '\x00T%d\x00' % (len(blocks) - 1)

    masked = _TABLE_BLOCK_RE.sub(_repl, text)

    def _restore(t: str) -> str:
        for i, b in enumerate(blocks):
            t = t.replace('\x00T%d\x00' % i, b)
        return t

    return masked, _restore


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
    # 保护 HTML 表格块：复杂表格 markdown 表达不了，保留原始 <table> 块（含
    # 行列结构），整体掩掉后再剥其余 raw HTML，避免逐行剥标签把表格拆散。
    masked, restore = _mask_table_blocks(markdown)
    # 围栏/缩进代码块的跳过逻辑统一由 _strip_outside_code 处理（图片与内部
    # 链接的剥离共用同一套保护）。
    stripped_md = _strip_outside_code(masked, lambda line: _strip_html_line(line, []))
    stripped_md = restore(stripped_md)
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
# 文字层**硬下限**：超过这个字符数就一律不算扫描版。图文并茂的书
# （漫画/绘本，正文字符数十万）图也很多，会被「图片字节压过文字×200」误判，
# 但它们有可读文字层，pandoc/docling 直接抽即可，不应去 OCR 几千张图。
# 真正的扫描版文字层只有目录/页眉（实测统计教材 5474 字符）。
_EPUB_SCAN_TEXT_FLOOR = 20000
# pandoc 转出的正文少于这个字符数才考虑「可能是扫描版」（够长就不折腾）。
_EPUB_SCAN_MD_MIN_CHARS = 20000
# 太小的图是装饰（图标、分隔线、角标），不值得 OCR。
_EPUB_OCR_MIN_IMAGE_BYTES = 20 * 1024

_TAG_RE = re.compile(r'<[^>]+>')
# 正文里的图片引用（XHTML 的 <img src> 与内嵌 SVG 的 <image xlink:href>）。
_EPUB_IMG_SRC_RE = re.compile(
    r'<img[^>]*?\bsrc\s*=\s*["\']([^"\']+)["\']'
    r'|<image[^>]*?\bxlink:href\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE)


def _epub_scan_profile(filepath: str) -> Optional[Tuple[int, int, list]]:
    """读取 epub 结构 → (图片总字节, 文字层字符数, 图片 item 列表)。

    图片 item 按**阅读顺序**排列（见
    :func:`_epub_images_in_reading_order`）——扫描版的文件名顺序经常不等于页
    序。读取失败返回 None（调用方按「非扫描版」处理，行为与改动前一致）。
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
    imgs = _epub_images_in_reading_order(book, docs, imgs)
    return img_bytes, text_chars, imgs


def _epub_images_in_reading_order(book, docs, imgs):
    """按**阅读顺序**重排图片：spine 文档里 ``<img>`` 的出现次序。

    manifest（``Image00000.jpg``…）的文件名顺序**不等于页序**——实测某本
    534 页的书 ``Image00000`` 对应的是第 470 页。若按 manifest 顺序 OCR，
    拼接出的正文前后颠倒，切片语义全乱，检索质量急剧下降。

    未在正文里被引用的图片（封面、纯装饰图）按文件名接在末尾，保证不丢页。
    """
    import posixpath

    def _href(item):
        return getattr(item, "file_name", "") or ""

    def _base(href):
        return posixpath.basename((href or "").split("#")[0])

    by_href = {}
    by_base = {}
    for im in imgs:
        href = _href(im)
        by_href.setdefault(href, im)
        by_href.setdefault(href.lstrip("./"), im)
        by_href.setdefault(_base(href), im)
        by_base.setdefault(_base(href), im)

    ordered_docs: List = []
    try:
        for entry in book.spine or []:
            idref = entry[0] if isinstance(entry, (tuple, list)) else entry
            item = book.get_item_with_id(idref)
            if item is not None:
                ordered_docs.append(item)
    except Exception as e:
        logger.debug("解析 epub spine 失败，回退文档顺序: %s", e)
    if not ordered_docs:
        ordered_docs = list(docs)

    result: List = []
    used = set()
    for d in ordered_docs:
        try:
            raw = (d.get_content() or b"").decode("utf-8", "replace")
        except Exception:
            continue
        base_dir = posixpath.dirname(_href(d))
        for groups in _EPUB_IMG_SRC_RE.findall(raw):
            # 两个捕获组（img src / svg image xlink:href），取命中的那个。
            src = next((g for g in groups if g), "")
            src = src.split("#")[0].strip()
            if not src:
                continue
            cand = (by_href.get(src)
                    or by_href.get(posixpath.normpath(posixpath.join(base_dir, src)))
                    or by_base.get(_base(src)))
            if cand is not None and id(cand) not in used:
                used.add(id(cand))
                result.append(cand)

    for im in sorted(imgs, key=_href):
        if id(im) not in used:
            result.append(im)
    return result


def _is_scanned_profile(prof) -> bool:
    """给定扫描特征，判定是否为扫描图片版 epub（双条件说明见 _is_scanned_epub）。"""
    if prof is None:
        return False
    img_bytes, text_chars, _imgs = prof
    return (img_bytes >= _EPUB_SCAN_MIN_IMAGE_BYTES
            and text_chars * _EPUB_SCAN_TEXT_WEIGHT < img_bytes
            and text_chars < _EPUB_SCAN_TEXT_FLOOR)


def _is_scanned_epub(filepath: str) -> bool:
    """是否为扫描图片版 epub（正文全是图，文字层只有目录/页眉）。

    双条件，缺一不可：
    1. 图片字节占压倒性多数（图字节 ≥ 阈值 且 文字层×200 < 图片字节）；
    2. 文字层本身极少（< ``_EPUB_SCAN_TEXT_FLOOR``）—— 这是防误判的关键：
       图文并茂的漫画/绘本正文字符数十万，会被条件 1 误伤，但它们有可读
       文字层，走 pandoc/docling 直接抽正文即可，不该去 OCR 几千张图。
    """
    return _is_scanned_profile(_epub_scan_profile(filepath))


def _ocr_image_array(img, stats: Optional[dict] = None) -> str:
    """对单张**已预处理**的灰度 ndarray 做识别。

    直接委托 :func:`agent.rag.parsers.ocr_recognize_array`——ocrmac(Apple
    Vision) 优先、回退 RapidOCR。与 PyMuPDF 兜底 PDF 页共用同一入口，避免
    两条通路的引擎选择策略漂移。
    """
    from .parsers import ocr_recognize_array

    return ocr_recognize_array(img, 0, "epub-page", stats=stats)


def _ocr_one_image(raw: bytes, stats: dict) -> Optional[str]:
    """单张图片解码 + OCR（供线程池调用）。返回识别文本或 None。"""
    import io
    import numpy as np
    from PIL import Image
    try:
        # with 保证解码器/文件句柄一定释放：np.asarray 或后续任一步抛异常时，
        # 旧的 del pil / del gray 都会被跳过（与 ovis_ocr 的写法对齐）。
        with Image.open(io.BytesIO(raw)) as pil:
            pil.load()
            gray = pil.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
        del gray
        # 仅送灰度原图：完整预处理（缩放/归一化/锐化/二值化）由
        # parsers.ocr_recognize_array 内部按引擎分别施加——ocrmac 走灰度、
        # RapidOCR 兜底走二值化。这里若先做二值化会把 ocrmac 的输入污染掉。
        return _ocr_image_array(arr, stats=stats)
    except Exception as e:
        logger.debug("epub 图片 OCR 解码/识别失败: %s", e)
        return None


def ocr_epub_images(filepath: str, log=None, profile=None) -> Optional[str]:
    """对扫描版 epub 的图片**并行** OCR，返回拼接正文（无可识别内容返回 None）。

    预处理（长边 Lanczos ≤1500px、灰度、USM 锐化、可选 Otsu 二值化）由
    ``parsers.ocr_recognize_array`` 统一施加——与 PDF 图片页**完全一致**，且按引擎
    分流：ocrmac 走灰度（不二值化，避免 Vision 读不出字）、RapidOCR 兜底走二值化。

    并发（两层）：
    * **跨书**：由 ``convert_ebook`` 的扫描版串行闸保证同一时刻只 1 本在 OCR，
      所以这本内部再怎么并行也不会突破「1 本在 OCR」的内存上限；
    * **单书内**：用 ``_EPUB_OCR_MAX_WORKERS``(=2) 线程池并行 OCR 图片，让
      RapidOCR 的 2 实例真正同时开工（这是把 RapidOCR 实例池从 1 调到 2 的意义）。
    内存：逐张读取、识别后立即释放，不会把整本书的图片（可达数百 MB）常驻；
    线程池用信号量限制「预取」图片数，避免一次性把上千张图全排队进内存。
    """
    prof = profile if profile is not None else _epub_scan_profile(filepath)
    if prof is None:
        return None
    _img_bytes, _text_chars, imgs = prof
    if not imgs:
        return None

    # 过滤装饰图（太小），并记录（idx, raw）以便按原顺序拼接。
    tasks: List[Tuple[int, bytes]] = []
    for idx, item in enumerate(imgs):
        try:
            raw = item.get_content() or b""
        except Exception:
            continue
        if len(raw) < _EPUB_OCR_MIN_IMAGE_BYTES:
            continue  # 装饰图，跳过
        tasks.append((idx, raw))
    if not tasks:
        return None

    parts_map: dict = {}
    done = 0
    _stats: dict = {}
    _t_start = time.time()
    name = os.path.basename(filepath)
    # 背压：最多允许 max_workers*3 张图「已提交未回收」，限制常驻字节。
    _prefetch = threading.Semaphore(_EPUB_OCR_MAX_WORKERS * 3)

    def _progress() -> None:
        nonlocal done
        done += 1
        if done % 20 == 0:
            elapsed = time.time() - _t_start
            logger.info("  扫描版 OCR 进度: %s %d/%d 张（已 %.0fs）",
                        name, done, len(tasks), elapsed)
            if log:
                log("    ├─ 扫描版 OCR 进度: %d/%d 张", done, len(tasks))

    with ThreadPoolExecutor(max_workers=_EPUB_OCR_MAX_WORKERS) as ex:
        futures = []
        for idx, raw in tasks:
            _prefetch.acquire()
            fut = ex.submit(_ocr_one_image, raw, _stats)
            fut.add_done_callback(lambda f: _prefetch.release())
            futures.append((idx, fut))
        timed_out = False
        for idx, fut in futures:
            if timed_out:
                fut.cancel()
                continue
            try:
                text = fut.result()
            except Exception:
                text = None
            if text and text.strip():
                parts_map[idx] = text.strip()
            _progress()
            # 时间预算：图极多的套装书不该无限期占住扫描闸。超时就带着已识别的
            # 部分收尾并告警；剩余部分下一轮同步不会重试（该文件已入库，只是
            # 正文不完整）—— 但至少同步能继续走。
            if time.time() - _t_start > _EPUB_OCR_TIME_BUDGET_S:
                logger.warning(
                    "%s: 扫描版 OCR 超过时间预算 %.0fs（%d/%d 张已完成），"
                    "提前结束本次 OCR——已识别部分仍会入库，剩余内容本次不处理",
                    name, _EPUB_OCR_TIME_BUDGET_S, done, len(tasks))
                if log:
                    log("    ├─ 扫描版 OCR 超时预算 %.0fs，已处理 %d/%d 张",
                        _EPUB_OCR_TIME_BUDGET_S, done, len(tasks))
                timed_out = True

    # 汇总一次「引擎命中/回退」情况：这是日志区判断「ocrmac 是否真在生效」
    # 的唯一依据。旧实现里回退完全无声（异常只记 debug、空结果干脆不记），
    # 于是整本走 RapidOCR 也看不出原因。
    _mac = _stats.get("mac_hit", 0)
    _fb = _stats.get("fallback", 0)
    _reasons = _stats.get("reasons", {})
    _rapid_won = _stats.get("rapid_won", 0)
    if _mac or _fb:
        _detail = ", ".join(
            f"{k}={v}" for k, v in sorted(_reasons.items())) or "无"
        logger.info(
            "%s: 扫描版 OCR 引擎统计 —— ocrmac 直接采信 %d 张，"
            "补充/改用 RapidOCR %d 张（其中 RapidOCR 产出更完整 %d 张）；"
            "回退原因: %s",
            name, _mac, _fb, _rapid_won, _detail)
    if not parts_map:
        return None
    return "\n\n".join(parts_map[i] for i in sorted(parts_map))


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
                # 打标记：让兜底路径 parsers._extract_mobi 跳过它自己的那次
                # calibre 调用——同一个文件、同一个工具，没必要再付一次完整
                # 转换（最坏 600s + 120s 才走到裸解析）。
                mark_ebook_convert_failed(filepath)
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
        # 正文极短 → 可能是扫描版：进入「扫描版 OCR 串行闸」后再读全本判定。
        # 闸门保证同一时刻只有 1 本扫描书在 read_epub + OCR（峰值内存数 GB），
        # 文字版（正文够长）根本不进闸，保持 2 路并行（用户要求：扫描版串行、
        # 文字版 2 并行）。profile 只算一次，OCR 段直接复用，避免重复读整本。
        if len(_WS_RE.sub("", body_md)) < _EPUB_SCAN_MD_MIN_CHARS:
            with _scanned_epub_gate(filepath):
                prof = _epub_scan_profile(src)
                if _is_scanned_profile(prof):
                    logger.info("%s: 判定为扫描图片版 epub，改用图片 OCR 取正文",
                                os.path.basename(filepath))
                    ocr_md = ocr_epub_images(src, profile=prof)
                    if ocr_md:
                        return ocr_md, toc, None
                    logger.warning(
                        "%s: 扫描版图片 OCR 未取到内容，回退 pandoc 结果",
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
    max_chars: int,
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

    ``max_chars`` **不设默认值**（必传）：唯一调用方是 engine 的 epub 路径，
    显式传 ``engine._chunk_size``（即 config.rag_chunk_size）。旧的
    ``=1200`` 默认值是个陷阱——未来新增调用点一旦漏传，epub 超长章节就
    会按 1200 切，与用户在设置面板配置的切片大小脱钩。
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
