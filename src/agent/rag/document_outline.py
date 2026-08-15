"""Document outline extraction — build a TOC (目录结构) digest.

Given parsed document text (docling Markdown, EPUB, or plain-text PDF), this
module produces a condensed "缩略版本" (outline digest): the document's table
of contents — heading titles with their hierarchy — as meta-context.

Extraction strategy (best-effort, in order):
  1. Standard Markdown headings (``#`` … ``######``).
  2. Plain-text heuristics for documents without Markdown headings:
     a. A "目录/Contents" block: a marker line (``目录``/``目 录``/``Contents``/
        ``Table of Contents``) followed by a run of one-title-per-line entries.
     b. Standalone chapter-title lines in the body:
        * Chinese "第N章/节/卷/篇/部" prefixes (Arabic or CJK numerals);
        * numbered headings ("1", "1.1", "1.1.1", "第一章"…);
        * ALL-CAPS short lines (common in EPUB headings).

The digest is the *meta-context* for Meta-context + Targeted RAG: the model
first reads the whole document structure, then plans which chapters to drill
into, and RAG retrieves only those passages.

We deliberately do NOT generate per-chapter summaries — the title hierarchy
alone is sufficient and avoids the cost/slowness/emptiness of LLM summaries.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# 中文章节标题：第{数字|中文数字}章/节/卷/篇/部/部分，后跟空格或标题。
# 也兼容无「第」字的数字+卷/章格式（如「1卷 我的财产告白」）。
_CJK_NUM = "零一二三四五六七八九十百千〇"
_CN_CHAPTER_RE = re.compile(
    r"^(?:第\s*)?([0-9]+|[%s]+)\s*[章卷篇部](?:[：:\s]+.+)?$" % _CJK_NUM
)
_CN_SECTION_RE = re.compile(
    r"^(?:第\s*)?([0-9]+|[%s]+)\s*节(?:[：:\s]+.+)?$" % _CJK_NUM
)
# 编号标题：1 / 1.1 / 1.1.1 / (1) / 1、 等，后跟标题文本。
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[\.、\s）)](.+)?$")
# 英文全大写标题行：≥3 个词的全大写字母，允许部分符号。
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9 .,&'()/:-]{3,}$")

# EPUB 目录链接格式：[TITLE](#anchor) → TITLE。
_EPUB_LINK_RE = re.compile(r"^\[(.+)\]\(#[^)]*\)$")

# HTML 注释 / 图片占位噪声。
_HTML_NOISE_RE = re.compile(r"^\s*<!--.*?-->\s*$")

# 「目录」标记行（用于识别目录列表段）。
_TOC_MARKERS = (
    "目录", "目 录", "目錄", "contents", "table of contents",
)

# 明显非标题的噪声行（作者、版权、出版社等）。
_NOISE_TITLES = {
    "copyright", "all rights reserved", "all right reserved", "isbn",
    "references", "references and citations", "img",
}


class Chapter:
    """A single heading + its body text."""

    __slots__ = ("level", "title", "text", "summary")

    def __init__(self, level: int, title: str, text: str, summary: str = ""):
        self.level = level
        self.title = title
        self.text = text
        self.summary = summary


# ---------------------------------------------------------------------------
# Markdown headings
# ---------------------------------------------------------------------------

def extract_chapters(markdown: str) -> List[Chapter]:
    """Split Markdown into chapters at ``#`` heading boundaries."""
    if not markdown or not markdown.strip():
        return []

    headings = list(_HEADING_RE.finditer(markdown))
    chapters: List[Chapter] = []

    if headings:
        prefix = markdown[: headings[0].start()].strip()
        if prefix:
            chapters.append(Chapter(0, "前言", prefix))

    for i, m in enumerate(headings):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        body = markdown[start:end].strip()
        if body or title:
            chapters.append(Chapter(level, title, body))

    return chapters


# ---------------------------------------------------------------------------
# Plain-text heuristics
# ---------------------------------------------------------------------------

def _split_lines(text: str) -> List[str]:
    """Split into lines with CRLF/CR normalised (blank lines preserved)."""
    return [ln.strip() for ln in re.split(r"\r\n|\r|\n", text)]


def _looks_like_title(line: str) -> bool:
    """True if a line looks like a standalone heading (short, no sentence end)."""
    if _HTML_NOISE_RE.match(line):
        return False
    n = len(line)
    if n < 2 or n > 80:
        return False
    # 不以句末标点结尾（标题一般无句号/逗号/分号）。
    if line[-1] in "。，；：！？.,;:!?":
        return False
    # 明显噪声。
    if line.lower() in _NOISE_TITLES:
        return False
    return True


def _normalize_title(line: str) -> str:
    """Normalise a raw title line into a clean title.

    Strips EPUB link wrappers (``[TITLE](#anchor)`` → ``TITLE``) and other
    obvious wrappers, then collapses internal whitespace.
    """
    line = line.strip()
    m = _EPUB_LINK_RE.match(line)
    if m:
        line = m.group(1).strip()
    line = re.sub(r"\s+", " ", line)
    return line


def _cn_section_match(line: str) -> Optional[Tuple[int, str]]:
    """Match a Chinese section heading (``第X节``) → (level=2, title)."""
    m = _CN_SECTION_RE.match(line)
    if m:
        return (2, line)
    return None


def _cn_chapter_match(line: str) -> Optional[Tuple[int, str]]:
    """Match a Chinese chapter/volume/part heading → (level=1, title).

    ``第X章``/``X卷``/``第X篇``/``第X部``/``第X部分`` 等均为 level 1。
    """
    m = _CN_CHAPTER_RE.match(line)
    if m:
        return (1, line)
    return None


def _numbered_match(line: str) -> Optional[Tuple[int, str]]:
    """Match a numbered heading (``1.2.3 Title``) → (level, title)."""
    m = _NUMBERED_RE.match(line)
    if not m:
        return None
    num = m.group(1)
    level = num.count(".") + 1
    rest = (m.group(2) or "").strip()
    # 纯数字行（如页码）不算标题。
    if not rest:
        return None
    return (level, line)


def _extract_from_toc_block(lines: List[str]) -> List[Chapter]:
    """Extract titles from a "目录/Contents" block (one title per line).

    Locates the first TOC marker, then collects subsequent short lines as
    entries until a paragraph-length line (body text) is hit.  Level is inferred
    from the title's shape: chapter/volume → 1, section/numbered → 2, plain
    short titles → 1.
    """
    for i, ln in enumerate(lines):
        if ln.lower() in _TOC_MARKERS:
            break
    else:
        return []

    chapters: List[Chapter] = []
    seen = set()
    for ln in lines[i + 1:]:
        # 终止条件：正文段落通常以句末标点结尾（目录标题一般不会）。
        # 硬换行的 PDF 正文每行都短，无法靠行长度区分目录与正文，
        # 但正文句以「。，；！？」结尾，目录标题则不会。
        if len(ln) > 80 or (ln and ln[-1] in "。，；！？．,;!?"):
            break
        if not _looks_like_title(ln):
            continue
        title = _normalize_title(ln)
        hit = _cn_section_match(title) or _cn_chapter_match(title) or _numbered_match(title)
        if hit is not None:
            level, _ = hit
        elif _ALLCAPS_RE.match(title):
            level = 1
        else:
            # 目录里的普通短标题（如「自序」「我的财产告白」）→ level 1。
            level = 1
        if title in seen:
            continue
        seen.add(title)
        chapters.append(Chapter(level, title, ""))
    return chapters


def _extract_from_body_headings(lines: List[str]) -> List[Chapter]:
    """Scan the whole body for standalone chapter-title lines.

    Strong patterns (``第X章``/``第X节``/numbered) are accepted outright;
    weak patterns (ALL-CAPS) additionally require surrounding blank lines to
    avoid matching ordinary prose.
    """
    chapters: List[Chapter] = []
    seen = set()
    for idx, ln in enumerate(lines):
        if not ln or not _looks_like_title(ln):
            continue
        title = _normalize_title(ln)
        hit = _cn_section_match(title) or _cn_chapter_match(title) or _numbered_match(title)
        if hit is not None:
            level, _ = hit
        elif _ALLCAPS_RE.match(title):
            # 弱模式：要求前后有空行，避免把正文里的大写短语当标题。
            prev_blank = idx == 0 or lines[idx - 1] == ""
            nxt_blank = idx + 1 >= len(lines) or lines[idx + 1] == ""
            if not (prev_blank and nxt_blank):
                continue
            level = 1
        else:
            continue
        if title in seen:
            continue
        seen.add(title)
        chapters.append(Chapter(level, title, ""))
    return chapters


def extract_chapters_plain(text: str) -> List[Chapter]:
    """Extract a chapter list from plain text (no Markdown headings).

    Prefers a TOC block when present (more accurate and complete); otherwise
    falls back to scanning the body for standalone title lines.
    """
    lines = _split_lines(text)
    if not lines:
        return []

    chapters = _extract_from_toc_block(lines)
    if chapters:
        return chapters
    return _extract_from_body_headings(lines)


# ---------------------------------------------------------------------------
# Digest assembly
# ---------------------------------------------------------------------------

def build_toc(chapters: List[Chapter]) -> str:
    """Render the table of contents from a chapter list."""
    lines = []
    for c in chapters:
        indent = "  " * max(0, c.level - 1) if c.level > 0 else ""
        prefix = "#" * c.level if c.level > 0 else ""
        lines.append(f"{indent}{prefix} {c.title}".rstrip())
    return "\n".join(lines)


def build_digest(markdown: str) -> Tuple[str, List[Chapter]]:
    """Build the condensed digest (缩略版本) for a document.

    Tries Markdown headings first; if none are found, falls back to plain-text
    heuristics so pure-text PDFs/EPUBs still yield a meaningful TOC.

    Returns ``(digest_text, chapters)`` where ``digest_text`` is the TOC and
    ``chapters`` is the full chapter list (useful for targeted retrieval by
    title).
    """
    chapters = extract_chapters(markdown)
    if not chapters:
        chapters = extract_chapters_plain(markdown)

    lines = ["# 目录\n", build_toc(chapters)]
    digest = "\n".join(lines)
    return digest, chapters


def find_chapters_by_title(chapters: List[Chapter], keyword: str) -> List[Chapter]:
    """Return chapters whose title contains ``keyword`` (for targeted RAG)."""
    if not keyword:
        return []
    kw = keyword.lower()
    return [c for c in chapters if kw in c.title.lower()]
