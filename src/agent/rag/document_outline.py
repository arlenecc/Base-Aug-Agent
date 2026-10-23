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

# 明显非标题的噪声行（作者、版权、出版社等）。
_NOISE_TITLES = {
    "copyright", "all rights reserved", "all right reserved", "isbn",
    "references", "references and citations", "img",
}

# PDF 页眉模式：「章节名 + 分隔符 + 纯数字页码」，如「第二章 → 23」、
# 「第十章 ／ 147」、「第六章 ～65」。这是书籍左/右页页眉（章节名+页码），
# 不是真正的标题，应整体丢弃。
_PDF_HEADER_PAGENO_RE = re.compile(
    r"^(.{2,40}?)\s*[→／/～~|—\-]\s*\d{1,4}$"
)

# 页眉变体 2：「章节名 + 空格 + 纯数字页码」（无分隔符），如「第二章 29」、
# 「第五章 753」「第九章 85」。书名/章节名后紧跟页码。
_PDF_HEADER_CHAPTER_PAGENO_RE = re.compile(
    r"^(第[%s0-9一二三四五六七八九十百]+[章卷篇部节]|[A-Za-z][A-Za-z0-9 .,&'()/:-]{2,40}?)\s+\d{1,4}$" % _CJK_NUM
)

# 页眉变体 3：「页码 + 分隔符 + 章节名」（页码在前），如「128 ／ 薄伽梵歌」。
_PDF_HEADER_PAGENO_CHAPTER_RE = re.compile(
    r"^\d{1,4}\s*[→／/～~|—\-]\s*\S.{0,38}$"
)

# 被截断的章节名（缺「第」字，如「三章」「一章」「二章」）——通常是页眉
# 「第三章」的「第」字与数字被拆到上一行所致，单独成行时是噪声。
_TRUNCATED_CN_CHAPTER_RE = re.compile(
    r"^[%s]{1,3}[章卷篇部]$" % _CJK_NUM
)

# 年份开头的正文行：如「1947）。《薄伽梵歌》是…」「2007 年12月」。
# 标题不会以 4 位年份开头（除非恰是年份书名，罕见），视为正文噪声。
_YEAR_PREFIX_RE = re.compile(r"^(?:19|20)\d{2}\s*[年）).、]")

# 强章节标记前缀：行以「第X章/节/卷/篇/部」开头。这类前缀是权威的章节号，
# 即使行内因并入了引文/正文而含句读，也交由 _normalize_title 清尾，不整体拒绝。
_CN_CHAPTER_PREFIX_RE = re.compile(
    r"^第\s*[0-9%s]+\s*[章卷篇部节]" % _CJK_NUM
)

# 「目录」标记归一化：去掉 Markdown 强调/标题符号与空白后比对，兼容
# 「**总目录**」「## 目 录」「目次」「Contents」等写法。
_TOC_MARKER_CANON = {
    "目录", "目錄", "目次", "总目录", "總目錄", "總目次", "全书目录",
    "全書目錄", "contents", "tableofcontents",
}


def _is_toc_marker(line: str) -> bool:
    """True if ``line`` is a TOC marker (目录/总目录/Contents …).

    Normalises away Markdown emphasis/heading symbols and whitespace so bold
    or heading-wrapped markers (``**总目录**``, ``## 目 录``) are recognised.
    """
    s = re.sub(r"[*_#`>\s]+", "", line or "").lower()
    return s in _TOC_MARKER_CANON


# 有效字符：中文 / 字母数字 / 常见中英文标点。用于判定 OCR 乱码行。
_MEANINGFUL_EXTRA = set(
    "，。！？、；：""''（）《》【】—…·.,;:!?()'\"-–—"
)


def _meaningful_ratio(line: str) -> float:
    """Fraction of characters that are meaningful text (CJK/alnum/punct)."""
    if not line:
        return 0.0
    good = sum(
        1 for c in line
        if c.isalnum() or "\u4e00" <= c <= "\u9fff" or c in _MEANINGFUL_EXTRA
    )
    return good / len(line)


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

def _find_headings(markdown: str) -> List[Tuple[int, str, int, int]]:
    """逐行扫描找出 Markdown 标题，**跳过围栏代码块内的 ``#``**。

    返回 ``[(level, title, start, end)]``（start/end 为原文绝对偏移）。
    围栏代码块里的 shell/Python 注释（``# comment``、``#!/bin/bash``）会被
    误认为标题，把代码样本切成假章节——hybrid_chunker 已做了同样的围栏跳过，
    这里补齐（否则目录 digest 会被代码注释污染）。
    """
    results: List[Tuple[int, str, int, int]] = []
    in_fence = False
    fence_marker = ""
    offset = 0
    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        if not in_fence:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = True
                fence_marker = stripped[:3]
                offset += len(line)
                continue
        else:
            # 只有同一种围栏才能闭合（``` 不能被 ~~~ 闭合，反之亦然）。
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            offset += len(line)
            continue
        m = _HEADING_RE.match(line)
        if m:
            results.append((
                len(m.group(1)), m.group(2).strip(),
                offset + m.start(), offset + m.end(),
            ))
        offset += len(line)
    return results


# ---------------------------------------------------------------------------
# Markdown 标题净化（docling 等提取器常把正文句子误判为标题）
# ---------------------------------------------------------------------------

# 章节锚点：命中即视为「真正的章节 / 前言 / 目录」，用于 (a) 作为内容起点、
# 剔除其前的书名页/版权页噪声；(b) 豁免句子式过滤（如「序二鉴往知来，学习
# 佛陀的人间关怀…」含逗号但确为标题）。
_ANCHOR_KEYWORDS = (
    "章", "卷", "节", "部", "篇", "附录", "后记", "译后记", "跋",
    "序言", "原序", "中译本序", "译者序", "前言", "引言", "引论", "导言",
    "绪论", "目录", "目次", "序", "contents", "table of contents",
)
_ANCHOR_KEYWORD_SET = {k.lower() for k in _ANCHOR_KEYWORDS}

# 「第X章 / 卷一 / X部 / 章三」两种语序，以及纯数字编号起步。
_ANCHOR_RE = re.compile(
    r"^(?:"
    r"(?:第\s*)?(?:[0-9]+|[一二三四五六七八九十百千零〇]+)\s*[章卷节部篇]"
    r"|"
    r"[章卷节部篇]\s*(?:之\s*)?(?:[0-9]+|[一二三四五六七八九十百千零〇]+)"
    r")"
)
# 中文序号起步（「一 个人和宇宙的关系」「二、灵魂意识」「三、心智回归」）。
# 序号后接 `、`/`.`/`．` 或空格分隔符（不要求再后有空格）。
_CN_ORDINAL_RE = re.compile(r"^[一二三四五六七八九十百零〇]+\s*(?:[、.．]|\s)")
# 「目录」类标记标题（保留为锚点用于裁剪，但最终从章节列表中丢弃）。
_TOC_MARKER_SET = {
    "目录", "目 录", "目錄", "目 錄", "目次", "contents", "table of contents",
}
# 中文句末/句中标点：标题几乎不会包含这些，命中多为被误判的正文句子。
_SENT_PUNCT = "。！？；，、"


def _is_anchor(title: str) -> bool:
    """True if ``title`` is a real chapter/section/preface/TOC anchor."""
    t = title.strip()
    if not t:
        return False
    if _ANCHOR_RE.match(t):
        return True
    if _CN_ORDINAL_RE.match(t):
        return True
    if _NUMBERED_RE.match(t):
        return True
    low = t.lower()
    if low in _ANCHOR_KEYWORD_SET:
        return True
    return any(low.startswith(k) for k in _ANCHOR_KEYWORDS)


def _is_sentence_like(title: str) -> bool:
    """True if a (non-anchor) heading looks like a body sentence, not a title.

    Real chapter titles are short noun phrases; docling frequently promotes full
    sentences (ending in 。/！/？ or riddled with ，/；) to headings.  Strip
    surrounding quotes/brackets first so a quoted title (``"因为生命是无限"``)
    is not mistaken for a sentence.  A title that *ends* with a colon/ellipsis is
    almost always a mis-detected clause (``我努力找尋幸福源頭：``).
    """
    t = title.strip().strip("\"'「」『』“”‘’（）()[]【】")
    if any(p in t for p in _SENT_PUNCT):
        return True
    if re.search(r"[!?;,]{1,}", t):
        return True
    if re.search(r"[：:]\s*$|[.．・…]{2,}\s*$", t):
        return True
    if len(t) > 40:
        return True
    return False


# 版权页 / 勒口 / 读者回函卡等「后 matter」噪声关键词（命中即丢弃）。
_COLOPHON_KEYWORDS = (
    "定價", "定価", "美術設計", "裝幀", "裝訂", "責任編輯", "封面", "封底",
    "讀者回函", "讀者服務", "ISBN", "版權", "印製", "印刷", "印次", "印張",
    "發行", "發行人", "出版者", "出版日期", "初版", "二版", "三版", "四版",
    "五版", "六版", "七版", "八版", "九版", "郵政", "劃撥", "帳號", "電話",
    "傳真", "網址", "部落格", "臉書", "地址", "台北", "台灣", "號", "出版社",
    "總經銷", "排版", "經銷", "劃撥", "公司", "國家圖書館", "出版品預行編目",
    "預行編目", "編目資料", "版權所有", "翻印必究", "印次",
)
_DECORATIVE_DOTS = "·・．•∙"


def _is_decorative_like(title: str) -> bool:
    """True if a title is a decorative dotted rendition (``哲·學·釣・感・森``)."""
    t = title.strip()
    if len(t) <= 1:
        return False
    return sum(1 for c in t if c in _DECORATIVE_DOTS) >= 2


def _is_colophon_like(title: str) -> bool:
    """True if a title looks like copyright/colophon back-matter.

    Catches price/date/phone lines, ISBN/CIP metadata, and Latin-only
    back-matter (English titles, ``My Daily Bread``) or Dewey-style call
    numbers (``IV.B502.44``) — none of which are real chapter titles.
    """
    t = title.strip()
    if any(k in t for k in _COLOPHON_KEYWORDS):
        return True
    if re.search(r"\d{2,4}\s*元", t):
        return True
    if re.search(r"^\d{3,4}\s*年", t):
        return True
    if re.search(r"20\d{2}\s*[)）]", t):
        return True
    if re.search(r"\d{2,3}\s*[-－]\d{2,3}[-－]\d{2,3}", t):
        return True
    # 杜威分类号 / 索书号（如 IV.B502.44、B502.44）。
    if re.search(r"^[IVXLC]+\.?\s*[A-Za-z]?\d{2,}", t):
        return True
    if re.search(r"图书在版编目|（CIP）|CIP数据", t):
        return True
    # 纯拉丁（无中文）且较长 → 多为英文书名 / 版权行，非章节标题。
    if len(t) >= 8 and not any("\u4e00" <= c <= "\u9fff" for c in t):
        return True
    return False


def _is_garbage_like(title: str) -> bool:
    """True if a title is pure garbage (``..`` / ``tld~`` / ``2004)`` …).

    Also flags OCR-spaced sentences where every character is isolated by spaces
    (``天 父 啊 請 幫 助 …``) — these are body sentences, not titles.
    """
    t = title.strip()
    if not t:
        return True
    if len(t) <= 5 and not any("\u4e00" <= c <= "\u9fff" for c in t):
        return True
    # 空格隔开的单字过多 → 逐字空格化的正文句子。
    tokens = [tok for tok in re.split(r"\s+", t) if tok]
    single = sum(1 for tok in tokens if len(tok) == 1)
    if len(tokens) >= 5 and single >= 4:
        return True
    return False


def _norm_title(t: str) -> str:
    """Normalise a title for *near*-duplicate detection.

    Drops whitespace, punctuation, decorative dots, and the function word
    ``的/旳`` (which OCR routinely drops or inserts) so OCR-corrupted variants of
    the same running-header chapter (``安頓心靈的七個步驟`` / ``安頓心靈七個步驟``)
    collapse to one entry.
    """
    t = re.sub(
        r'[\s，。、；：！？·・．…—\-_/\\()（）\[\]【】""\'\'「」『』<>~]',
        "", t,
    )
    t = t.replace("的", "").replace("旳", "")
    return t.lower()


def _cjk_only(t: str) -> str:
    """Keep only CJK ideographs (for fuzzy near-duplicate comparison)."""
    return re.sub(r"[^\u4e00-\u9fff]", "", t)


def _lcs_ratio(a: str, b: str) -> float:
    """Ratio of the longest common subsequence length to ``min(len(a), len(b))``."""
    if not a or not b:
        return 0.0
    # 经典 O(n*m) LCS，标题均很短，成本可忽略。
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ca = a[i - 1]
        for j in range(1, len(b) + 1):
            if ca == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)] / min(len(a), len(b))


def _is_near_duplicate(t: str, seen_cjk: List[str]) -> bool:
    """True if ``t`` (CJK-normalised) is a near-duplicate of an already-seen title.

    Collapses OCR-corrupted repetitions of the same running-header chapter
    (``安頓心靈七個步驟`` vs ``安頓心靈钓七個步驟`` …) without merging genuinely
    distinct sections: requires a long shared core (≥5 CJK chars) and a high
    LCS ratio (≥0.75).
    """
    cjk = _cjk_only(t)
    if len(cjk) < 5:
        return False
    for s in seen_cjk:
        if len(s) < 5:
            continue
        if _lcs_ratio(cjk, s) >= 0.75:
            return True
    return False


def extract_chapters(markdown: str) -> List[Chapter]:
    """Split Markdown into chapters at ``#`` heading boundaries.

    In addition to splitting on headings, this *cleans* docling-style noise:

      * A heading whose text looks like a body sentence (contains sentence
        punctuation, or is very long) is dropped — unless it is a genuine
        chapter anchor (``第X章`` / ``卷一`` / numbered / 序言 / 目录 …).
      * Leading front-matter (title page, copyright, publisher, ISBN) is
        discarded: everything before the first real chapter anchor is noise.
      * Duplicate titles (running-header boilerplate, e.g. a repeated section
        name) are collapsed to their first occurrence.
      * ``目录``/``Contents`` markers are dropped from the final list (but still
        act as the front-matter anchor).
    """
    if not markdown or not markdown.strip():
        return []

    headings = _find_headings(markdown)
    if not headings:
        return []

    # 1) 过滤明显非标题：正文句子 / 过长段落 / 装饰性圆点标题 / 版权页勒口 /
    #    无意义乱码；「锚点标题」豁免句子过滤，但仍可能被版权页/装饰规则丢弃。
    filtered = []
    for level, title, start, end in headings:
        t = title.strip()
        if not t:
            continue
        if _is_garbage_like(t):
            continue
        if _is_decorative_like(t):
            continue
        if _is_colophon_like(t):
            continue
        if not _is_anchor(t) and _is_sentence_like(t):
            continue
        filtered.append((level, t, start, end))

    # 2) 以首个「锚点标题」为内容起点，丢弃其前的书名页/版权页噪声。
    first_anchor_idx = next(
        (i for i, (lv, t, s, e) in enumerate(filtered) if _is_anchor(t)), None
    )
    if first_anchor_idx is not None:
        kept = filtered[first_anchor_idx:]
    else:
        # 没有任何锚点（如全文只有一个无结构标题）：退回不过滤、保留前言包裹。
        kept = filtered

    # 3) 去重（保留首次出现，含 OCR 近重合并），并丢弃「目录」类标记标题。
    seen: set = set()
    seen_cjk: List[str] = []
    dedup = []
    for lv, t, s, e in kept:
        if t.lower() in _TOC_MARKER_SET:
            continue
        key = _norm_title(t)
        if key in seen:
            continue
        # OCR 近重合并：同一标题被识别成多个错字变体时只保留首个。
        # 「锚点标题」（第X章/卷/序/编号…）带有权威章节号，绝不参与近重合并，
        # 以免把「第六十章」「第六十一章」这类仅差一字的相邻章节误并成一条。
        if not _is_anchor(t) and _is_near_duplicate(t, seen_cjk):
            continue
        seen.add(key)
        seen_cjk.append(_cjk_only(t))
        dedup.append((lv, t, s, e))

    if not dedup:
        return []

    chapters: List[Chapter] = []
    # 把首个保留标题之前的文本作为「前言」包裹 —— 但仅当没有「被裁掉的书名页/
    # 版权页噪声」（first_anchor_idx > 0 表示我们确实丢弃了前导标题，其前的文本
    # 是标题页垃圾，不应包装成前言）。若首个标题本身就是真正的章（idx==0）或全文
    # 无锚点（None），则保留前言包裹（兼容既有的前缀测试与无结构文档）。
    if first_anchor_idx is None or first_anchor_idx == 0:
        prefix = markdown[: dedup[0][2]].strip()
        if prefix:
            chapters.append(Chapter(0, "前言", prefix))

    for i, (level, title, _start, end) in enumerate(dedup):
        next_start = dedup[i + 1][2] if i + 1 < len(dedup) else len(markdown)
        body = markdown[end:next_start].strip()
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
    # 行内含句读（。！？；，）→ 正文句子或被并入正文的标题行，非干净标题。但以强
    # 章节标记（第X章/节/卷/篇/部）开头的行，其「第X章」前缀权威，交由
    # _normalize_title 去掉尾部引文/页码，不在此整体拒绝（否则会丢掉
    # 「第3章 标题 "引文"」这类真实章节）。
    if any(p in line for p in "。！？；，"):
        if not _CN_CHAPTER_PREFIX_RE.match(line):
            return False
    # 明显噪声。
    if line.lower() in _NOISE_TITLES:
        return False
    # 「目录」标记本身（含带空格/强调变体）不是标题。
    if _is_toc_marker(line):
        return False
    # 年份开头的正文行（如「1947）。《薄伽梵歌》…」「2007 年12月」）。
    if _YEAR_PREFIX_RE.match(line):
        return False
    # PDF 页眉（「第二章 → 23」「第十章 ／ 147」）不是标题。
    if _PDF_HEADER_PAGENO_RE.match(line):
        return False
    # 页眉变体：「章节名 + 页码」（「第二章 29」）和「页码 + 章节名」（「128 ／ 薄伽梵歌」）。
    if _PDF_HEADER_CHAPTER_PAGENO_RE.match(line):
        return False
    if _PDF_HEADER_PAGENO_CHAPTER_RE.match(line):
        return False
    # 被截断的章节名（「三章」「一章」）不是标题。
    if _TRUNCATED_CN_CHAPTER_RE.match(line):
        return False
    # OCR 乱码行：有效字符（中文/字母数字/常见标点）占比过低（大量 ①•⋯ 等）。
    if n >= 8 and _meaningful_ratio(line) < 0.5:
        return False
    return True


def _normalize_title(line: str) -> str:
    """Normalise a raw title line into a clean title.

    Strips EPUB link wrappers (``[TITLE](#anchor)`` → ``TITLE``), a trailing
    pull-quote merged onto the heading (``第9章 界面 ｜ 223 "…"``), a trailing
    page number (``第2章 优泰｜：61`` → ``第2章 优泰``), then collapses internal
    whitespace.
    """
    line = line.strip()
    m = _EPUB_LINK_RE.match(line)
    if m:
        line = m.group(1).strip()
    # 尾部 pull-quote（被并入标题的引文）："…" / "…"。
    line = re.sub(r'\s*["“][^"”]*["”]\s*$', "", line)
    # 尾部页码（章节名 ｜ 223 / 第2章 优泰|：61）。
    line = re.sub(r'[\s｜|/／→—\-~～:：]+\s*\d{1,4}\s*$', "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


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
    # rest 仅标点/符号（如页码碎片 "| "）→ 非标题。
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", rest):
        return None
    # rest 是长句子（含句读或过长）→ 正文编号列表/习题，非标题。单级编号
    # （"1."）在正文里常是列表项/习题句，标题通常很短；多级编号（"1.1"）
    # 几乎总是真标题，放宽长度上限。
    max_rest = 16 if "." not in num else 30
    if any(p in rest for p in "。！？；，") or len(rest) > max_rest:
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
        if _is_toc_marker(ln):
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
# Plain-text → Markdown (inject headings with the same heuristics)
# ---------------------------------------------------------------------------

def _classify_toc_entry(line: str) -> Optional[Tuple[int, str]]:
    """Classify a line *inside a TOC block* as a title.

    The TOC block is a curated list, so any line that survives ``_looks_like_title``
    is treated as a title (matching the acceptance used by
    :func:`_extract_from_toc_block`).  Level: ``第X节`` → 2, ``第X章/卷/篇/部``
    or numbered → its natural level, plain short title / ALL-CAPS → 1.
    """
    if not line or not _looks_like_title(line):
        return None
    title = _normalize_title(line)
    hit = (_cn_section_match(title) or _cn_chapter_match(title)
           or _numbered_match(title))
    if hit is not None:
        return hit
    return (1, title)


def _classify_title_line(lines: List[str], idx: int) -> Optional[Tuple[int, str]]:
    """Classify the line at ``idx`` as a *body* chapter title, or ``None``.

    Uses exactly the same matching rules as :func:`_extract_from_body_headings`:
    strong patterns (``第X章/节``, numbered) are accepted outright; the weak
    ALL-CAPS pattern additionally requires surrounding blank lines (so ordinary
    prose in ALL-CAPS is not mistaken for a heading).
    """
    line = lines[idx]
    if not line or not _looks_like_title(line):
        return None
    title = _normalize_title(line)
    hit = (_cn_section_match(title) or _cn_chapter_match(title)
           or _numbered_match(title))
    if hit is not None:
        return hit
    if _ALLCAPS_RE.match(title):
        prev_blank = idx == 0 or lines[idx - 1] == ""
        nxt_blank = idx + 1 >= len(lines) or lines[idx + 1] == ""
        if prev_blank and nxt_blank:
            return (1, title)
    return None


def _toc_block_span(lines: List[str]) -> Tuple[int, int]:
    """Return ``(marker_index, end_index)`` of a leading TOC block, else ``(-1, -1)``.

    The block runs from just after the "目录/Contents" marker up to (but not
    including) the first body line.  It terminates at:
      * a blank-line-separated **repeat** of a TOC entry (the body re-states a
        chapter already listed — the common "TOC → Chapter 1" transition); or
      * the first line that is neither a TOC entry nor blank (real body prose).

    This avoids over-extending the block into the first body heading.
    """
    for i, ln in enumerate(lines):
        if _is_toc_marker(ln):
            break
    else:
        return (-1, -1)
    seen_in_toc: set = set()
    end = len(lines)
    j = i + 1
    while j < len(lines):
        ln = lines[j]
        if not ln:
            j += 1
            continue
        entry = _classify_toc_entry(ln)
        if entry is not None:
            title = entry[1]
            if title in seen_in_toc:
                # body re-states a listed chapter → block ends here
                end = j
                break
            seen_in_toc.add(title)
            j += 1
            continue
        # non-title, non-blank line → real body prose begins
        end = j
        break
    return (i, end)


def convert_plain_text_to_markdown(text: str) -> Tuple[str, List[Chapter]]:
    """Convert unstructured plain text into markdown by injecting ``#`` headings
    at detected chapter-title lines.

    Reuses the *same* heuristics as :func:`extract_chapters_plain` (the existing
    plain-text chapter-structure recovery), but works positionally so each
    detected title can be turned into a real Markdown heading **in place** —
    turning an otherwise flat document into a structure-aware one that
    :mod:`hybrid_chunker` can split on (``#`` … ``######`` boundaries).

    Strategy:
      * Locate a leading "目录/Contents" block.  A TOC entry whose title also
        appears later in the body is treated as a duplicate copy and dropped
        (the body copy becomes the real heading).  TOC entries with **no** body
        counterpart become headings themselves — so a document that is *only* a
        TOC still gains structure.
      * Scan line-by-line; each detected title becomes ``"#" * level + " " +
        title``.  Body titles use the stricter :func:`_classify_title_line`
        rules (avoiding false positives in prose); TOC-block lines use the more
        permissive :func:`_classify_toc_entry` (a curated list is all titles).
        All other lines (and blank-line paragraph grouping) are preserved.

    Returns ``(markdown, chapters)`` where ``chapters`` mirrors the injected
    heading structure (titles only, empty body) — useful for the digest, though
    the canonical output is the markdown string fed to chunking/RAG.
    """
    if not text or not text.strip():
        return "", []
    lines = _split_lines(text)
    if not lines:
        return "", []

    toc_marker, toc_end = _toc_block_span(lines)

    # Titles that genuinely appear as body headings (so TOC copies can be dropped).
    body_title_texts: set = set()
    if toc_marker >= 0:
        for j in range(toc_end, len(lines)):
            hit = _classify_title_line(lines, j)
            if hit is not None:
                body_title_texts.add(hit[1])

    out: List[str] = []
    chapters: List[Chapter] = []
    seen: set = set()
    prev_blank = False

    for idx, line in enumerate(lines):
        # Always drop the TOC marker line itself.
        if toc_marker >= 0 and idx == toc_marker:
            continue

        in_toc = toc_marker >= 0 and toc_marker < idx < toc_end
        hit = (_classify_toc_entry(line) if in_toc
               else _classify_title_line(lines, idx))
        if hit is not None:
            level, title = hit
            if in_toc and title in body_title_texts:
                # TOC copy of a real body heading → drop (body copy is injected)
                continue
            if title in seen:
                # Duplicate title → keep as plain text, don't double the heading.
                if line:
                    out.append(line)
                    prev_blank = False
                continue
            seen.add(title)
            chapters.append(Chapter(level, title, ""))
            out.append("#" * level + " " + title)
            prev_blank = False
            continue

        # Normal line — preserve, normalising consecutive blank lines.
        if not line:
            if not prev_blank and out:
                out.append("")
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False

    markdown = "\n".join(out).strip()
    return markdown, chapters


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
