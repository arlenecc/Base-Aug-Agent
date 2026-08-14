"""Document outline extraction — build a TOC + per-chapter summary digest.

Given docling-produced Markdown, this module:
  1. Parses the heading structure (``#`` … ``######``) into a chapter tree.
  2. Extracts the full text of each chapter.
  3. Produces a condensed "缩略版本" (outline digest): the full TOC plus a
     ≤50-char summary per chapter that keeps key data points and entities.

The digest is the *meta-context* used by Meta-context + Targeted RAG: the
model first reads the whole document structure + per-chapter gist, then plans
which chapters to drill into, and RAG retrieves only those passages.

Chapter summaries can be produced by an LLM (``summarizer`` callable) or, when
no LLM is available, fall back to a local extractive first-sentence heuristic
so the digest is always populated during ingestion.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# When no LLM summarizer is provided, keep at most this many leading chars of
# the chapter as a stand-in "summary".
_FALLBACK_SUMMARY_CHARS = 50


class Chapter:
    """A single heading + its body text."""

    __slots__ = ("level", "title", "text", "summary")

    def __init__(self, level: int, title: str, text: str, summary: str = ""):
        self.level = level
        self.title = title
        self.text = text
        self.summary = summary


def extract_chapters(markdown: str) -> List[Chapter]:
    """Split Markdown into chapters at heading boundaries.

    Each chapter starts at a heading and extends to the next heading of the
    same or higher level.  Text before the first heading is attached as a
    level-0 "前言" chapter.  Empty chapters are dropped.
    """
    if not markdown or not markdown.strip():
        return []

    headings = list(_HEADING_RE.finditer(markdown))
    chapters: List[Chapter] = []

    # Text before the first heading.
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


def build_toc(chapters: List[Chapter]) -> str:
    """Render the table of contents from a chapter list."""
    lines = []
    for c in chapters:
        indent = "  " * max(0, c.level - 1) if c.level > 0 else ""
        prefix = "#" * c.level if c.level > 0 else ""
        lines.append(f"{indent}{prefix} {c.title}".rstrip())
    return "\n".join(lines)


def _fallback_summary(text: str, max_chars: int = _FALLBACK_SUMMARY_CHARS) -> str:
    """Extractive fallback: first sentence(s), truncated to max_chars."""
    if not text:
        return ""
    # Take up to the first sentence boundary(s), then hard-truncate.
    s = text.strip()
    # Prefer ending at a CJK or ASCII sentence terminator.
    cut = max_chars
    for term in ("。", "！", "？", ". ", "! ", "? "):
        idx = s.find(term)
        if 0 < idx < cut:
            cut = idx + len(term.rstrip())
            break
    summary = s[:cut].strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip() + "…"
    return summary


def summarize_chapters(
    chapters: List[Chapter],
    summarizer: Optional[Callable[[str, str], str]] = None,
    max_summary_chars: int = 50,
) -> List[Chapter]:
    """Populate each chapter's summary.

    ``summarizer(title, text) -> str`` is called when provided (an LLM-backed
    callable).  Otherwise a local extractive fallback is used.  Summaries are
    truncated to ``max_summary_chars``.
    """
    for c in chapters:
        if not c.text:
            c.summary = ""
            continue
        try:
            if summarizer is not None:
                summary = summarizer(c.title, c.text)
            else:
                summary = _fallback_summary(c.text)
        except Exception as e:
            logger.warning("章节摘要生成失败 (%s): %s，回退提取式摘要", c.title, e)
            summary = _fallback_summary(c.text)
        c.summary = _truncate(summary or "", max_summary_chars)
    return chapters


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def build_digest(
    markdown: str,
    summarizer: Optional[Callable[[str, str], str]] = None,
    max_summary_chars: int = 50,
) -> Tuple[str, List[Chapter]]:
    """Build the condensed digest (缩略版本) for a Markdown document.

    Returns ``(digest_text, chapters)`` where ``digest_text`` is the
    TOC + per-chapter summaries, and ``chapters`` is the full chapter list
    (useful for later targeted retrieval).
    """
    chapters = extract_chapters(markdown)
    summarize_chapters(chapters, summarizer=summarizer, max_summary_chars=max_summary_chars)

    lines = ["# 目录\n", build_toc(chapters), "\n\n# 章节摘要\n"]
    for c in chapters:
        label = f"{'#' * c.level} {c.title}" if c.level > 0 else "前言"
        summary = c.summary or "（无内容）"
        lines.append(f"**{label}**: {summary}")
    digest = "\n".join(lines)
    return digest, chapters


def find_chapters_by_title(chapters: List[Chapter], keyword: str) -> List[Chapter]:
    """Return chapters whose title contains ``keyword`` (for targeted RAG)."""
    if not keyword:
        return []
    kw = keyword.lower()
    return [c for c in chapters if kw in c.title.lower()]
