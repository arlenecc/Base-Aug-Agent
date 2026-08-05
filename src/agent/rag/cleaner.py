"""Text cleaner — removes garbage, tags, and normalizes whitespace."""
from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean extracted text: remove garbage tags, normalize whitespace,
    strip non-printable characters, and collapse redundant blank lines."""
    if not text:
        return ""
    logger.debug("      清洗文本: 原始 %d 字符", len(text))

    # 1. Remove entire script/style blocks (including their content)
    text = re.sub(r'<script[^>]*>[\s\S]*?</script>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<style[^>]*>[\s\S]*?</style>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<noscript[^>]*>[\s\S]*?</noscript>', ' ', text, flags=re.IGNORECASE)

    # 2. Remove HTML comments
    text = re.sub(r'<!--[\s\S]*?-->', ' ', text)

    # 3. Remove remaining HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 4. Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&#x[0-9a-fA-F]+;', ' ', text)

    # 5. Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)

    # 6. Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # 7. Remove control characters except common whitespace (\n, \t, \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # 8. Replace various Unicode whitespace with normal space
    text = re.sub(r'[\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]', ' ', text)

    # 9. Remove zero-width characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]', '', text)

    # 10. Collapse multiple spaces (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # 11. Remove lines that are just garbage (very short, no letters/digits)
    lines = text.split('\n')
    cleaned_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
        # Skip lines that are mostly non-alphanumeric garbage
        alpha_ratio = sum(1 for c in stripped if c.isalnum() or c.isspace()) / max(len(stripped), 1)
        if alpha_ratio < 0.3 and len(stripped) < 20:
            continue
        # Skip lines that are just page numbers or headers
        if re.match(r'^\d{1,4}$', stripped):
            continue
        if re.match(r'^第\s*\d{1,4}\s*页$', stripped):
            continue
        if re.match(r'^Page\s+\d{1,4}$', stripped, re.IGNORECASE):
            continue
        # Skip separator lines (long runs of the same character)
        if re.match(r'^([\-=_*#~]{3,})\s*$', stripped):
            continue
        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    # 10. Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 11. Strip leading/trailing whitespace
    text = text.strip()

    logger.debug("      清洗文本完成: %d 字符", len(text))
    return text


def normalize_markdown(text: str) -> str:
    """Convert cleaned text to clean markdown format.

    代码块内的空行保留原样，不会被当作段落分隔符。
    非代码块段落之间用一个空行分隔。
    """
    if not text:
        return ""
    logger.debug("      转换 Markdown: %d 字符", len(text))

    lines = text.split('\n')
    result: List[str] = []
    in_code_block = False
    prev_blank = False  # 上一行是否是空行（用于合并连续空行）

    for line in lines:
        stripped = line.strip()

        # Detect code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            result.append(stripped)
            prev_blank = False
            continue

        if in_code_block:
            # 代码块内：保留原始行（包括空行），不做任何修改
            result.append(line)
            prev_blank = False
            continue

        if not stripped:
            # 非代码块的空行：确保段落之间只有一个空行（合并连续空行）
            if not prev_blank and result:
                result.append('')
            prev_blank = True
            continue

        # 非代码块的非空行：strip 后追加
        result.append(stripped)
        prev_blank = False

    # 去除末尾空行
    while result and not result[-1].strip():
        result.pop()

    output = '\n'.join(result)
    logger.debug("      转换 Markdown 完成: %d 行", len(result))
    return output