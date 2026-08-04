"""Text cleaner — removes garbage, tags, and normalizes whitespace."""
from __future__ import annotations

import re
from typing import List


def clean_text(text: str) -> str:
    """Clean extracted text: remove garbage tags, normalize whitespace,
    strip non-printable characters, and collapse redundant blank lines."""
    if not text:
        return ""

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

    # 3. Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)

    # 4. Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # 5. Remove control characters except common whitespace (\n, \t, \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # 6. Replace various Unicode whitespace with normal space
    text = re.sub(r'[\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]', ' ', text)

    # 7. Remove zero-width characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]', '', text)

    # 8. Collapse multiple spaces (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # 9. Remove lines that are just garbage (very short, no letters/digits)
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

    return text


def normalize_markdown(text: str) -> str:
    """Convert cleaned text to clean markdown format."""
    if not text:
        return ""

    lines = text.split('\n')
    result: List[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append('')
            continue

        # Detect code blocks
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            result.append(stripped)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Detect headings
        if re.match(r'^#{1,6}\s', stripped):
            result.append(stripped)
            continue

        # Detect list items
        if re.match(r'^[\*\-\+]\s', stripped):
            result.append(stripped)
            continue

        if re.match(r'^\d+\.\s', stripped):
            result.append(stripped)
            continue

        # Regular paragraph
        result.append(stripped)

    return '\n\n'.join(
        p for p in '\n'.join(result).split('\n\n')
        if p.strip()
    )