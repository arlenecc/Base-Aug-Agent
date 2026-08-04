"""Text chunker — splits documents into overlapping chunks for RAG.

Chunk sizes are measured in *tokens* (estimated via character-based heuristic:
~2 chars/token for Chinese, ~3 chars/token for English).
"""
from __future__ import annotations

import re
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Roughly estimate token count.

    Chinese: ~2 chars per token.  English: ~3 chars per token.
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return chinese_chars // 2 + other_chars // 3


def _token_count(text: str) -> int:
    return max(estimate_tokens(text), 1) if text else 0


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] | None = None,
) -> List[str]:
    """Split text into overlapping chunks using token-based sizing.

    Args:
        text: The text to split.
        chunk_size: Target size of each chunk in **tokens**.
        chunk_overlap: Overlap between chunks in **tokens**.
        separators: Priority-ordered list of separators to split on.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    if separators is None:
        separators = ["\n\n", "\n", "。", ". ", " ", ""]

    return _split_recursive(text, separators, chunk_size, chunk_overlap)


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Chunk multiple documents with metadata.

    Args:
        documents: List of dicts with 'source' (filepath) and 'text' keys.
        chunk_size: Target chunk size in **tokens**.
        chunk_overlap: Overlap between chunks in **tokens**.

    Returns:
        List of dicts with 'source', 'chunk_index', and 'text' keys.
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "chunk_index": i,
                "text": chunk,
            })
    return all_chunks


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _split_recursive(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Recursively split text at the most appropriate separator level."""
    if _token_count(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator in priority order
    for sep in separators:
        if sep == "":
            # Character-level split
            return _split_by_tokens(text, chunk_size, chunk_overlap)

        if sep not in text:
            continue

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if _token_count(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current)
                # If a single part is too large, split it further
                if _token_count(part) > chunk_size:
                    sub_chunks = _split_recursive(
                        part, separators[separators.index(sep) + 1:],
                        chunk_size, chunk_overlap,
                    )
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current)

        if chunks:
            return _merge_overlap(chunks, chunk_overlap, sep)

    # Fallback: return as-is
    return [text]


def _split_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text by character positions, respecting token budget.

    Uses incremental token counting: O(n) overall instead of O(n²).
    Maintains running CJK/other character counts and computes tokens
    as ``cjk // 2 + other // 3`` — matching ``estimate_tokens`` exactly
    but without rescanning the substring on every step.
    """
    chunks: List[str] = []
    seen: set = set()  # O(1) dedup instead of O(n) `in chunks`
    pos = 0  # character position
    text_len = len(text)

    while pos < text_len:
        # Find a character window that fits within token budget.
        # Incremental: add one char at a time, update running counts.
        end = pos
        cjk = 0
        other = 0
        while end < text_len:
            ch = text[end]
            if 0x4e00 <= ord(ch) <= 0x9fff:
                cjk += 1
            else:
                other += 1
            if cjk // 2 + other // 3 > chunk_size:
                break
            end += 1

        # Try to break at a natural boundary within the window
        if end < text_len:
            # Search backward for a sentence or paragraph break
            for break_char in ['\n', '。', '. ', ' ']:
                last_break = text.rfind(break_char, pos, end)
                if last_break > pos + (end - pos) // 2:
                    end = last_break + len(break_char)
                    break

        chunk = text[pos:end].strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            chunks.append(chunk)

        if end >= text_len:
            break

        # Advance position with overlap — incremental token count
        overlap_start = end
        ov_cjk = 0
        ov_other = 0
        while overlap_start > pos:
            ch = text[overlap_start - 1]
            if 0x4e00 <= ord(ch) <= 0x9fff:
                ov_cjk += 1
            else:
                ov_other += 1
            overlap_start -= 1
            if ov_cjk // 2 + ov_other // 3 >= chunk_overlap:
                break
        pos = max(overlap_start, pos + 1)  # ensure forward progress

    return chunks


def _merge_overlap(chunks: List[str], overlap: int, sep: str) -> List[str]:
    """Add token-based overlap between consecutive chunks."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    merged: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = merged[-1]
        curr = chunks[i]

        # Find the overlap boundary: take last N tokens from prev.
        # Incremental: walk backward, maintaining running counts.
        if _token_count(prev) > overlap:
            boundary = len(prev)
            m_cjk = 0
            m_other = 0
            while boundary > 0:
                ch = prev[boundary - 1]
                if 0x4e00 <= ord(ch) <= 0x9fff:
                    m_cjk += 1
                else:
                    m_other += 1
                boundary -= 1
                if m_cjk // 2 + m_other // 3 >= overlap:
                    break
            prev_end = prev[boundary:]

            # Find where this overlap appears in curr
            idx = curr.find(prev_end)
            if idx >= 0:
                merged.append(curr[idx:])
            else:
                merged.append(curr)
        else:
            merged.append(curr)

    return merged