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

    Uses character sliding window with token-count boundary checks.
    Falls back to sentence/whitespace boundaries when possible.
    """
    chunks: List[str] = []
    pos = 0  # character position
    text_len = len(text)

    while pos < text_len:
        # Find a character window that fits within token budget
        end = pos
        while end < text_len and _token_count(text[pos:end + 1]) <= chunk_size:
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
        if chunk and chunk not in chunks:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Advance position with overlap — find overlap boundary by token count
        overlap_start = end
        overlap_tokens = 0
        while overlap_start > pos and overlap_tokens < chunk_overlap:
            overlap_start -= 1
            overlap_tokens = _token_count(text[overlap_start:end])
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

        # Find the overlap boundary: take last N tokens from prev
        if _token_count(prev) > overlap:
            # Walk backward to find the overlap text
            boundary = len(prev)
            while boundary > 0 and _token_count(prev[boundary:]) < overlap:
                boundary -= 1
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