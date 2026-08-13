"""Text chunker — splits documents into overlapping chunks for RAG.

Chunk sizes are measured in *tokens* (estimated via character-based heuristic:
~2 chars/token for Chinese, ~3 chars/token for English).
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


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
    chunk_size: int = 800,
    chunk_overlap: int = 80,
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
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    min_chunk_size: int = 100,
    overlap_percent: float = 0.10,
) -> List[Dict[str, Any]]:
    """Chunk multiple documents with metadata.

    Args:
        documents: List of dicts with 'source' (filepath) and 'text' keys.
        chunk_size: Target maximum chunk size in **tokens** (800).
        chunk_overlap: Overlap between chunks in **tokens** (default 10% of
            ``chunk_size`` = 80).
        min_chunk_size: Target minimum chunk size in **tokens** (100) — used by
            the semantic path to merge undersized chunks.
        overlap_percent: Fraction of ``chunk_size`` used as overlap between
            consecutive chunks (0.10 => 80 tokens), preserving continuity —
            used by the semantic path.

    Returns:
        List of dicts with 'source', 'chunk_index', and 'text' keys.

    Chunking strategy: hybrid chunking (document structure + semantic split
    via chonkie SemanticChunker + shared nomic-embed-text model) when
    available, falling back to plain semantic chunking, then the recursive
    chunker.
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        source = doc.get("source", "unknown")
        text = doc.get("text", "")
        total_tokens = estimate_tokens(text)
        logger.debug("      文档切片: %s (%d 字符, 约 %d tokens)", source, len(text), total_tokens)
        chunks = _chunk_with_hybrid_fallback(
            text, chunk_size, chunk_overlap, min_chunk_size, overlap_percent,
        )
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "chunk_index": i,
                "text": chunk,
            })
    return all_chunks


def _chunk_with_hybrid_fallback(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    overlap_percent: float,
) -> List[str]:
    """Hybrid-chunk first (structure + semantic), then semantic, then recursive.

    The fallback chain is deliberately ordered from highest to lowest
    fidelity:
      1. hybrid_chunk_text  — Markdown-structure-aware + semantic split
      2. semantic_chunk_text — pure semantic split (chonkie)
      3. chunk_text         — recursive character split (no model needed)
    """
    # 1. Hybrid (structure + semantic).
    try:
        from .hybrid_chunker import hybrid_chunk_text

        hybrid = hybrid_chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            overlap_percent=overlap_percent,
        )
        if hybrid:
            return hybrid
    except Exception:  # pragma: no cover - defensive
        logger.warning("hybrid chunking raised, falling back to semantic", exc_info=True)

    # 2. Semantic (chonkie).
    try:
        from .semantic_chunker import semantic_chunk_text

        semantic = semantic_chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            overlap_percent=overlap_percent,
        )
        if semantic is not None:
            return semantic
    except Exception:  # pragma: no cover - defensive; semantic_chunk_text already guards
        logger.warning("semantic chunking raised, falling back to recursive", exc_info=True)

    # 3. Recursive.
    return chunk_text(text, chunk_size, chunk_overlap)


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
    for sep_idx, sep in enumerate(separators):
        if sep == "":
            # Character-level split
            return _split_by_tokens(text, chunk_size, chunk_overlap)

        if sep not in text:
            continue

        parts = text.split(sep)
        chunks: List[str] = []
        current = ""
        # Maintain running counts to avoid O(n²) rescans of `current`
        cur_cjk = 0
        cur_other = 0

        for part in parts:
            if current:
                sep_cjk, sep_other = _char_counts(sep)
                part_cjk, part_other = _char_counts(part)
                new_cjk = cur_cjk + sep_cjk + part_cjk
                new_other = cur_other + sep_other + part_other
                new_tokens = new_cjk // 2 + new_other // 3
                if new_tokens <= chunk_size:
                    cur_cjk = new_cjk
                    cur_other = new_other
                    current = current + sep + part
                else:
                    if current.strip():
                        chunks.append(current)
                    # If a single part is too large, split it further
                    if _token_count(part) > chunk_size:
                        sub_chunks = _split_recursive(
                            part, separators[sep_idx + 1:],
                            chunk_size, chunk_overlap,
                        )
                        chunks.extend(sub_chunks)
                        current = ""
                        cur_cjk = 0
                        cur_other = 0
                    else:
                        current = part
                        cur_cjk, cur_other = _char_counts(part)
            else:
                # First part
                part_cjk, part_other = _char_counts(part)
                if part_cjk // 2 + part_other // 3 <= chunk_size:
                    current = part
                    cur_cjk = part_cjk
                    cur_other = part_other
                else:
                    # Single part is too large — split further
                    sub_chunks = _split_recursive(
                        part, separators[sep_idx + 1:],
                        chunk_size, chunk_overlap,
                    )
                    chunks.extend(sub_chunks)
                    # Don't reset current — next part starts fresh

        if current.strip():
            chunks.append(current)

        if chunks:
            return _merge_overlap(chunks, chunk_overlap, sep)

    # Fallback: character-level split (was previously returning an oversized chunk)
    return _split_by_tokens(text, chunk_size, chunk_overlap)


def _char_counts(text: str) -> tuple:
    """Return (cjk_count, other_count) for incremental token estimation."""
    cjk = 0
    other = 0
    for ch in text:
        if 0x4e00 <= ord(ch) <= 0x9fff:
            cjk += 1
        else:
            other += 1
    return cjk, other


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
    """Add token-based overlap between consecutive chunks.

    Prepends the tail of the previous chunk to the current chunk so that
    retrieval has surrounding context at chunk boundaries. The separator
    is included so the result remains a substring of the original text.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    merged: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = merged[-1]
        curr = chunks[i]

        # Take last N tokens from prev as overlap.
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

            # Prepend overlap to current chunk for context continuity.
            # prev_end + sep + curr is still a substring of the source text
            # because prev_end is a suffix of the previous segment and curr
            # is the next segment in the original split.
            if prev_end.strip():
                merged.append(prev_end + sep + curr)
            else:
                merged.append(curr)
        else:
            merged.append(curr)

    return merged