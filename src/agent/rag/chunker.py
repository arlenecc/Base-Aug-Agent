"""Text chunker — splits documents into overlapping chunks for RAG.

Chunk sizes are measured in *tokens* (estimated via character-based heuristic:
~1.25 chars/token for Chinese, ~3 chars/token for English).
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------

# 预编译正则 + 用 finditer 计数：estimate_tokens() 在切片热路径上被调用
# 非常频繁，re.findall 会把所有匹配物化成一个列表（整篇文档几万个中文字符
# = 几万个临时单字符串对象），纯属浪费。
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')


# 中文按每 token ≈1.25 字校准（1.2~1.3 的中点）：BERT 中文词表实测 ≈1 字/token，
# Qwen 等现代词表 ≈1.3~1.5。非中文（数字/标点/英文）按 2 字符/token 保守估：
# BERT 对数字与标点是逐字符 1 token（"1.2.3"=5 token），英文 BPE 实际 ~4 字符/token。
# 注意：对 bge-small-zh 这类 ≈1 字/token 的模型，中文真实 token ≈ 1.25×估算值、
# 数字密集文本 ≈1.4~1.6×，切片预算上限校验须按最坏比例放宽（见 config 注释）。
_TOKENS_PER_CJK_CHARS = 5   # 每 5 个中文字符估 4 个 token → 1.25 字/token
_CJK_TOKEN_NUM = 4
_OTHER_CHARS_PER_TOKEN = 2  # 数字/标点/英文：2 字符/token（保守）


def _tokens_from_counts(cjk: int, other: int) -> int:
    """由字符计数估算 token（与 estimate_tokens 同一公式，供增量计数热路径复用）。"""
    return cjk * _CJK_TOKEN_NUM // _TOKENS_PER_CJK_CHARS + other // _OTHER_CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """Roughly estimate token count.

    Chinese: ~1.25 chars per token.  English: ~3 chars per token.
    """
    chinese_chars = sum(1 for _ in _CJK_RE.finditer(text))
    other_chars = len(text) - chinese_chars
    return _tokens_from_counts(chinese_chars, other_chars)


def _token_count(text: str) -> int:
    return max(estimate_tokens(text), 1) if text else 0


# ---------------------------------------------------------------------------
# sentence-boundary fallback splitting
# ---------------------------------------------------------------------------

# 句末边界：中文句读（。！？；…）连同其后的行尾空白一起消费；英文句号只在
# 后跟空白或文本末尾时才算边界（避免切开 "3.5"、"e.g." 这类小数/缩写）；
# 换行也视为边界——无句读的标题/列表行能独立成块，不会被胶进下一句。
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？；…]+[ \t]*\n?|\.(?=\s)|\.(?=\Z)|\n")


def _iter_sentence_spans(text: str) -> List[tuple]:
    """Return (start, end) spans of sentence pieces whose union is all of ``text``.

    相邻 span 首尾相接（边界标点归前一句），按序拼接仍是原文的连续子串——
    与 semantic_chunker/_tail_by_tokens 的「切片必须是原文子串」约束一致。
    """
    spans: List[tuple] = []
    start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = m.end()
        if end > start:
            spans.append((start, end))
            start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def split_by_sentences(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    """Pack whole sentences into chunks of at most ``chunk_size`` tokens.

    兜底切分：保证分片只在句子结束处断开，绝不从句子中间切断。唯一例外是
    单句本身超过 ``chunk_size``（无句读的超长段落/表格行）——此时无法既保句
    完整又保长度上限，退回字符级 ``_split_by_tokens``。
    """
    if not text or not text.strip():
        return []
    if _token_count(text) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    # O(1) 去重（存哈希），与 _split_by_tokens 同款约定。
    seen: set = set()
    cur: List[str] = []
    cur_cjk = cur_other = 0

    def _flush() -> None:
        nonlocal cur, cur_cjk, cur_other
        merged = "".join(cur).strip()
        if merged:
            digest = hash(merged)
            if digest not in seen:
                seen.add(digest)
                chunks.append(merged)
        cur, cur_cjk, cur_other = [], 0, 0

    for s, e in _iter_sentence_spans(text):
        piece = text[s:e]
        pc, po = _char_counts(piece)
        if _tokens_from_counts(pc, po) > chunk_size:
            # 先把已攒的句子成块落盘，再对超长单句做字符级切分。
            if cur:
                _flush()
            chunks.extend(_split_by_tokens(piece, chunk_size, chunk_overlap))
            continue
        # 预算判断必须累计"字符数"后整体取 floor（与 _split_recursive 同款）：
        # 若按逐片 token 求和，floor 的分段损失会累积，合并结果可能超出上限。
        if cur and _tokens_from_counts(cur_cjk + pc, cur_other + po) > chunk_size:
            _flush()
        cur.append(piece)
        cur_cjk += pc
        cur_other += po
    if cur:
        _flush()
    return chunks


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
    embedding_function=None,
) -> List[Dict[str, Any]]:
    """Chunk multiple documents with metadata.

    ``embedding_function`` 会一路透传给 hybrid/semantic 切分器。RAGEngine 注入
    的自定义 EF（测试的 fake_ef、未来可替换的远端 embedding）必须能被语义
    切片用到——否则语义切片仍会去加载共享的本地 ONNX 模型，既浪费内存，
    在离线环境还会卡在模型下载上。

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
            embedding_function,
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
    embedding_function=None,
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
            embedding_function=embedding_function,
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
            embedding_function=embedding_function,
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
    apply_overlap: bool = True,
) -> List[str]:
    """Recursively split text at the most appropriate separator level.

    ``apply_overlap=False`` 供递归子调用使用：只打包不补 overlap，由最外层
    统一做一次 `_merge_overlap`。旧实现每层递归各自 merge，子层已带 overlap
    的块在父层再被叠加（双重 overlap，实测可到 chunk_size 的 1.6 倍），且
    混入 overlap 文本的块在原文里相位错位，`_gap_between` 对不齐、间隔取错，
    把不相邻的原文拼进 chunk。
    """
    if _token_count(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator in priority order
    for sep_idx, sep in enumerate(separators):
        if sep == "":
            # 句子级兜底：分片只在句子结束处断开（单句超限才字符级切）。
            chunks = split_by_sentences(text, chunk_size, chunk_overlap)
            return (
                _merge_overlap(chunks, chunk_overlap, sep, source=text)
                if apply_overlap else chunks
            )

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
                new_tokens = _tokens_from_counts(new_cjk, new_other)
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
                            apply_overlap=False,
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
                if _tokens_from_counts(part_cjk, part_other) <= chunk_size:
                    current = part
                    cur_cjk = part_cjk
                    cur_other = part_other
                else:
                    # Single part is too large — split further
                    sub_chunks = _split_recursive(
                        part, separators[sep_idx + 1:],
                        chunk_size, chunk_overlap,
                        apply_overlap=False,
                    )
                    chunks.extend(sub_chunks)
                    # Don't reset current — next part starts fresh

        if current.strip():
            chunks.append(current)

        if chunks:
            return (
                _merge_overlap(chunks, chunk_overlap, sep, source=text)
                if apply_overlap else chunks
            )

    # Final fallback: sentence-level split — chunks only break at sentence ends.
    chunks = split_by_sentences(text, chunk_size, chunk_overlap)
    return (
        _merge_overlap(chunks, chunk_overlap, "", source=text)
        if apply_overlap else chunks
    )


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
    Maintains running CJK/other character counts and computes tokens via
    ``_tokens_from_counts`` — matching ``estimate_tokens`` exactly
    but without rescanning the substring on every step.
    """
    chunks: List[str] = []
    # O(1) dedup instead of O(n) `in chunks`. 存哈希而非 chunk 全文：切片
    # 本身已经是文档全文的副本，再去重集合里各存一份会让峰值内存翻倍。
    seen: set = set()
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
            if _tokens_from_counts(cjk, other) > chunk_size:
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
        if chunk:
            digest = hash(chunk)
            if digest not in seen:
                seen.add(digest)
                chunks.append(chunk)

        if end >= text_len:
            break

        # Advance position with overlap — incremental token count
        if chunk_overlap <= 0:
            # 零重叠：直接推进。旧实现循环条件 >= 0 恒真，仍会先吃进 1 个
            # 字符再判断，导致永远有 1 字符重叠，与「零重叠」语义不符。
            pos = end
            continue
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
            if _tokens_from_counts(ov_cjk, ov_other) >= chunk_overlap:
                break
        pos = max(overlap_start, pos + 1)  # ensure forward progress

    return chunks


def _gap_between(source: Optional[str], left: str, right: str, fallback: str = "\n\n") -> str:
    """Return the exact text that separates ``left`` and ``right`` in ``source``.

    拼接相邻 chunk 时硬编码分隔符会在「源文本里两者的真实间隔不是该分隔符」
    时伪造出原文没有的内容（chunk 不再是原文子串）。这里在原文中定位两个
    片段的相邻出现，返回真实间隔；定位失败/间隔异常大时退回 fallback。

    left 可能在原文出现多次（重复段落/样板文本），str.find 命中更早实例后
    j 会跨越真实 chunk 边界，算出的 gap 把不属于这两个 chunk 的文本拼进
    结果。这里限定 gap 上限，超限时尝试 left 的下一个出现位置。
    """
    if not source or not left or not right:
        return fallback
    _MAX_GAP = 256
    # 周期性重复文本（重复段落/样板句）里 left 会出现多次，第一次出现后面
    # 跟的未必是 right——只接受"首个 ≤256 的间隔"会取到错位的大间隔，把不
    # 相邻的原文拼进 chunk（且撑爆 token 预算）。这里扫描多个出现位置取
    # **最小**间隔；完美相邻（间隔 0）立即返回。扫描次数封顶防 O(n²)。
    _MAX_SCANS = 256
    best: tuple | None = None  # (gap_len, gap)
    start = 0
    for _ in range(_MAX_SCANS):
        i = source.find(left, start)
        if i < 0:
            break
        j = source.find(right, i + len(left))
        if j < 0:
            break
        gap = source[i + len(left):j]
        if not gap:
            return gap
        if best is None or len(gap) < best[0]:
            best = (len(gap), gap)
        start = i + 1
    return best[1] if best is not None else fallback


def _merge_overlap(chunks: List[str], overlap: int, sep: str, source: Optional[str] = None) -> List[str]:
    """Add token-based overlap between consecutive chunks.

    Prepends the tail of the previous chunk to the current chunk so that
    retrieval has surrounding context at chunk boundaries. The separator
    is included so the result remains a substring of the original text.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    merged: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        # 取「原始前一段」的尾部，而不是 merged[-1] 的尾部。merged[-1]
        # 已经带上了一段重叠文本，若某段比 overlap 还短，向后回退会越界取
        # 到上一段的重叠部分，导致重叠逐段复合、chunk 无限膨胀。
        prev = chunks[i - 1]
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
                if _tokens_from_counts(m_cjk, m_other) >= overlap:
                    break
            prev_end = prev[boundary:]

            # Prepend overlap to current chunk for context continuity.
            # prev_end + gap + curr 仍是原文子串：prev_end 是前段后缀、curr
            # 是原文里的下一段。分隔符必须是两者在原文中的**真实间隔**——
            # chunks 列表会混入深层递归的 sub_chunks（外层按 "\n\n" 切、
            # 深层按句号/空格切），相邻片段之间未必存在外层 sep；硬编码
            # sep 会伪造出原文没有的字符（与 semantic_chunker 的同类 bug）。
            # 没有原文可查时退回 sep（与旧行为一致）。
            if prev_end.strip():
                gap = _gap_between(source, prev, curr, fallback=sep)
                merged.append(prev_end + gap + curr)
            else:
                merged.append(curr)
        else:
            merged.append(curr)

    return merged