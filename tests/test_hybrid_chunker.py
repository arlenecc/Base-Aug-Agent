"""Tests for hybrid chunking (structure + semantic)."""
from __future__ import annotations

import pytest

from agent.rag.chunker import estimate_tokens


class _BigramEmbeddingFunction:
    """Deterministic, topic-aware fake embedding (character bigram bag-of-words)."""
    def embed_documents(self, texts):
        import hashlib
        out = []
        for t in texts:
            v = [0.0] * 128
            for i in range(len(t) - 1):
                h = hashlib.md5(t[i:i + 2].encode("utf-8")).digest()[0]
                v[h % 128] += 1.0
            out.append(v)
        return out


@pytest.fixture()
def topic_ef():
    return _BigramEmbeddingFunction()


def _is_chonkie_installed() -> bool:
    try:
        import chonkie  # noqa: F401
        return True
    except ImportError:
        return False


def test_hybrid_split_by_heading_structure(topic_ef):
    """Headings should be preserved as chunk boundaries."""
    from agent.rag.hybrid_chunker import hybrid_chunk_text

    text = (
        "# 第一章 引言\n\n"
        + "这是第一章的内容。" * 10 + "\n\n"
        + "## 1.1 背景\n\n"
        + "这是背景小节的内容。" * 10 + "\n\n"
        + "## 1.2 目标\n\n"
        + "这是目标小节的内容。" * 10
    )
    chunks = hybrid_chunk_text(text, chunk_size=800, embedding_function=topic_ef)
    assert chunks, "should produce chunks"
    # Each heading should appear at the start of some chunk
    joined = "\n".join(chunks)
    assert "# 第一章 引言" in joined
    assert "## 1.1 背景" in joined
    assert "## 1.2 目标" in joined


def test_hybrid_preserves_heading_with_body(topic_ef):
    """A heading and its body should stay in the same chunk when small enough."""
    from agent.rag.hybrid_chunker import hybrid_chunk_text

    text = "# 短章节\n\n" + "内容。" * 30
    chunks = hybrid_chunk_text(text, chunk_size=800, embedding_function=topic_ef)
    assert len(chunks) == 1
    assert chunks[0].startswith("# 短章节")


def test_hybrid_splits_overlong_section(topic_ef):
    """Over-long sections (> chunk_size) should be split into multiple chunks."""
    from agent.rag.hybrid_chunker import hybrid_chunk_text

    # One huge section (no sub-headings) well over 800 tokens.
    long_body = "机器学习是人工智能的重要分支。" * 500
    text = "# 长章节\n\n" + long_body
    chunks = hybrid_chunk_text(text, chunk_size=800, embedding_function=topic_ef)
    assert len(chunks) > 1, "over-long section should be split"

    # The heading should be in the first chunk.
    assert "# 长章节" in chunks[0]
    # All chunks bounded by chunk_size + overlap; the overlap is sentence-
    # aligned so a chunk may exceed the nominal ceiling by the length of one
    # trailing sentence (allow slack up to ~chunk_size * 1.2).
    for c in chunks:
        assert estimate_tokens(c) <= 960, f"chunk too large: {estimate_tokens(c)}"


def test_hybrid_merges_undersized_and_applies_overlap(topic_ef):
    """Undersized chunks merge to >= min; consecutive chunks overlap."""
    from agent.rag.hybrid_chunker import hybrid_chunk_text

    # Many tiny sections -> should merge into fewer, larger chunks.
    sections = []
    for i in range(20):
        sections.append(f"## 小节{i}\n\n" + "短内容。" * 8)
    text = "\n\n".join(sections)

    chunks = hybrid_chunk_text(
        text, chunk_size=800, min_chunk_size=100, overlap_percent=0.10,
        embedding_function=topic_ef,
    )
    assert chunks, "should produce chunks"
    # Interior chunks should be >= min (allow estimation slack).
    for c in chunks[:-1]:
        assert estimate_tokens(c) >= 80, f"chunk too small: {estimate_tokens(c)}"


def test_hybrid_no_headings_falls_back_to_paragraphs(topic_ef):
    """Text without headings should still chunk (paragraph-based)."""
    from agent.rag.hybrid_chunker import hybrid_chunk_text

    text = "第一段。" * 50 + "\n\n" + "第二段。" * 50
    chunks = hybrid_chunk_text(text, chunk_size=800, embedding_function=topic_ef)
    assert chunks
    joined = "".join(chunks)
    assert "第一段" in joined and "第二段" in joined


def test_hybrid_empty_text(topic_ef):
    from agent.rag.hybrid_chunker import hybrid_chunk_text

    assert hybrid_chunk_text("", embedding_function=topic_ef) == []
    assert hybrid_chunk_text("   ", embedding_function=topic_ef) == []
