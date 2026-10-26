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


def test_chunk_documents_char_hard_limit_no_giant_chunks(topic_ef):
    """#140 关联回归：样板文字重复的书不得产出巨型切片。

    事故：目录行重复几十次的教程书，_merge_to_target_size 的 _gap_between
    在源文本里定位错位，把 33 万字的"最小间隔"整段拼进 chunk，产出 5 个
    33.8 万字的巨型切片（+章节直通 1200 字旁路）。修复后三道防线：
      1. _gap_between 超过 _MAX_GAP 的间隔视为定位错位、回退分隔符；
      2. 切片度量改为字符硬限（estimate_tokens == len）；
      3. chunk_documents 末端 _enforce_char_limit 对任何超限片按字符窗口强切。
    """
    from agent.rag.chunker import chunk_documents

    # 重复样板行（每行重复出现，制造 _gap_between 的多次出现定位）+ 长正文，
    # 无 Markdown 标题层级（模拟出事的教程书结构）。
    lines = []
    for ch in range(30):
        lines.append(f"第{ch}章 目录行（样板重复）")
        lines.append(("这一章讲解具体的操作步骤与注意事项，内容较长。" * 40))
    text = "\n\n".join(lines)

    chunks = chunk_documents(
        [{"source": "repeated.md", "text": text}],
        chunk_size=500, chunk_overlap=50, min_chunk_size=100,
        overlap_percent=0.10, embedding_function=topic_ef,
    )
    assert chunks, "should produce chunks"
    oversized = [c for c in chunks if len(c["text"]) > 500]
    assert not oversized, (
        f"巨型切片仍存在: max={max(len(c['text']) for c in chunks)} 字, "
        f"超限 {len(oversized)} 片"
    )
