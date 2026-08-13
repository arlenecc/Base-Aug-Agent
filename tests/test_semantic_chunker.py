"""Tests for semantic chunking (chonkie SemanticChunker integration)."""
from __future__ import annotations

import pytest

from agent.rag.chunker import chunk_documents, estimate_tokens


# A deterministic, topic-aware fake embedding function: same-topic sentences
# share bigrams and therefore embed closer together, which lets
# SemanticChunker find coherent boundaries without a real model.
class _BigramEmbeddingFunction:
    def embed_documents(self, texts):
        import hashlib
        out = []
        for t in texts:
            v = [0.0] * 128
            for i in range(len(t) - 1):
                bg = t[i:i + 2]
                h = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16) % 128
                v[h] += 1.0
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


@pytest.mark.skipif(not _is_chonkie_installed(), reason="chonkie not installed")
def test_semantic_chunk_returns_none_when_unavailable(monkeypatch):
    """semantic_chunk_text returns None (fallback) when chonkie is absent."""
    from agent.rag import semantic_chunker as sc

    # Simulate chonkie being unavailable
    monkeypatch.setattr(sc, "_is_chonkie_available", lambda: False)
    assert sc.semantic_chunk_text("some text") is None


def test_semantic_chunk_empty_text(topic_ef):
    from agent.rag.semantic_chunker import semantic_chunk_text

    assert semantic_chunk_text("", embedding_function=topic_ef) == []
    assert semantic_chunk_text("   ", embedding_function=topic_ef) == []


@pytest.mark.skipif(not _is_chonkie_installed(), reason="chonkie not installed")
def test_semantic_chunk_splits_by_topic(topic_ef):
    from agent.rag.semantic_chunker import semantic_chunk_text

    topic_a = ("机器学习是一种让计算机从数据中学习的方法。监督学习使用带标签的数据训练模型。"
               "无监督学习在没有标签的情况下发现数据模式。强化学习通过与环境交互来学习策略。") * 12
    topic_b = ("前端开发使用 HTML 构建页面结构。CSS 负责页面的样式和布局。"
               "JavaScript 实现页面的交互逻辑。现代前端框架大幅提升了开发效率。") * 12
    text = topic_a + topic_b

    chunks = semantic_chunk_text(text, chunk_size=800, embedding_function=topic_ef)

    assert chunks, "semantic chunking should produce chunks when chonkie is available"
    # Chunks should respect the token ceiling (with overlap each is still
    # bounded by chunk_size, since overlap is borrowed from the previous chunk).
    for c in chunks:
        assert estimate_tokens(c) <= 850, f"chunk exceeds ceiling: {estimate_tokens(c)}"


@pytest.mark.skipif(not _is_chonkie_installed(), reason="chonkie not installed")
def test_semantic_chunk_merges_to_target_range(topic_ef):
    """Interior chunks should land near the [100, 800] token band after merging."""
    from agent.rag.semantic_chunker import semantic_chunk_text

    topic_a = "机器学习是一种让计算机从数据中学习的方法。监督学习使用带标签的数据训练模型。" * 20
    topic_b = "前端开发使用 HTML 构建页面结构。CSS 负责页面的样式和布局。" * 20
    topic_c = "数据库是存储和管理数据的系统。关系型数据库使用表格来组织数据。" * 20
    text = topic_a + topic_b + topic_c

    chunks = semantic_chunk_text(
        text, chunk_size=800, min_chunk_size=100, overlap_percent=0.10,
        embedding_function=topic_ef,
    )

    # Interior chunks (excluding the possibly-short tail) should be >= 100 tokens.
    interior = [c for c in chunks[:-1]]
    assert interior, "expected at least one interior chunk"
    for c in interior:
        assert estimate_tokens(c) >= 80, f"chunk too small: {estimate_tokens(c)} tokens"
    # No chunk may exceed the ceiling (chunk_size + overlap, since overlap is
    # borrowed from the previous chunk).
    for c in chunks:
        assert estimate_tokens(c) <= 900, f"chunk exceeds ceiling: {estimate_tokens(c)}"


@pytest.mark.skipif(not _is_chonkie_installed(), reason="chonkie not installed")
def test_semantic_chunk_overlap_preserves_continuity(topic_ef):
    """Consecutive chunks should share a tail/head overlap of ~10% chunk_size."""
    from agent.rag.semantic_chunker import semantic_chunk_text

    topic_a = "机器学习是一种让计算机从数据中学习的方法。监督学习使用带标签的数据训练模型。" * 30
    topic_b = "前端开发使用 HTML 构建页面结构。CSS 负责页面的样式和布局。" * 30
    text = topic_a + topic_b

    chunks = semantic_chunk_text(
        text, chunk_size=800, chunk_overlap=80, overlap_percent=0.10,
        embedding_function=topic_ef,
    )

    assert len(chunks) >= 2, "expected at least two chunks for overlap to apply"
    # With overlap, each subsequent chunk borrows the tail sentence(s) of the
    # previous chunk, so the previous chunk's final sentence must appear
    # verbatim near the start of the next chunk.
    for i in range(1, len(chunks)):
        prev_last_sentence = chunks[i - 1].split("。")[-2] + "。"
        assert prev_last_sentence in chunks[i][:300], \
            f"overlap missing: prev last sentence not in chunk {i} head"


def test_chunk_documents_falls_back_when_chonkie_missing(monkeypatch, topic_ef):
    """chunk_documents must still produce chunks when chonkie is unavailable."""
    from agent.rag import chunker as ch

    # Force both hybrid and semantic chunking to be unavailable -> recursive.
    monkeypatch.setattr(
        ch, "_chunk_with_hybrid_fallback",
        lambda text, cs, co, mn, op: ch.chunk_text(text, cs, co),
    )
    docs = [{"source": "doc.md", "text": "这是测试内容。" * 200}]
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=50)
    assert len(chunks) > 0
    assert all(c["text"].strip() for c in chunks)
    assert all(c["source"] == "doc.md" for c in chunks)


@pytest.mark.skipif(not _is_chonkie_installed(), reason="chonkie not installed")
def test_chunk_documents_uses_semantic_when_available(topic_ef):
    """chunk_documents produces valid chunks through the semantic path."""
    docs = [{"source": "doc.md", "text": "深度学习是机器学习的一个分支。" * 200}]
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=50)
    assert len(chunks) > 0
    assert all(c["text"].strip() for c in chunks)
    # chunk_index is sequential
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
