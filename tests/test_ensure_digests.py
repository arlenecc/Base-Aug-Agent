"""Tests for ensure_digests_for_cached_documents (补做缺失缩略版本)."""
from __future__ import annotations

import os

import pytest

from agent.rag.engine import RAGEngine, _safe_filename


@pytest.fixture()
def engine(tmp_path, fake_ef):
    kb = tmp_path / "kb"
    kb.mkdir()
    return RAGEngine(
        workspace=str(tmp_path / "ws"),
        knowledge_base=str(kb),
        embedding_function=fake_ef,
    )


def _write_cached_markdown(engine, doc_name: str, markdown: str) -> str:
    """模拟旧版同步结果：写入 markdown 缓存，但不写入 documents 表。"""
    md_dir = engine.markdown_dir
    os.makedirs(md_dir, exist_ok=True)
    fname = _safe_filename(doc_name) + ".md"
    path = os.path.join(md_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return fname


def test_ensure_digests_backfills_cached_documents(engine):
    """已缓存 markdown 但无 digest 的文档，应被补做进 documents 表。"""
    _write_cached_markdown(
        engine, "亲密关系.pdf",
        "# 第一章 沟通\n\n沟通很重要。\n\n# 第二章 冲突\n\n冲突处理需要合作。",
    )
    _write_cached_markdown(
        engine, "人生的活法.pdf",
        "# 第一章 活法\n\n人生需要规划。",
    )

    # 补做前：documents 表为空
    assert engine.list_documents() == []

    created = engine.ensure_digests_for_cached_documents()
    assert created == 2

    docs = sorted(engine.list_documents())
    assert "亲密关系.pdf" in docs
    assert "人生的活法.pdf" in docs

    # digest 应包含目录和摘要
    outline = engine.get_document_outline("亲密关系.pdf")
    assert "# 目录" in outline["digest"]
    assert "第一章 沟通" in outline["digest"]


def test_ensure_digests_is_idempotent(engine):
    """重复调用不应重复写入或产生重复记录。"""
    _write_cached_markdown(engine, "a.pdf", "# 第一章\n\n内容。")
    assert engine.ensure_digests_for_cached_documents() == 1
    # 第二次调用：已有 digest，不重复补做
    assert engine.ensure_digests_for_cached_documents() == 0
    # 列表里只有一个文档
    assert engine.list_documents() == ["a.pdf"]


def test_ensure_digests_skips_empty_markdown(engine):
    """空 markdown 缓存应被跳过。"""
    _write_cached_markdown(engine, "empty.pdf", "")
    assert engine.ensure_digests_for_cached_documents() == 0
    assert engine.list_documents() == []


def test_ensure_digests_no_cache_dir(engine):
    """无 markdown 缓存目录时返回 0。"""
    assert engine.ensure_digests_for_cached_documents() == 0
