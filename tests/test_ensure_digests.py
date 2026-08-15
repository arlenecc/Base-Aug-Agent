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


def test_ensure_digests_refresh_stale_digest(engine):
    """refresh_stale=True 时，旧格式（含「章节摘要」段）的 digest 应被重刷。"""
    # 写入缓存 markdown，并先手动生成一个旧格式（含章节摘要段）的 digest。
    doc_name = "旧格式.pdf"
    _write_cached_markdown(engine, doc_name, "# 第一章\n\n内容。")
    engine.build_and_store_digest(doc_name, "# 第一章\n\n内容。")
    # 人为把 digest 改成旧格式（含废弃的「章节摘要」段）。
    store = engine._get_store()
    doc = store.get_document_digest(doc_name)
    assert doc is not None
    store.upsert_document(
        doc_name=doc_name,
        digest="# 目录\n\n\n\n# 章节摘要\n",
        markdown=doc["markdown"],
        chapters=doc["chapters"],
    )

    # 默认 refresh_stale=False：不重刷，返回 0。
    assert engine.ensure_digests_for_cached_documents() == 0

    # refresh_stale=True：重刷为纯目录结构。
    assert engine.ensure_digests_for_cached_documents(refresh_stale=True) == 1
    refreshed = engine.get_document_outline(doc_name)
    assert "# 章节摘要" not in refreshed["digest"]
    assert "# 目录" in refreshed["digest"]
    assert "第一章" in refreshed["digest"]


def test_ensure_digests_refresh_empty_toc(engine):
    """refresh_stale=True 时，目录为空的旧 digest 也应被重刷（或用文档名兜底）。"""
    doc_name = "无标题.pdf"
    # 纯文本、无 Markdown 标题的 markdown。
    _write_cached_markdown(engine, doc_name, "这是没有标题的纯文本内容。")
    engine.build_and_store_digest(doc_name, "这是没有标题的纯文本内容。")
    # 人为写入一个目录为空的旧 digest。
    store = engine._get_store()
    doc = store.get_document_digest(doc_name)
    store.upsert_document(
        doc_name=doc_name,
        digest="# 目录\n",
        markdown=doc["markdown"],
        chapters=doc["chapters"],
    )

    assert engine.ensure_digests_for_cached_documents(refresh_stale=True) == 1
    refreshed = engine.get_document_outline(doc_name)
    # 无标题文档：digest 兜底为文档名，不再只有空目录。
    assert doc_name in refreshed["digest"]
