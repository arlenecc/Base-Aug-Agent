"""Tests for the documents metadata table (digest storage for Meta-context RAG)."""
from __future__ import annotations

import pytest

from agent.rag.vector_store import VectorStore


@pytest.fixture()
def store(tmp_path, fake_ef):
    return VectorStore(persist_dir=str(tmp_path), embedding_function=fake_ef)


def test_upsert_and_get_document(store):
    store.upsert_document(
        doc_name="亲密关系.pdf",
        digest="# 目录\n第一章 沟通",
        markdown="# 第一章 沟通\n\n沟通很重要。",
        chapters='[{"level":1,"title":"第一章 沟通","summary":"沟通很重要"}]',
    )
    doc = store.get_document_digest("亲密关系.pdf")
    assert doc is not None
    assert doc["digest"].startswith("# 目录")
    assert "第一章 沟通" in doc["markdown"]
    assert doc["chapters"]


def test_get_document_partial_match(store):
    store.upsert_document(
        doc_name="人生的活法-本多静六.pdf",
        digest="# 目录",
        markdown="# 第一章",
        chapters="[]",
    )
    doc = store.get_document_digest("本多静六")
    assert doc is not None
    assert doc["doc_name"] == "人生的活法-本多静六.pdf"


def test_upsert_overwrites_previous(store):
    store.upsert_document("a.pdf", digest="v1", markdown="m1", chapters="[]")
    store.upsert_document("a.pdf", digest="v2", markdown="m2", chapters="[]")
    doc = store.get_document_digest("a.pdf")
    assert doc["digest"] == "v2"


def test_list_documents(store):
    store.upsert_document("a.pdf", digest="d", markdown="m", chapters="[]")
    store.upsert_document("b.pdf", digest="d", markdown="m", chapters="[]")
    assert sorted(store.list_documents()) == ["a.pdf", "b.pdf"]


def test_get_missing_document(store):
    assert store.get_document_digest("不存在.pdf") is None


def test_delete_document(store):
    store.upsert_document("a.pdf", digest="d", markdown="m", chapters="[]")
    store.delete_document("a.pdf")
    assert store.get_document_digest("a.pdf") is None


def test_documents_table_survives_reopen(tmp_path, fake_ef):
    s1 = VectorStore(persist_dir=str(tmp_path), embedding_function=fake_ef)
    s1.upsert_document("a.pdf", digest="d", markdown="m", chapters="[]")
    s1.close()

    s2 = VectorStore(persist_dir=str(tmp_path), embedding_function=fake_ef)
    assert s2.get_document_digest("a.pdf") is not None
    s2.close()
