"""Embedding 任务前缀策略（embedding_prefixes）单测。

验证：
  1. 模型名/本地路径 → 模型族识别（nomic / bge / jina / e5 / 未知）；
  2. 各模型族的 document/query 前缀取值（含 jina 的 clustering/classification）；
  3. 前缀应用幂等（已带前缀不重复加、空前缀原样返回）；
  4. 各嵌入后端构造期按模型路径自动解析前缀（不触发模型加载）。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.rag import embedding_prefixes as ep


# ---------------------------------------------------------------- 模型族识别

def test_detect_family_by_hf_name():
    assert ep.detect_family("nomic-ai/nomic-embed-text-v1.5-Q") == "nomic"
    assert ep.detect_family("BAAI/bge-small-zh-v1.5") == "bge"
    assert ep.detect_family("jinaai/jina-embeddings-v3") == "jina"
    assert ep.detect_family("intfloat/multilingual-e5-large") == "e5"


def test_detect_family_by_local_path():
    assert ep.detect_family("models/nomic-embed-text-v1.5-mlx-int8") == "nomic"
    assert ep.detect_family("models/bge-small-zh-v1.5-mlx") == "bge"


def test_detect_family_unknown():
    assert ep.detect_family("sentence-transformers/all-MiniLM-L6-v2") == ""
    assert ep.detect_family("") == ""


# ---------------------------------------------------------------- 前缀取值

def test_nomic_prefixes():
    model = "models/nomic-embed-text-v1.5-mlx-int8"
    doc, query = ep.prefixes(model)
    assert doc == "search_document: "
    assert query == "search_query: "


def test_bge_prefixes():
    model = "BAAI/bge-small-zh-v1.5"
    doc, query = ep.prefixes(model)
    assert doc == ""                      # bge 文档侧不加前缀
    assert query == "为这个句子生成表示以检索相关文档"


def test_jina_prefixes():
    model = "jinaai/jina-embeddings-v3"
    doc, query = ep.prefixes(model)
    assert doc == "passage: "
    assert query == "query: "
    # jina 族额外定义聚类 / 分类任务前缀
    assert ep.task_prefix(model, "clustering") == "clustering: "
    assert ep.task_prefix(model, "classification") == "classification: "


def test_e5_prefixes():
    model = "intfloat/multilingual-e5-large"
    doc, query = ep.prefixes(model)
    assert doc == "passage: "
    assert query == "query: "
    # e5 官方未定义聚类/分类前缀 → 空串
    assert ep.task_prefix(model, "clustering") == ""
    assert ep.task_prefix(model, "classification") == ""


def test_unknown_model_no_prefix():
    model = "sentence-transformers/all-MiniLM-L6-v2"
    assert ep.prefixes(model) == ("", "")
    assert ep.task_prefix(model, "document") == ""
    assert ep.task_prefix(model, "query") == ""


# ---------------------------------------------------------------- 幂等应用

def test_add_prefix_idempotent():
    model = "nomic-ai/nomic-embed-text-v1.5-Q"
    once = ep.add_prefix(model, "document", "你好世界")
    assert once == "search_document: 你好世界"
    assert ep.add_prefix(model, "document", once) == once  # 已带前缀不重复


def test_add_prefix_empty_prefix_noop():
    # bge 文档侧前缀为空 → 原样返回
    assert ep.add_prefix("BAAI/bge-small-zh-v1.5", "document", "原文") == "原文"


def test_apply_prefix_guard():
    assert ep.apply_prefix("query: ", "hello") == "query: hello"
    assert ep.apply_prefix("query: ", "query: hello") == "query: hello"
    assert ep.apply_prefix("", "hello") == "hello"


# ------------------------------------------------- 后端构造期解析（不加载模型）

def test_mlx_ef_resolves_prefix_by_path():
    from agent.rag.model_backends import MLXEmbeddingFunction

    nomic = MLXEmbeddingFunction("models/nomic-embed-text-v1.5-mlx-int8")
    assert nomic._document_prefix == "search_document: "
    assert nomic._query_prefix == "search_query: "

    bge = MLXEmbeddingFunction("models/bge-small-zh-v1.5-mlx")
    assert bge._document_prefix == ""
    assert bge._query_prefix == "为这个句子生成表示以检索相关文档"


def test_mlx_ef_explicit_prefix_overrides():
    from agent.rag.model_backends import MLXEmbeddingFunction

    ef = MLXEmbeddingFunction(
        "models/nomic-embed-text-v1.5-mlx-int8",
        document_prefix="DOC:", query_prefix="QRY:",
    )
    assert ef._document_prefix == "DOC:"
    assert ef._query_prefix == "QRY:"


def test_fastembed_ef_resolves_prefix_by_name():
    from agent.rag.vector_store import FastEmbedEmbeddingFunction

    nomic = FastEmbedEmbeddingFunction("nomic-ai/nomic-embed-text-v1.5-Q")
    assert nomic._doc_prefix == "search_document: "
    assert nomic._query_prefix == "search_query: "

    e5 = FastEmbedEmbeddingFunction("intfloat/multilingual-e5-large")
    assert e5._doc_prefix == "passage: "
    assert e5._query_prefix == "query: "
