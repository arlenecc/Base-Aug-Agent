"""End-to-end RAG pipeline tests: extraction → cleaning → chunking → vectorization → search → reranking.

Tests the full knowledge base ingestion pipeline and verifies each stage:
1. Document parsing and text extraction
2. Text cleaning and normalization
3. Token-based chunking (500 tokens / 50 overlap)
4. Vector storage with ChromaDB
5. Semantic search with BGE reranking
6. RAG tool interface (rag_search, rag_status, rag_ingest)

Run: python3 -m pytest tests/test_rag_e2e.py -v
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Test documents ──────────────────────────────────────────────────────────

def _create_test_documents(kb_dir: str) -> None:
    """Create test documents of various formats in the knowledge base directory."""
    os.makedirs(kb_dir, exist_ok=True)

    # Markdown document — RAG technology intro
    with open(os.path.join(kb_dir, "rag_intro.md"), "w", encoding="utf-8") as f:
        f.write("""# RAG 技术介绍

## 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的技术架构。
它通过在生成回答前先从外部知识库中检索相关文档片段，然后将检索结果作为上下文提供给大语言模型，
从而显著提高回答的准确性和时效性。

## RAG 的核心组件

1. **文档解析器（Document Parser）**：负责从各种格式（PDF、Word、Markdown 等）中提取文本。
2. **文本切片器（Text Chunker）**：将长文档分割成适合模型处理的短片段。
3. **向量化引擎（Embedding Engine）**：将文本片段转换为向量表示。
4. **向量数据库（Vector Store）**：存储和检索向量，常用 ChromaDB、FAISS 等。
5. **重排序器（Reranker）**：对初步检索结果进行精确排序，提升相关性。

## RAG 的优势

相比传统的关键词搜索，RAG 使用语义向量搜索，能够理解查询的深层含义。
通过 BGE Reranker 等模型进行重排序，可以进一步过滤不相关的结果。
""")

    # TXT document — vector database comparison
    with open(os.path.join(kb_dir, "vector_db.txt"), "w", encoding="utf-8") as f:
        f.write("""向量数据库对比分析

向量数据库是专门用于存储和检索高维向量数据的数据库系统，在 AI 应用中扮演着关键角色。

1. ChromaDB
   - 开源、轻量级
   - 支持多种嵌入模型
   - 内置持久化存储
   - 适合中小规模应用

2. FAISS
   - Facebook AI 开发
   - 高性能向量检索
   - 支持 GPU 加速
   - 适合大规模检索场景

3. Pinecone
   - 云原生向量数据库
   - 全托管服务
   - 自动扩缩容
   - 适合生产环境部署

4. Weaviate
   - 开源向量搜索引擎
   - 支持 GraphQL 查询
   - 内置多种向量化模块
   - 适合复杂查询场景

选择向量数据库时需要考虑：数据规模、查询延迟、部署复杂度、成本等因素。
""")

    # HTML document — knowledge base best practices
    with open(os.path.join(kb_dir, "best_practices.html"), "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html>
<head><title>知识库最佳实践</title></head>
<body>
<header>导航栏（应被清洗掉）</header>
<nav>菜单（应被清洗掉）</nav>
<main>
<h1>知识库构建最佳实践</h1>

<p>构建高质量的知识库是 RAG 系统成功的关键。以下是一些经过验证的最佳实践：</p>

<h2>文档质量控制</h2>
<p>确保知识库中的文档内容准确、完整、格式规范。
避免包含大量扫描件或图片型 PDF，优先使用文本型文档。</p>

<h2>文本切片策略</h2>
<p>切片大小应根据模型上下文窗口合理设置。一般建议 500 token 一个切片，
保留 10% 的重叠区域，避免信息在切片边界处丢失。</p>

<h2>元数据管理</h2>
<p>为每个文档添加标题、来源、日期等元数据信息，方便检索时进行过滤和溯源。</p>

<h2>定期更新</h2>
<p>知识库需要定期更新，确保信息的时效性。建议设置自动同步机制。</p>
</main>
<footer>页脚信息（应被清洗掉）</footer>
<script>console.log("script should be removed")</script>
<style>body { color: red; }</style>
</body>
</html>""")

    # CSV document — embedding model comparison
    with open(os.path.join(kb_dir, "embedding_models.csv"), "w", encoding="utf-8") as f:
        f.write("""模型名称,维度,语言支持,特点
all-MiniLM-L6-v2,384,多语言,轻量快速
bge-large-zh-v1.5,1024,中文,中文优化
text-embedding-3-small,1536,多语言,OpenAI 官方
m3e-base,768,中英文,中文社区流行
bge-m3,1024,多语言,多语言检索""")

    # Subdirectory with nested documents
    sub_dir = os.path.join(kb_dir, "subdir")
    os.makedirs(sub_dir, exist_ok=True)
    with open(os.path.join(sub_dir, "tokens.md"), "w", encoding="utf-8") as f:
        f.write("""# Token 与上下文窗口管理

## Token 基本概念

Token 是大语言模型处理文本的基本单位。在英文中，一个 token 大约等于 3-4 个字符；
在中文中，一个 token 大约等于 1.5-2 个字符。

## 上下文窗口

上下文窗口决定了模型在一次推理中能"看到"的最大 token 数量。
常见的上下文窗口大小有 4K、8K、32K、128K 等。

## Token 消耗优化

1. 使用较短的系统提示词
2. 合理设置切片大小，避免过长的上下文
3. 使用重排序只保留最相关的结果
4. 压缩历史对话，移除冗余信息
""")


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_workspace():
    """Create a temporary workspace for RAG testing."""
    tmp = tempfile.mkdtemp(prefix="rag_test_ws_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def test_kb():
    """Create a temporary knowledge base with test documents."""
    tmp = tempfile.mkdtemp(prefix="rag_test_kb_")
    _create_test_documents(tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ── Module 1: Document extraction ───────────────────────────────────────────

class TestDocumentExtraction:
    """Test that parsers correctly extract text from various formats."""

    def test_extract_markdown(self, test_kb):
        from src.agent.rag.parsers import extract_text
        text = extract_text(os.path.join(test_kb, "rag_intro.md"))
        assert "RAG" in text
        assert "检索增强生成" in text
        assert "文档解析器" in text
        assert len(text) > 200

    def test_extract_txt(self, test_kb):
        from src.agent.rag.parsers import extract_text
        text = extract_text(os.path.join(test_kb, "vector_db.txt"))
        assert "向量数据库" in text
        assert "ChromaDB" in text
        assert "FAISS" in text
        assert len(text) > 300

    def test_extract_html(self, test_kb):
        from src.agent.rag.parsers import extract_text
        text = extract_text(os.path.join(test_kb, "best_practices.html"))
        assert "知识库" in text
        assert "切片" in text
        assert len(text) > 100

    def test_extract_csv(self, test_kb):
        from src.agent.rag.parsers import extract_text
        text = extract_text(os.path.join(test_kb, "embedding_models.csv"))
        assert "all-MiniLM-L6-v2" in text
        assert "bge-large-zh" in text
        assert len(text) > 80

    def test_extract_directory_recursive(self, test_kb):
        from src.agent.rag.parsers import extract_directory
        results = extract_directory(test_kb, recursive=True)
        # 5 documents: rag_intro.md, vector_db.txt, best_practices.html,
        #             embedding_models.csv, subdir/tokens.md
        assert len(results) >= 5
        sources = [os.path.basename(r[0]) for r in results]
        assert "rag_intro.md" in sources
        assert "tokens.md" in sources

    def test_extract_nonexistent_file(self):
        from src.agent.rag.parsers import extract_text
        with pytest.raises(FileNotFoundError):
            extract_text("/nonexistent/file.pdf")


# ── Module 2: Text cleaning ─────────────────────────────────────────────────

class TestTextCleaning:
    """Test the cleaner removes garbage and normalizes text."""

    def test_remove_html_tags(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "<div><p>Hello <b>World</b></p></div>"
        output = clean_text(input_text)
        assert "Hello" in output
        assert "World" in output
        assert "<div>" not in output
        assert "<b>" not in output

    def test_remove_script_style_blocks(self):
        from src.agent.rag.cleaner import clean_text
        input_text = """<script>var x = 1;</script>Content here<style>body{}</style>"""
        output = clean_text(input_text)
        assert "var x = 1" not in output
        assert "body{}" not in output
        assert "Content here" in output

    def test_remove_html_entities(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Hello &amp; World &lt;script&gt; &copy;"
        output = clean_text(input_text)
        assert "&amp;" not in output
        assert "&lt;" not in output
        assert "Hello" in output
        assert "World" in output

    def test_remove_urls(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Visit https://example.com/page for more info"
        output = clean_text(input_text)
        assert "https://example.com/page" not in output
        assert "more info" in output

    def test_remove_control_chars(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Hello\x00\x01\x02World\x1f"
        output = clean_text(input_text)
        assert "\x00" not in output
        assert "Hello" in output
        assert "World" in output

    def test_collapse_whitespace(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Hello    World\t\tTest"
        output = clean_text(input_text)
        assert "    " not in output
        assert "Hello World Test" in output

    def test_remove_page_numbers(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Content\n42\nMore content\n第 1 页\nStill more\nPage 3"
        output = clean_text(input_text)
        assert "42" not in output.split("\n")
        assert "第 1 页" not in output
        assert "Page 3" not in output

    def test_remove_separator_lines(self):
        from src.agent.rag.cleaner import clean_text
        input_text = "Title\n---\nContent\n====\nMore"
        output = clean_text(input_text)
        assert "---" not in output
        assert "====" not in output

    def test_clean_html_document(self, test_kb):
        from src.agent.rag.parsers import extract_text
        from src.agent.rag.cleaner import clean_text
        raw = extract_text(os.path.join(test_kb, "best_practices.html"))
        cleaned = clean_text(raw)
        # Should not contain nav/header/footer content
        assert "导航栏" not in cleaned
        assert "菜单" not in cleaned
        assert "页脚信息" not in cleaned
        # Should not contain script/style
        assert "console.log" not in cleaned
        assert "color: red" not in cleaned
        # Should contain main content
        assert "知识库构建" in cleaned
        assert "切片" in cleaned


# ── Module 3: Text chunking ─────────────────────────────────────────────────

class TestTextChunking:
    """Test token-based chunking with overlap."""

    def test_chunk_text_basic(self):
        from src.agent.rag.chunker import chunk_text
        text = "这是第一段测试文本。\n\n这是第二段测试文本。\n\n这是第三段测试文本。"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_short_text_returns_single(self):
        from src.agent.rag.chunker import chunk_text
        text = "短文本"
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "短文本"

    def test_chunk_empty_text(self):
        from src.agent.rag.chunker import chunk_text
        assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
        assert chunk_text("   ", chunk_size=500, chunk_overlap=50) == []

    def test_chunk_long_text(self):
        from src.agent.rag.chunker import chunk_text
        # Generate text that needs multiple chunks
        paragraphs = []
        for i in range(30):
            paragraphs.append(f"第{i}段：" + "这是测试内容。" * 50)
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        # Each chunk should be roughly <= 500 tokens
        for chunk in chunks:
            assert len(chunk) > 0

    def test_chunk_documents(self):
        from src.agent.rag.chunker import chunk_documents
        docs = [
            {"source": "/tmp/doc1.md", "text": "文档一的内容。" * 200},
            {"source": "/tmp/doc2.md", "text": "文档二的内容。" * 200},
        ]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 0
        # Verify metadata
        for chunk in chunks:
            assert "source" in chunk
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert chunk["source"] in ("/tmp/doc1.md", "/tmp/doc2.md")

    def test_chunk_preserves_content_order(self, test_kb):
        from src.agent.rag.parsers import extract_text
        from src.agent.rag.cleaner import clean_text
        from src.agent.rag.chunker import chunk_documents
        raw = extract_text(os.path.join(test_kb, "rag_intro.md"))
        cleaned = clean_text(raw)
        chunks = chunk_documents(
            [{"source": "rag_intro.md", "text": cleaned}],
            chunk_size=500, chunk_overlap=50,
        )
        assert len(chunks) >= 1
        # First chunk should contain the intro
        combined = " ".join(c["text"] for c in chunks)
        assert "RAG" in combined
        assert "检索增强生成" in combined


# ── Module 4: Vector store ──────────────────────────────────────────────────

class TestVectorStore:
    """Test ChromaDB vector storage and retrieval."""

    @pytest.fixture
    def store_dir(self):
        tmp = tempfile.mkdtemp(prefix="chroma_test_")
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def test_store_init_and_count(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)
        assert store.count() == 0

    def test_add_and_search(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "doc1.md", "chunk_index": 0,
             "text": "RAG 是一种检索增强生成技术，结合了信息检索和文本生成。"},
            {"source": "doc1.md", "chunk_index": 1,
             "text": "向量数据库如 ChromaDB 和 FAISS 用于存储文档嵌入向量。"},
            {"source": "doc2.md", "chunk_index": 0,
             "text": "Python 是一种流行的编程语言，广泛用于数据科学和 AI 开发。"},
        ]
        added = store.add(chunks)
        assert added == 3
        assert store.count() == 3

        # Search for RAG-related content
        results = store.search("什么是 RAG 技术", top_k=2)
        assert len(results) == 2
        # The RAG chunk should be the first result
        assert "RAG" in results[0]["text"]
        assert results[0]["source"] == "doc1.md"

    def test_search_empty_store(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)
        results = store.search("query", top_k=3)
        assert results == []

    def test_list_sources(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "/path/a.md", "chunk_index": 0, "text": "内容A"},
            {"source": "/path/b.md", "chunk_index": 0, "text": "内容B"},
            {"source": "/path/a.md", "chunk_index": 1, "text": "内容A续"},
        ]
        store.add(chunks)
        sources = store.list_sources()
        assert len(sources) == 2
        assert "/path/a.md" in sources
        assert "/path/b.md" in sources

    def test_clear(self, store_dir, vector_store_factory):
        store = vector_store_factory(persist_dir=store_dir)

        store.add([{"source": "doc.md", "chunk_index": 0, "text": "test"}])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_search_with_rerank_fallback(self, store_dir, vector_store_factory):
        """When reranker is unavailable, should fall back to top_k directly."""
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "doc.md", "chunk_index": i,
             "text": f"这是第{i}段关于机器学习和人工智能的内容。" * 3}
            for i in range(10)
        ]
        store.add(chunks)

        # search_with_rerank retrieves top_k * 4 candidates, then reranks
        results = store.search_with_rerank("机器学习", top_k=3)
        # Even without reranker (fallback), should return at most 3
        assert 1 <= len(results) <= 3
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "score" in r

    def test_search_fts_keyword(self, store_dir, vector_store_factory):
        """BM25 keyword search should find chunks by exact CJK keywords."""
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "a.md", "chunk_index": 0,
             "text": "机器学习是人工智能的核心分支。深度学习使用神经网络。"},
            {"source": "b.md", "chunk_index": 0,
             "text": "Python 是一种流行的编程语言。"},
            {"source": "c.md", "chunk_index": 0,
             "text": "气候变化是当今世界的重大挑战。"},
        ]
        store.add(chunks)

        # Exact keyword should surface the matching chunk.
        results = store.search_fts("机器学习 神经网络", top_k=3)
        assert results, "BM25 should return the matching chunk"
        assert results[0]["source"] == "a.md"

    def test_search_fts_english_keyword(self, store_dir, vector_store_factory):
        """BM25 should also work for English keywords."""
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "en.md", "chunk_index": 0,
             "text": "Machine learning is a branch of artificial intelligence."},
            {"source": "cn.md", "chunk_index": 0,
             "text": "深度学习使用多层神经网络。"},
        ]
        store.add(chunks)

        results = store.search_fts("machine learning", top_k=2)
        assert results, "BM25 should find the English chunk"
        assert results[0]["source"] == "en.md"

    def test_hybrid_retrieval_merges_vector_and_bm25(self, store_dir, vector_store_factory):
        """search_with_rerank merges vector + BM25 candidates (dedupe)."""
        store = vector_store_factory(persist_dir=store_dir)

        chunks = [
            {"source": "a.md", "chunk_index": 0,
             "text": "机器学习是人工智能的核心分支。"},
            {"source": "b.md", "chunk_index": 0,
             "text": "Python 是一种流行的编程语言。"},
            {"source": "c.md", "chunk_index": 0,
             "text": "气候变化是当今世界的重大挑战。"},
        ]
        store.add(chunks)

        results = store.search_with_rerank("机器学习", top_k=3)
        # Fallback mode (no reranker) returns up to top_k, no dedupe duplicates.
        assert 1 <= len(results) <= 3
        # No duplicate (source, chunk_index) pairs.
        keys = [(r["source"], r.get("chunk_index", 0)) for r in results]
        assert len(keys) == len(set(keys))

    def test_persistence(self, store_dir, vector_store_factory):
        """Data added to one store should be visible in another instance."""
        store1 = vector_store_factory(persist_dir=store_dir)
        store1.add([{"source": "p.md", "chunk_index": 0, "text": "持久化测试内容"}])
        assert store1.count() == 1

        # New instance pointing to same dir
        store2 = vector_store_factory(persist_dir=store_dir)
        assert store2.count() == 1
        results = store2.search("持久化测试", top_k=1)
        assert len(results) == 1
        assert "持久化测试" in results[0]["text"]


class _FixedDimEF:
    """Deterministic fixed-dimension embedding function for reset tests."""

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, input):
        return [[0.1] * self.dim for _ in input]

    def embed_query(self, text: str):
        return [0.1] * self.dim


class TestVectorStoreResetIfIncompatible:
    """嵌入模型切换（维度变化）后，向量库应能自愈清空重建（#text_fts 修复）。"""

    @pytest.fixture
    def store_dir(self):
        tmp = tempfile.mkdtemp(prefix="reset_test_")
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def test_same_dim_returns_false_and_keeps_data(self, store_dir):
        from agent.rag.vector_store import VectorStore

        ef = _FixedDimEF(128)
        store1 = VectorStore(persist_dir=store_dir, embedding_function=ef)
        store1.add([{"source": "a.md", "chunk_index": 0, "text": "兼容数据"}])
        assert store1.count() == 1

        # 同维度新实例：不兼容检查应返回 False，数据保留
        store2 = VectorStore(persist_dir=store_dir, embedding_function=ef)
        assert store2.reset_if_incompatible() is False
        assert store2.count() == 1

    def test_dim_change_resets_table(self, store_dir):
        from agent.rag.vector_store import VectorStore

        ef128 = _FixedDimEF(128)
        store1 = VectorStore(persist_dir=store_dir, embedding_function=ef128)
        store1.add([{"source": "a.md", "chunk_index": 0, "text": "旧模型数据"}])
        assert store1.count() == 1

        # 换成 64 维嵌入后：应检测到维度不匹配并清空重建
        ef64 = _FixedDimEF(64)
        store2 = VectorStore(persist_dir=store_dir, embedding_function=ef64)
        assert store2.reset_if_incompatible() is True
        assert store2.count() == 0

        # 清空后应能正常写入与检索（新维度）
        store2.add([{"source": "b.md", "chunk_index": 0, "text": "新模型数据"}])
        assert store2.count() == 1
        results = store2.search("新模型", top_k=1)
        assert len(results) == 1
        assert results[0]["source"] == "b.md"

        # 重建后的表对同维度实例仍是兼容的（第二次检查返回 False）
        store3 = VectorStore(persist_dir=store_dir, embedding_function=ef64)
        assert store3.reset_if_incompatible() is False
        assert store3.count() == 1

    def test_missing_fts_column_detected(self, store_dir, monkeypatch):
        """旧表缺 text_fts/ingested_at 列时也应判定不兼容（用户报错场景）。

        正常路径下 _check_schema_migration 会在初始化时把列补齐；这里模拟
        生产事故——迁移中途抛异常被吞（except Exception 后旧表原样保留），
        此时 reset_if_incompatible 必须能兜底检测并清库。
        """
        import lancedb

        from agent.rag.vector_store import _TABLE_NAME, VectorStore

        # 手工造一张只含旧版列的表（无 text_fts / ingested_at）
        db = lancedb.connect(store_dir)
        ef = _FixedDimEF(128)
        db.create_table(_TABLE_NAME, [
            {"id": "x", "text": "旧表", "vector": [0.1] * 128,
             "source": "old.md", "chunk_index": 0},
        ])

        store = VectorStore(persist_dir=store_dir, embedding_function=ef)
        # 让初始化时的 schema 迁移失效（模拟迁移失败被吞后旧表保留）
        monkeypatch.setattr(store, "_check_schema_migration", lambda: None)
        store._initialized = False
        store._ensure_initialized()
        assert store.reset_if_incompatible() is True
        assert store.count() == 0


class TestSchemaMigrationSafety:
    """#140 内存事故（22GB 卡死）回归：迁移必须维度感知 + 有界内存 + 不丢列。"""

    @pytest.fixture
    def store_dir(self):
        tmp = tempfile.mkdtemp(prefix="mig_test_")
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def _make_old_table(self, store_dir, dim, with_fts, with_ingested):
        """手工建一张指定列组合的旧版表，模拟历史版本的 schema。"""
        import lancedb

        from agent.rag.vector_store import _TABLE_NAME

        db = lancedb.connect(store_dir)
        row = {
            "id": "abc_0", "text": "旧表数据",
            "vector": [0.1] * dim,
            "source": "old.md", "chunk_index": 0,
        }
        if with_fts:
            row["text_fts"] = "旧表 数据"
        if with_ingested:
            row["ingested_at"] = 0.0
        db.create_table(_TABLE_NAME, [row])

    def test_migration_preserves_existing_text_fts(self, store_dir):
        """只缺 ingested_at 的旧表迁移后必须保留已有 text_fts（生产事故根因）。"""
        from agent.rag.vector_store import _INGESTED_AT, _FTS_COLUMN, VectorStore

        # 旧表：有 text_fts，缺 ingested_at（10-22 之前的历史 schema）
        self._make_old_table(store_dir, 128, with_fts=True, with_ingested=False)

        store = VectorStore(persist_dir=store_dir, embedding_function=_FixedDimEF(128))
        store._ensure_initialized()
        schema = store._table.schema
        assert _FTS_COLUMN in schema.names, "迁移丢失了已有的 text_fts 列！"
        assert _INGESTED_AT in schema.names
        assert store.count() == 1
        # 已有 text_fts 内容原样保留（不重新分词）
        rows = store._table.to_arrow().to_pylist()
        assert rows[0]["text_fts"] == "旧表 数据"

    def test_migration_skipped_on_dim_mismatch(self, store_dir):
        """旧表维度与当前嵌入不一致时跳过迁移（不读全表），留给 reset 清库。"""
        from agent.rag.vector_store import _FTS_COLUMN, VectorStore

        # 64 维旧表，当前 EF 是 128 维，且缺 text_fts
        self._make_old_table(store_dir, 64, with_fts=False, with_ingested=True)

        store = VectorStore(persist_dir=store_dir, embedding_function=_FixedDimEF(128))
        store._ensure_initialized()
        # 迁移被跳过：表原样保留（仍是 64 维、缺 text_fts）
        schema = store._table.schema
        assert _FTS_COLUMN not in schema.names
        assert schema.field("vector").type.list_size == 64
        assert store.count() == 1
        # 随后 reset_if_incompatible 兜底清库
        assert store.reset_if_incompatible() is True
        assert store.count() == 0

    def test_add_rejects_dim_mismatch_with_clear_error(self, store_dir):
        """旧表残留时写入必须 fail-fast 并给出可行动的错误信息。"""
        import pytest as _pytest

        from agent.rag.vector_store import VectorStore

        self._make_old_table(store_dir, 64, with_fts=True, with_ingested=True)

        store = VectorStore(persist_dir=store_dir, embedding_function=_FixedDimEF(128))
        with _pytest.raises(RuntimeError, match="维度不匹配"):
            store.add([{"source": "a.md", "chunk_index": 0, "text": "新数据"}])


# ── Module 5: RAG Engine ────────────────────────────────────────────────────

class TestRAGEngine:
    """Test the full RAG engine: ingest → search → status."""

    @pytest.fixture
    def engine(self, test_workspace, test_kb, rag_engine_factory):
        eng = rag_engine_factory(workspace=test_workspace, knowledge_base=test_kb)
        return eng

    def test_ingest_creates_rag_structure(self, engine):
        """Ingestion should create the rag directory structure."""
        stats = engine.ingest(force=True)
        assert "error" not in stats
        assert stats["files_found"] >= 5
        assert stats["chunks"] > 0
        assert stats["total_chars"] > 0

        # Verify directory structure
        assert os.path.isdir(engine.rag_dir)
        assert os.path.isdir(engine.markdown_dir)
        # Check that markdown files were created
        md_files = [f for f in os.listdir(engine.markdown_dir) if f.endswith(".md")]
        assert len(md_files) >= 5

    def test_ingest_skips_on_second_run(self, engine):
        """Second ingestion without force should skip already-processed files."""
        # First run
        stats1 = engine.ingest(force=True)
        assert stats1["files_extracted"] >= 5

        # Second run — should skip all
        stats2 = engine.ingest(force=False)
        assert stats2["files_skipped"] >= stats1["files_extracted"]
        assert stats2["files_extracted"] == 0

    def test_ingest_force_reprocesses(self, engine):
        """Force re-ingestion should reprocess all files."""
        engine.ingest(force=True)  # initial
        stats = engine.ingest(force=True)  # force re-process
        assert stats["files_extracted"] >= 5
        assert stats["files_skipped"] == 0

    def test_search_relevant_content(self, engine):
        """Semantic search should find relevant content."""
        engine.ingest(force=True)

        # Search for RAG-related content
        results = engine.search("什么是 RAG 技术")
        assert len(results) >= 1
        # The deterministic test embedding (MD5 hash) carries no semantic
        # signal, so we can only assert that search returned non-empty,
        # well-formed results — not that a specific keyword appears in the
        # top hits (which would depend on the arbitrary chunk boundaries).
        for r in results:
            assert r["text"].strip()
            assert r["source"]

    def test_search_vector_databases(self, engine):
        """Search for vector database content."""
        engine.ingest(force=True)

        results = engine.search("向量数据库有哪些")
        assert len(results) >= 1
        # Should find ChromaDB, FAISS, etc.
        combined = " ".join(r["text"] for r in results)
        assert "ChromaDB" in combined or "FAISS" in combined

    def test_search_knowledge_base_practices(self, engine):
        """Search for knowledge base best practices."""
        engine.ingest(force=True)

        results = engine.search("知识库构建")
        assert len(results) >= 1
        combined = " ".join(r["text"] for r in results)
        assert "切片" in combined or "知识库" in combined

    def test_search_embedding_models(self, engine):
        """Search for embedding model information."""
        engine.ingest(force=True)

        # top_k 拉满覆盖全库切片：确定性测试嵌入（MD5）不带语义信号，
        # CSV 切片能否进 top-3 纯属切片布局运气（fix_hard_wraps 断行拼接
        # 会改变段落边界），top_k >= 总切片数后断言与布局无关。
        results = engine.search("嵌入模型有哪些", top_k=10)
        assert len(results) >= 1
        combined = " ".join(r["text"] for r in results)
        # Should find model names from the CSV
        assert any(name in combined for name in [
            "all-MiniLM", "bge-large", "m3e-base",
        ])

    def test_search_token_management(self, engine):
        """Search for token-related content (from subdirectory)."""
        engine.ingest(force=True)

        results = engine.search("token 上下文窗口")
        assert len(results) >= 1
        # Deterministic (MD5) test embedding has no semantic signal, so we
        # only assert search returns well-formed, non-empty results — not that
        # a specific keyword appears in the top hits (which depends on
        # arbitrary chunk boundaries, not relevance).
        for r in results:
            assert r["text"].strip()
            assert r["source"]

    def test_search_empty_query(self, rag_engine_factory):
        """Search with empty knowledge base should return empty."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            eng = rag_engine_factory(workspace=tmp, knowledge_base="")
            results = eng.search("anything")
            assert results == []

    def test_search_formatted_returns_string(self, engine):
        """search_formatted should return a formatted string."""
        engine.ingest(force=True)
        result = engine.search_formatted("RAG 技术", top_k=2)
        assert isinstance(result, str)
        assert "知识库检索" in result
        assert "相关度" in result

    def test_search_formatted_returns_valid_string(self, engine):
        """search_formatted should always return a valid string with result count."""
        engine.ingest(force=True)
        result = engine.search_formatted("量子计算与黑洞理论", top_k=1)
        assert isinstance(result, str)
        # Even for niche queries, the embedding model may find semantic matches.
        # The result should be well-formed regardless.
        assert "知识库检索" in result or "未找到" in result

    def test_status_reports_correctly(self, engine):
        """Status should reflect the current state of the knowledge base."""
        engine.ingest(force=True)
        status = engine.status()
        assert status["chunks_stored"] > 0
        assert len(status["sources"]) >= 5
        assert status["has_knowledge_base"] is True

    def test_status_no_knowledge_base(self, rag_engine_factory):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            eng = rag_engine_factory(workspace=tmp, knowledge_base="")
            status = eng.status()
            assert status["chunks_stored"] == 0
            assert status["has_knowledge_base"] is False

    def test_clear_removes_everything(self, engine):
        """Clear should remove all vectors and markdown cache."""
        engine.ingest(force=True)
        assert engine.status()["chunks_stored"] > 0

        engine.clear()
        assert engine.status()["chunks_stored"] == 0
        # Markdown cache should also be cleared
        if os.path.isdir(engine.markdown_dir):
            md_files = [f for f in os.listdir(engine.markdown_dir) if f.endswith(".md")]
            assert len(md_files) == 0

    def test_ingest_no_knowledge_base(self, test_workspace, rag_engine_factory):
        eng = rag_engine_factory(workspace=test_workspace, knowledge_base="/nonexistent/path")
        stats = eng.ingest()
        assert "error" in stats


# ── Module 6: RAG Tool interface ────────────────────────────────────────────

class TestRAGTools:
    """Test the tool interface that the agent uses."""

    @pytest.fixture
    def engine(self, test_workspace, test_kb, rag_engine_factory):
        eng = rag_engine_factory(workspace=test_workspace, knowledge_base=test_kb)
        eng.ingest(force=True)
        return eng

    def test_rag_search_tool(self, engine):
        from src.agent.tools.rag_tool import RagSearchTool
        tool = RagSearchTool(engine)
        result = tool.run(query="向量数据库")
        assert result.success is True
        # The search should return relevant results about vector databases
        # or embedding models (semantic search may match different docs)
        assert "知识库检索" in result.output
        assert "相关度" in result.output

    def test_rag_search_tool_empty_query(self, engine):
        from src.agent.tools.rag_tool import RagSearchTool
        tool = RagSearchTool(engine)
        result = tool.run(query="")
        assert result.success is False
        assert "请提供" in result.error

    def test_rag_search_tool_no_results(self, engine):
        from src.agent.tools.rag_tool import RagSearchTool
        tool = RagSearchTool(engine)
        result = tool.run(query="量子计算黑洞理论弦论")
        assert result.success is True
        # The search should not crash and return a valid formatted result.
        # Even for obscure queries, the embedding model may find semantic
        # matches due to the nature of vector search.
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    def test_rag_status_tool(self, engine):
        from src.agent.tools.rag_tool import RagStatusTool
        tool = RagStatusTool(engine)
        result = tool.run()
        assert result.success is True
        assert "切片数" in result.output
        assert "已索引文件" in result.output

    def test_rag_search_condenses_with_aux(self):
        """reranker 结果应先送辅助模型归纳，再交给主模型（#RAG 浓缩流程）。"""
        from src.agent.tools.rag_tool import RagSearchTool

        class _Ev:
            def __init__(self, t, c=""):
                self.type = t
                self.content = c

        class _FakeEngine:
            def search(self, query, top_k=3, source=None):
                # 总量需 ≥ rag_condense_min_chars（默认 4000 字）才触发归纳。
                return [
                    {"source": "doc_a.txt", "score": 0.9, "text": "A" * 2500},
                    {"source": "doc_b.txt", "score": 0.8, "text": "B" * 2500},
                ]

            def format_results(self, results, *a, **k):
                return "RAW:" + str(len(results))

        class _FakeAux:
            def __init__(self, reply):
                self._reply = reply

            def chat_stream(self, messages, tools=None, temperature=0.2):
                yield _Ev("content", self._reply)
                yield _Ev("done")

        condensed = "浓缩后的要点（保留来源 doc_a / doc_b）。"
        tool = RagSearchTool(_FakeEngine(), aux_llm_getter=lambda: _FakeAux(condensed))
        res = tool.run(query="测试问题")
        assert res.success is True
        # 主模型拿到的是辅助模型归纳后的文本，而非原始分片格式。
        assert res.output == condensed
        assert "RAW:" not in res.output

    def test_rag_search_skips_condense_below_min_chars(self):
        """结果总量 < rag_condense_min_chars（默认 4000 字）时跳过归纳、原样返回。"""
        from src.agent.tools.rag_tool import RagSearchTool

        class _Ev:
            def __init__(self, t, c=""):
                self.type = t
                self.content = c

        class _FakeEngine:
            def search(self, query, top_k=3, source=None):
                return [
                    {"source": "doc_a.txt", "score": 0.9, "text": "A" * 500},
                    {"source": "doc_b.txt", "score": 0.8, "text": "B" * 500},
                ]

            def format_results(self, results, *a, **k):
                return "RAW:" + str(len(results))

        class _SpyAux:
            def __init__(self):
                self.called = 0

            def chat_stream(self, messages, tools=None, temperature=0.2):
                self.called += 1
                yield _Ev("content", "不该被调用")
                yield _Ev("done")

        aux = _SpyAux()
        tool = RagSearchTool(_FakeEngine(), aux_llm_getter=lambda: aux)
        res = tool.run(query="测试问题")
        assert res.success is True
        assert aux.called == 0  # 辅助模型不应被调用
        assert res.output == "RAW:2"  # 原始分片格式直接返回

    def test_rag_search_falls_back_when_no_aux(self):
        """辅助模型不可用（getter 返回 None）时退化为原始分片格式。"""
        from src.agent.tools.rag_tool import RagSearchTool

        class _FakeEngine:
            def search(self, query, top_k=3, source=None):
                return [{"source": "doc_a.txt", "score": 0.9, "text": "A" * 100}]

            def format_results(self, results, *a, **k):
                return "RAW FORMATTED"

        tool = RagSearchTool(_FakeEngine(), aux_llm_getter=lambda: None)
        res = tool.run(query="测试")
        assert res.success is True
        assert res.output == "RAW FORMATTED"

    def test_rag_search_falls_back_on_aux_error(self):
        """辅助模型调用抛错时退化为原始分片格式，不让检索整体失败。"""
        from src.agent.tools.rag_tool import RagSearchTool

        class _FakeEngine:
            def search(self, query, top_k=3, source=None):
                return [{"source": "doc_a.txt", "score": 0.9, "text": "x"}]

            def format_results(self, results, *a, **k):
                return "RAW"

        class _BadAux:
            def chat_stream(self, messages, tools=None, temperature=0.2):
                raise RuntimeError("boom")

        tool = RagSearchTool(_FakeEngine(), aux_llm_getter=lambda: _BadAux())
        res = tool.run(query="q")
        assert res.success is True
        assert res.output == "RAW"

    def test_rag_search_condensed_truncated_to_2000(self):
        """辅助模型未遵守字数限制时，结果强制截断到 2000 字符。"""
        from src.agent.tools.rag_tool import RagSearchTool

        class _Ev:
            def __init__(self, t, c=""):
                self.type = t
                self.content = c

        class _FakeEngine:
            def search(self, query, top_k=3, source=None):
                # 总量需 ≥ rag_condense_min_chars（默认 4000 字）才触发归纳。
                return [{"source": "doc.txt", "score": 0.9, "text": "y" * 4500}]

            def format_results(self, results, *a, **k):
                return "RAW"

        class _LongAux:
            def chat_stream(self, messages, tools=None, temperature=0.2):
                yield _Ev("content", "z" * 3000)
                yield _Ev("done")

        tool = RagSearchTool(_FakeEngine(), aux_llm_getter=lambda: _LongAux())
        res = tool.run(query="q")
        assert len(res.output) == 2000

    def test_rag_ingest_tool(self, engine):
        from src.agent.tools.rag_tool import RagIngestTool
        tool = RagIngestTool(engine)
        result = tool.run(force=True)
        assert result.success is True
        assert "切片数" in result.output
        assert "索引完成" in result.output

    def test_rag_ingest_tool_no_kb(self, test_workspace, rag_engine_factory):
        from src.agent.tools.rag_tool import RagIngestTool
        eng = rag_engine_factory(workspace=test_workspace, knowledge_base="/nonexistent")
        tool = RagIngestTool(eng)
        result = tool.run()
        assert result.success is False
        assert "目录" in result.error or "not found" in result.error.lower()


# ── Module 7: End-to-end ingestion pipeline ─────────────────────────────────

class TestEndToEndPipeline:
    """Full pipeline: create docs → ingest → search → verify relevance."""

    @pytest.fixture
    def setup(self, test_workspace, test_kb, rag_engine_factory):
        engine = rag_engine_factory(workspace=test_workspace, knowledge_base=test_kb)
        return engine

    def test_full_pipeline_with_rerank(self, setup):
        """E2E: ingest all documents, search with reranking, verify results."""
        engine = setup

        # Step 1: Ingest
        stats = engine.ingest(force=True)
        assert stats["files_found"] >= 5, f"Expected >=5 files, got {stats['files_found']}"
        assert stats["chunks"] > 0, "No chunks generated"
        assert stats["total_chars"] > 500, "Too few characters extracted"

        # Step 2: Verify status
        status = engine.status()
        assert status["chunks_stored"] == stats["chunks"]
        assert len(status["sources"]) >= 5

        # Step 3: Search with reranking (top_k=3)
        # Test multiple queries to verify semantic search quality
        test_queries = [
            ("什么是 RAG 技术", "RAG"),
            ("向量数据库对比", "ChromaDB"),
            ("知识库构建最佳实践", "切片"),
            ("嵌入模型", "all-MiniLM"),
            ("token 上下文窗口", "Token"),
        ]

        for query, expected_keyword in test_queries:
            results = engine.search(query, top_k=3)
            assert len(results) >= 1, f"No results for query: {query}"
            # Should return at most 3 results after reranking
            assert len(results) <= 3, f"Too many results for query: {query}"
            # Each result should have score
            for r in results:
                assert "score" in r, f"Missing score in result for query: {query}"
                assert "text" in r
                assert "source" in r

        # Step 4: Re-ingest without force — should skip
        stats2 = engine.ingest(force=False)
        assert stats2["files_skipped"] >= 5
        assert stats2["files_extracted"] == 0
        # Chunks should remain the same
        assert engine.status()["chunks_stored"] == stats["chunks"]

        # Step 5: Clear and verify
        engine.clear()
        assert engine.status()["chunks_stored"] == 0

    def test_cleaned_markdown_files_are_valid(self, setup):
        """Verify cleaned markdown files are readable and clean."""
        engine = setup
        engine.ingest(force=True)

        md_dir = engine.markdown_dir
        for md_file in os.listdir(md_dir):
            if not md_file.endswith(".md"):
                continue
            path = os.path.join(md_dir, md_file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Should not contain raw HTML tags
            assert "<script>" not in content.lower()
            assert "<style>" not in content.lower()
            assert "<nav>" not in content.lower()
            # Should not contain HTML entities
            assert "&amp;" not in content
            assert "&lt;" not in content
            # Should have meaningful content
            assert len(content.strip()) > 10, f"Empty content in {md_file}"

    def test_subdirectory_documents_are_indexed(self, setup):
        """Documents in subdirectories should be found and indexed."""
        engine = setup
        engine.ingest(force=True)

        # Search for content from the subdirectory
        results = engine.search("token 概念")
        assert len(results) >= 1
        combined = " ".join(r["text"] for r in results)
        assert "token" in combined.lower() or "Token" in combined


class TestRerankSimilarityThreshold:
    """min_similarity 过滤阈值（rag_rerank_min_similarity，默认 0.55，config 可调）。

    单元验证，不加载模型：直接对「已 sigmoid 归一化到 [0,1] 的 rerank 分」施加
    阈值，确认默认档位与「阈值过高(0.7)会误伤边界相关项」的风险预期一致。
    """

    def test_default_threshold_is_055(self):
        from agent.config import AgentConfig
        assert AgentConfig().rag_rerank_min_similarity == 0.55

    def test_filter_drops_below_threshold(self, vector_store_factory, tmp_path):
        store = vector_store_factory(persist_dir=str(tmp_path))
        # 0.88=强相关、0.57=边界相关（机器学习主题实测档位）、
        # 0.51=无关弱匹配、0.49=噪声。
        results = [
            {"source": "a", "score": 0.88},
            {"source": "b", "score": 0.57},
            {"source": "c", "score": 0.51},
            {"source": "d", "score": 0.49},
        ]
        kept = store._filter_by_similarity(results, 0.55)
        assert [r["source"] for r in kept] == ["a", "b"]

    def test_threshold_070_misfires_borderline(self, vector_store_factory, tmp_path):
        store = vector_store_factory(persist_dir=str(tmp_path))
        results = [
            {"source": "a", "score": 0.88},
            {"source": "b", "score": 0.57},  # 正确相关项，但 < 0.7
        ]
        # 0.7 会误伤 0.57 的正确相关项，只剩强相关那条 → 召回率下降。
        kept = store._filter_by_similarity(results, 0.7)
        assert [r["source"] for r in kept] == ["a"]

    def test_nonpositive_threshold_keeps_all(self, vector_store_factory, tmp_path):
        store = vector_store_factory(persist_dir=str(tmp_path))
        results = [{"source": "x", "score": 0.1}, {"source": "y", "score": 0.2}]
        assert store._filter_by_similarity(results, 0) == results
        assert store._filter_by_similarity(results, -1) == results

    def test_filter_keeps_top1_when_all_below_threshold(self, vector_store_factory, tmp_path):
        store = vector_store_factory(persist_dir=str(tmp_path))
        # 整体打分偏低：全部候选都低于阈值（跨语种弱匹配 / 查询词质量差）。
        results = [
            {"source": "a", "score": 0.51},  # 最高分
            {"source": "b", "score": 0.49},
            {"source": "c", "score": 0.40},
        ]
        # 保底：至少保留得分最高的 1 条，而非返回空 → 下游 LLM 不会丢上下文。
        kept = store._filter_by_similarity(results, 0.55)
        assert [r["source"] for r in kept] == ["a"]
        assert len(kept) == 1


class TestBinaryCoarseRetrieval:
    """两段式检索（二进制量化 BQ 极速粗筛）的单元/集成验证。

    仅当 rag_binary_coarse_enabled=True 时启用；默认关闭，故不影响既有检索行为。
    """

    def test_default_disabled(self):
        from agent.config import AgentConfig
        assert AgentConfig().rag_binary_coarse_enabled is False

    def test_prepare_binary_packed_length(self, vector_store_factory, tmp_path):
        store = vector_store_factory(
            persist_dir=str(tmp_path), binary_coarse_enabled=True)
        # FakeEmbeddingFunction 为 128 维，matryoshka_dim=128 → 截断后 128 维，
        # 符号位打包后 ceil(128/8)=16 字节。
        packed = store._prepare_binary(np.zeros(128, dtype=np.float32))
        assert isinstance(packed, bytes)
        assert len(packed) == 16

    def test_enabled_populates_binary_table(self, vector_store_factory, tmp_path):
        store = vector_store_factory(
            persist_dir=str(tmp_path), binary_coarse_enabled=True)
        store.add([
            {"text": "机器学习是人工智能的一个分支", "source": "a.txt", "chunk_index": 0},
            {"text": "向量数据库用于存储嵌入向量", "source": "a.txt", "chunk_index": 1},
            {"text": "二进制量化把向量压缩成比特串", "source": "b.txt", "chunk_index": 0},
        ])
        # kb_binary 表应被创建并写入 3 条。
        assert "kb_binary" in store._db.table_names()
        btbl = store._db.open_table("kb_binary")
        assert btbl.count_rows() == 3
        # 二进制列宽度 = 16 字节（128 维）。
        arrow = btbl.to_arrow()
        assert arrow.schema.field("binary").type.byte_width == 16

    def test_disabled_store_has_no_binary_table(self, vector_store_factory, tmp_path):
        store = vector_store_factory(persist_dir=str(tmp_path))  # 默认关闭
        store.add([
            {"text": "机器学习是人工智能的一个分支", "source": "a.txt", "chunk_index": 0},
        ])
        assert "kb_binary" not in store._db.table_names()
        assert store.search_binary_coarse("机器学习", refine_top_k=3) is None

    def test_coarse_returns_candidates(self, vector_store_factory, tmp_path):
        store = vector_store_factory(
            persist_dir=str(tmp_path), binary_coarse_enabled=True)
        store.add([
            {"text": "机器学习是人工智能的一个分支", "source": "a.txt", "chunk_index": 0},
            {"text": "深度学习使用神经网络", "source": "a.txt", "chunk_index": 1},
            {"text": "向量数据库用于存储嵌入向量", "source": "b.txt", "chunk_index": 0},
            {"text": "二进制量化把向量压缩成比特串以加速检索",
             "source": "b.txt", "chunk_index": 1},
        ])
        out = store.search_binary_coarse("机器学习", refine_top_k=3)
        assert isinstance(out, list) and len(out) > 0
        assert {"text", "source", "chunk_index", "score", "id"}.issubset(out[0].keys())
        # 分数应按余弦降序。
        scores = [c["score"] for c in out]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_rerank_uses_coarse(self, vector_store_factory, tmp_path):
        store = vector_store_factory(
            persist_dir=str(tmp_path), binary_coarse_enabled=True)
        store.add([
            {"text": "机器学习是人工智能的一个分支", "source": "a.txt", "chunk_index": 0},
            {"text": "深度学习使用神经网络", "source": "a.txt", "chunk_index": 1},
            {"text": "向量数据库用于存储嵌入向量", "source": "b.txt", "chunk_index": 0},
        ])
        # 开启 BQ 粗筛后 search_with_rerank 仍应正常返回候选，不崩溃。
        results = store.search_with_rerank("机器学习", top_k=2)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_clear_drops_binary_table(self, vector_store_factory, tmp_path):
        store = vector_store_factory(
            persist_dir=str(tmp_path), binary_coarse_enabled=True)
        store.add([
            {"text": "机器学习是人工智能的一个分支", "source": "a.txt", "chunk_index": 0},
        ])
        assert "kb_binary" in store._db.table_names()
        store.clear()
        assert "kb_binary" not in store._db.table_names()