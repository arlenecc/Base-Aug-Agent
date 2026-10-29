"""jina-embeddings-v3 支持测试：MLX/ONNX 后端 + VectorStore int8 管线。

覆盖：
  1. create_embedding_backend 按模型名/路径分流到 Jina 专用后端；
  2. 真实模型（models/jina-embeddings-v3-mlx-int4 / -onnx-int4）加载与
     检索排序（retrieval.passage / retrieval.query 任务类型对称）；
  3. VectorStore jina profile：截 256 维 → L2 重归一化 → int8 入库，
     暴力检索排序正确、where 过滤生效（伪 EF，确定性向量，不加载模型）。
"""

import os
import sys

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLX_DIR = os.path.join(ROOT, "models", "jina-embeddings-v3-mlx-int4")
ONNX_DIR = os.path.join(ROOT, "models", "jina-embeddings-v3-onnx-int4")

_HAS_MLX = os.path.isfile(os.path.join(MLX_DIR, "weights.safetensors"))
_HAS_ONNX = os.path.isfile(os.path.join(ONNX_DIR, "model.onnx"))

DOCS = [
    "Berlin is the capital and largest city of Germany.",
    "The Pacific Ocean is the largest ocean on Earth.",
]
QUERIES = [
    "What is the capital of Germany?",   # 应命中 DOCS[0]
    "Which is the largest ocean?",       # 应命中 DOCS[1]
]


# ---------------------------------------------------------------- 工厂分流

def test_create_backend_dispatch_jina_mlx():
    from agent.rag.model_backends import (
        JinaMLXEmbeddingFunction, create_embedding_backend)

    ef = create_embedding_backend("mlx", "jinaai/jina-embeddings-v3", MLX_DIR)
    assert isinstance(ef, JinaMLXEmbeddingFunction)
    # 前缀按 jina 模型族解析（embedding_prefixes）
    assert ef._document_prefix == "passage: "
    assert ef._query_prefix == "query: "


def test_create_backend_dispatch_jina_onnx():
    from agent.rag.model_backends import (
        JinaONNXEmbeddingFunction, create_embedding_backend)

    ef = create_embedding_backend("onnx", "jinaai/jina-embeddings-v3", ONNX_DIR)
    assert isinstance(ef, JinaONNXEmbeddingFunction)


def test_create_backend_jina_fastembed_rejected():
    from agent.rag.model_backends import create_embedding_backend

    # fastembed 无 jina-embeddings-v3，绝不允许静默回退成别的模型
    with pytest.raises(ValueError):
        create_embedding_backend(
            "fastembed", "jinaai/jina-embeddings-v3", ONNX_DIR)


# ------------------------------------------------------------ 真实模型（集成）

@pytest.mark.skipif(not _HAS_ONNX, reason="缺少本地 ONNX 模型")
def test_jina_onnx_embed_and_rank():
    from agent.rag.model_backends import JinaONNXEmbeddingFunction

    ef = JinaONNXEmbeddingFunction(ONNX_DIR)
    docs = ef.embed_documents(DOCS)          # retrieval.passage
    assert len(docs) == 2 and len(docs[0]) == 1024
    q = ef.embed_query(QUERIES[0])           # retrieval.query
    assert q.shape == (1024,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-3  # 已 L2 归一化
    sims = [float(np.dot(q, d)) for d in docs]
    assert sims[0] > sims[1], f" Berlin 查询应命中 Berlin 文档: {sims}"


@pytest.mark.skipif(not _HAS_MLX, reason="缺少本地 MLX 模型")
def test_jina_mlx_embed_and_rank():
    from agent.rag.model_backends import JinaMLXEmbeddingFunction

    ef = JinaMLXEmbeddingFunction(MLX_DIR)
    docs = ef.embed_documents(DOCS)          # retrieval.passage
    assert len(docs) == 2 and len(docs[0]) == 1024
    q = ef.embed_query(QUERIES[1])           # retrieval.query
    assert q.shape == (1024,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-3
    sims = [float(np.dot(q, d)) for d in docs]
    assert sims[1] > sims[0], f"海洋查询应命中 Pacific 文档: {sims}"


# ------------------------------------------------- VectorStore int8 管线（伪 EF）

class _FakeJinaEF:
    """确定性伪嵌入：1024 维输出，信号维度决定截断/重归一化后的排序。

    doc A 的信号在第 0 维（截 256 后保留）；doc B 的信号在第 700 维
    （截 256 后被切掉、归一化守卫下变为零向量）——以此验证 jina 管线的
    「截 256 维 + 重归一化」语义真正生效。
    """

    def __init__(self):
        self.calls: list = []

    @staticmethod
    def _vec(sig_dim: int) -> list:
        v = np.zeros(1024, dtype=np.float32)
        v[sig_dim] = 1.0
        return v.tolist()

    def embed_documents(self, texts):
        self.calls.extend(texts)
        return [self._vec(0 if "A" in t else 700) for t in texts]

    def __call__(self, texts):
        return self.embed_documents(texts)

    def embed_query(self, text):
        return self._vec(0)

    def name(self):
        return "fake:jina-embeddings-v3"

    def close(self):
        pass


def test_vector_store_jina_int8_pipeline(tmp_path):  # noqa: C901 - 单测直白断言
    from agent.rag.vector_store import VectorStore

    ef = _FakeJinaEF()
    vs = VectorStore(
        persist_dir=str(tmp_path),
        embedding_model="test-jina-embeddings-v3",
        embedding_function=ef,
        embedding_backend="mlx",
        matryoshka_enabled=True,   # 启用裁剪 → int8 管线
        matryoshka_dim=256,        # 裁剪维度由配置控制（不再固定 256）
    )
    chunks = [
        {"text": "doc A", "source": "a.md", "chunk_index": 0},
        {"text": "doc B", "source": "b.md", "chunk_index": 0},
    ]
    assert vs.add(chunks) == 2

    table = vs._ensure_table()
    vec_type = table.schema.field("vector").type
    # 截 256 维；列类型恒 float16（int8 只是「值量化」）
    assert vec_type.list_size == 256
    assert pa.types.is_float16(vec_type.value_type)

    # 查询信号在第 0 维 → doc A 命中且相似度≈1（重归一化后同向）
    res = vs.search("needle", top_k=2)
    assert len(res) == 2
    assert res[0]["source"] == "a.md"
    assert res[0]["score"] > 0.99
    assert res[1]["source"] == "b.md"

    # where 过滤生效（原生检索路径）
    res2 = vs.search("needle", top_k=5, where="source = 'b.md'")
    assert res2 and all(r["source"] == "b.md" for r in res2)

    # 入库向量是量化整数（以 f16 承载），值域 ≤127
    rows = table.search().limit(2).to_arrow()
    stored = np.asarray(rows.column("vector").to_pylist(), dtype=np.float32)
    assert np.all(np.equal(np.mod(stored, 1), 0)), "应为量化整数值"
    assert float(np.abs(stored).max()) <= 127
    vs.close()


def test_prepare_vec_jina_trunc_renorm_int8():
    """直接验证对称处理：截 256 → 重归一化 → int8；重归一化在量化之前。"""
    from agent.rag.vector_store import VectorStore

    vs = VectorStore(
        persist_dir="/tmp/nonexistent-jina-vs",
        embedding_model="test-jina-embeddings-v3",
        embedding_function=None,
        matryoshka_enabled=True,
        matryoshka_dim=256,
    )
    v = np.zeros(1024, dtype=np.float32)
    v[:256] = 2.0   # 前 256 维等值 → 截断后范数 = 2*16 = 32
    out = vs._prepare_vec(v)
    # 列恒 float16：int8 指值量化（整数以 f16 承载）
    assert out.dtype == np.float16
    assert out.shape == (256,)
    # 重归一化后每维 = 2/32 = 0.0625 → ×127 ≈ 7.9 → round = 8
    assert int(out[0]) == 8
    # 查询侧对称：同一向量得同一量化表示
    q = vs._prepare_query(v)
    assert np.array_equal(q, out)
