"""f16 向量存储单元测试（2026-10-27）。

覆盖用户硬性要求：
- nomic（Matryoshka）：入库/查询向量截前 rag_matryoshka_dim 维（默认 512，
  可配置）→ **重新归一化** → f16 存储；
- 其它模型：不截断，仅重归一化（f16 存储）；
- 入库与查询**对称**执行同一处理；
- f16 向量列建表 + lancedb 原生检索的正确性（相似近、随机远）；
- 旧 f32 表 / 维度不匹配 → reset_if_incompatible 自愈重建。
"""

import numpy as np
import pytest

from agent.config import AgentConfig
from agent.rag.vector_store import (
    _MATRYOSHKA_DIM,
    VectorStore,
)


def _fake_store(embedding_function=None, model="test-model", path="",
                matryoshka_dim=None) -> VectorStore:
    cfg = AgentConfig()
    store = VectorStore(
        persist_dir=":memory:",  # 不触盘（懒初始化被白盒注入绕过）
        embedding_model=model,
        embedding_model_path=path,
        embedding_function=embedding_function,
        rerank_enabled=False,
    )
    if matryoshka_dim is not None:
        store._matryoshka_dim = matryoshka_dim
    store._config = cfg
    return store


def _unit_vec(dim: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# 截断 + 重归一化管线性质
# ---------------------------------------------------------------------------

def test_prepare_matryoshka_truncates_and_renormalizes():
    """nomic：截 512 维（默认）后必须重归一化（范数=1），且两侧对称。"""
    store = _fake_store(model="models/nomic-embed-text-v1.5-mlx-int8",
                        path="models/nomic-embed-text-v1.5-mlx-int8")
    assert store._matryoshka is True
    assert store._matryoshka_dim == 512
    v768 = _unit_vec(768, 1)
    ing = store._prepare_vec(v768)
    assert isinstance(ing, np.ndarray) and ing.dtype == np.float16
    assert ing.shape == (_MATRYOSHKA_DIM,) == (512,)
    # 截断+重归一化后的 f16 向量范数应 ≈ 1（f16 舍入误差内）
    assert float(np.linalg.norm(ing.astype(np.float32))) == pytest.approx(
        1.0, abs=0.01)


def test_matryoshka_dim_configurable():
    """截断维度可配置：256 / 0（不截断）均生效。"""
    v768 = _unit_vec(768, 5)
    s256 = _fake_store(model="nomic-embed-text-v1.5", matryoshka_dim=256)
    assert s256._prepare_vec(v768).shape == (256,)
    s0 = _fake_store(model="nomic-embed-text-v1.5", matryoshka_dim=0)
    assert s0._prepare_vec(v768).shape == (768,)


def test_prepare_query_symmetric_with_ingest():
    """查询侧与入库侧必须走同一处理（对称性是检索正确性的前提）。"""
    store = _fake_store(model="nomic-embed-text-v1.5")
    v = _unit_vec(768, 2)
    ing = store._prepare_vec(v).astype(np.float32)
    qry = store._prepare_query(v)
    np.testing.assert_allclose(ing, qry.astype(np.float16).astype(np.float32),
                               atol=1e-3)


def test_prepare_generic_model_no_truncation():
    """非 nomic 模型：不截断，仅重归一化 + f16 存储（全维度保留）。"""
    store = _fake_store(model="BAAI/bge-small-zh-v1.5")
    assert store._matryoshka is False
    v = _unit_vec(512, 3)
    q = store._prepare_vec(v)
    assert q.shape == (512,)
    # 方向与原向量一致：余弦（f32 精度）≈ 1
    q32 = q.astype(np.float32)
    cos = float(np.dot(q32, np.asarray(v)) /
                (np.linalg.norm(q32) * np.linalg.norm(v)))
    assert cos > 0.999


# ---------------------------------------------------------------------------
# 真实 LanceDB 表：f16 列建表 + 原生检索
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_store(tmp_path):
    """真实 LanceDB 存储的 VectorStore（无模型依赖：直接注入确定性 EF）。"""
    class _EF:
        """确定性 768 维单位向量 EF：文本哈希决定向量。"""
        def embed_query(self, text):
            return self._vec(text)

        def __call__(self, texts):
            return [self._vec(t) for t in texts]

        @staticmethod
        def _vec(text):
            import zlib
            rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
            v = rng.standard_normal(768).astype(np.float32)
            return (v / np.linalg.norm(v)).tolist()

    store = VectorStore(
        persist_dir=str(tmp_path / "lancedb"),
        embedding_model="models/nomic-embed-text-v1.5-mlx-int8",
        embedding_function=_EF(),
        rerank_enabled=False,
    )
    store._ensure_initialized()
    return store


def test_table_column_is_f16_and_native_search_works(real_store):
    """入库：f16 列建表；检索：同一文本的查询应命中自身且分数接近 1。"""
    store = real_store
    docs = [{"text": f"第{i}章 内容，关于机器学习的介绍。", "source": f"d{i}.md",
             "chunk_index": 0} for i in range(8)]
    added = store.add(docs)
    assert added == 8
    schema = store._table.schema.field("vector").type
    assert schema.value_type == __import__("pyarrow").float16()
    assert schema.list_size == 512

    # 查询与某文档文本相同 → 应以高分数命中该文档（lancedb 原生 .search()）
    res = store.search("第3章 内容，关于机器学习的介绍。", top_k=3)
    assert res, "检索必须有结果"
    assert res[0]["source"] == "d3.md"
    assert res[0]["score"] > 0.9   # 同文本 → 余弦≈1（f16 舍入误差内）


def test_search_similarity_ordering(real_store):
    """相似文本得分应高于无关文本（排序正确）。

    real_store 的 EF 是哈希随机向量、无语义——这里注入一个词面重叠 EF
    （字符 bigram 命中维度），使「机器学习」文本与查询共享维度。
    """
    import re as _re
    import zlib

    class _LexicalEF:
        def _vec(self, text):
            v = np.zeros(768, dtype=np.float32)
            grams = _re.findall(r"[\u4e00-\u9fff]{2}", text)
            for g in grams:
                # zlib.crc32 确定性哈希（不用 hash()——PYTHONHASHSEED 随机化
                # 会让维度映射随进程变化）；维度必须落在前 512 维内——
                # Matryoshka 截断会把 ≥512 的维度全部丢掉
                v[zlib.crc32(g.encode("utf-8")) % _MATRYOSHKA_DIM] += 1.0
            n = np.linalg.norm(v)
            return (v / n).tolist() if n else [0.0] * 768

        def embed_query(self, text):
            return self._vec(text)

        def __call__(self, texts):
            return [self._vec(t) for t in texts]

    store = real_store
    store._ef = _LexicalEF()
    docs = [
        {"text": "机器学习是人工智能的核心分支，通过数据训练模型。", "source": "ml.md", "chunk_index": 0},
        {"text": "今天天气晴朗，适合去公园散步和野餐。", "source": "weather.md", "chunk_index": 0},
    ]
    store.add(docs)
    hit = store.search("机器学习与人工智能模型训练", top_k=2)
    scores = {r["source"]: r["score"] for r in hit}
    assert scores["ml.md"] > scores["weather.md"]


def test_reset_on_legacy_f32_table(real_store):
    """旧 f32 表（768 维）：reset_if_incompatible 必须判定不兼容并重建。"""
    store = real_store
    store.add([{"text": "初始化数据，确保表存在。", "source": "init.md", "chunk_index": 0}])
    # 手工替换为一张旧版 f32 768 维表
    store._db.drop_table("knowledge_base")
    rows = [{"id": "old_0", "text": "旧", "vector": [0.1] * 768,
             "source": "old.md", "chunk_index": 0,
             "text_fts": "旧", "ingested_at": 0.0}]
    store._db.create_table("knowledge_base", rows)
    store._table = store._db.open_table("knowledge_base")
    assert store.reset_if_incompatible() is True
    assert store._table is None          # 已 drop，等待重建


def test_where_filter_still_works(real_store):
    """where 定向检索（rag_search source=...）在原生检索下仍生效。"""
    store = real_store
    docs = [{"text": f"文档{i}的正文内容，讨论机器学习话题{i}。", "source": f"f{i}.md",
             "chunk_index": 0} for i in range(5)]
    store.add(docs)
    res = store.search("机器学习话题", top_k=5, where="source = 'f2.md'")
    assert res and all(r["source"] == "f2.md" for r in res)
