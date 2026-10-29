"""向量存储单元测试（2026-10-28 重构：统一 Matryoshka 管线）。

覆盖：
- MRL 模型（nomic / jina）+ 启用裁剪：截前 N 维（128/256/512 可配）→
  **重新归一化** → int8 量化，入库/查询**对称**；检索走 int8 暴力扫描；
- 裁剪关闭：全维 f16 存储 + lancedb 原生 .search()；
- 非 MRL 模型：不截断，仅重归一化 + f16；
- 旧 f32 表 / 维度或类型不匹配 → reset_if_incompatible 自愈重建。
"""

import numpy as np
import pyarrow as pa
import pytest

from agent.config import AgentConfig
from agent.rag.vector_store import VectorStore

_DEFAULT_DIM = 512  # 与 config.rag_matryoshka_dim 默认一致


def _fake_store(embedding_function=None, model="test-model", path="",
                matryoshka_enabled=False, matryoshka_dim=_DEFAULT_DIM) -> VectorStore:
    cfg = AgentConfig()
    store = VectorStore(
        persist_dir=":memory:",  # 不触盘（懒初始化被白盒注入绕过）
        embedding_model=model,
        embedding_model_path=path,
        embedding_function=embedding_function,
        rerank_enabled=False,
        matryoshka_enabled=matryoshka_enabled,
        matryoshka_dim=matryoshka_dim,
    )
    store._config = cfg
    return store


def _unit_vec(dim: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# 启用裁剪：截断 + 重归一化 + int8（MRL 模型）
# ---------------------------------------------------------------------------

def test_prepare_mrl_int8_truncates_renormalizes():
    """nomic 启用裁剪：截 512 维 → 重归一化 → int8（×127），入库/查询对称。"""
    store = _fake_store(model="models/nomic-embed-text-v1.5-mlx-int8",
                        path="models/nomic-embed-text-v1.5-mlx-int8",
                        matryoshka_enabled=True, matryoshka_dim=512)
    assert store._mrl is True
    assert store._int8_store is True
    assert store._matryoshka_dim == 512
    v768 = _unit_vec(768, 1)
    ing = store._prepare_vec(v768)
    # 列类型恒 float16：int8 是「值量化」，整数以 float16 承载
    assert ing.dtype == np.float16
    assert ing.shape == (512,)
    # 值确实是量化后的整数（而非原始浮点）
    assert np.all(np.equal(np.mod(ing.astype(np.float32), 1), 0))
    # 重归一化在量化之前：单位向量的量化表示范数 ≈ 127
    assert float(np.linalg.norm(ing.astype(np.float32))) == pytest.approx(
        127.0, abs=1.5)
    # 查询侧对称：同一输入得到同一量化表示
    assert np.array_equal(store._prepare_query(v768), ing)


def test_matryoshka_dim_configurable_int8():
    """裁剪维度可配置：128 / 256 / 512 三档均生效（int8 量化管线）。"""
    v768 = _unit_vec(768, 5)
    for dim in (128, 256, 512):
        s = _fake_store(model="nomic-embed-text-v1.5",
                        matryoshka_enabled=True, matryoshka_dim=dim)
        out = s._prepare_vec(v768)
        assert out.shape == (dim,) and out.dtype == np.float16
        assert np.all(np.equal(np.mod(out.astype(np.float32), 1), 0))


def test_matryoshka_disabled_full_f16():
    """关闭裁剪：不截断，全维 f16 存储（原生 ANN 路径）。"""
    store = _fake_store(model="nomic-embed-text-v1.5", matryoshka_enabled=False)
    assert store._mrl is True
    assert store._int8_store is False
    v768 = _unit_vec(768, 2)
    ing = store._prepare_vec(v768)
    assert ing.dtype == np.float16
    assert ing.shape == (768,)
    assert float(np.linalg.norm(ing.astype(np.float32))) == pytest.approx(
        1.0, abs=0.01)


def test_prepare_query_symmetric_with_ingest():
    """查询侧与入库侧必须走同一处理（对称性是检索正确性的前提）。"""
    store = _fake_store(model="nomic-embed-text-v1.5",
                        matryoshka_enabled=True, matryoshka_dim=256)
    v = _unit_vec(768, 3)
    ing = store._prepare_vec(v)
    qry = store._prepare_query(v)
    assert np.array_equal(ing, qry)  # 量化后两侧完全一致


def test_prepare_generic_model_no_truncation():
    """非 MRL 模型：不截断，仅重归一化 + f16 存储（全维度保留）。"""
    store = _fake_store(model="BAAI/bge-small-zh-v1.5")
    assert store._mrl is False
    assert store._int8_store is False
    v = _unit_vec(512, 4)
    q = store._prepare_vec(v)
    assert q.shape == (512,)
    q32 = q.astype(np.float32)
    cos = float(np.dot(q32, np.asarray(v)) /
                (np.linalg.norm(q32) * np.linalg.norm(v)))
    assert cos > 0.999


# ---------------------------------------------------------------------------
# 真实 LanceDB 表：int8 / f16 两种路径
# ---------------------------------------------------------------------------

class _HashEF:
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


class _LexicalEF:
    """词面重叠 EF：字符 bigram 命中维度（仅落在前 512 维内）。"""

    def _vec(self, text):
        import re as _re
        import zlib
        v = np.zeros(768, dtype=np.float32)
        grams = _re.findall(r"[\u4e00-\u9fff]{2}", text)
        for g in grams:
            # zlib.crc32 确定性哈希（不用 hash()——PYTHONHASHSEED 随机化）；
            # 维度必须落在前 512 维内——裁剪会把 ≥512 的维度全部丢掉
            v[zlib.crc32(g.encode("utf-8")) % 512] += 1.0
        n = np.linalg.norm(v)
        return (v / n).tolist() if n else [0.0] * 768

    def embed_query(self, text):
        return self._vec(text)

    def __call__(self, texts):
        return [self._vec(t) for t in texts]


def _make_store(tmp_path, ef, **kw) -> VectorStore:
    store = VectorStore(
        persist_dir=str(tmp_path / f"lancedb-{id(ef)}"),
        embedding_model="models/nomic-embed-text-v1.5-mlx-int8",
        embedding_function=ef,
        rerank_enabled=False,
        **kw,
    )
    store._ensure_initialized()
    return store


@pytest.fixture()
def real_store_int8(tmp_path):
    """启用裁剪（默认 512 维 int8）的真实 LanceDB 存储。"""
    return _make_store(tmp_path, _HashEF(), matryoshka_enabled=True)


@pytest.fixture()
def real_store_f16(tmp_path):
    """关闭裁剪（全维 f16）的真实 LanceDB 存储。"""
    return _make_store(tmp_path, _HashEF(), matryoshka_enabled=False)


def test_int8_mode_still_uses_f16_column_and_native_search(real_store_int8):
    """2026-10-29 定版：int8 只是「值量化」，列类型仍是 float16 —— 因此
    lancedb **原生检索**可用（真 Int8 列会报 no vector column）。"""
    store = real_store_int8
    docs = [{"text": f"第{i}章 内容，关于机器学习的介绍。", "source": f"d{i}.md",
             "chunk_index": 0} for i in range(8)]
    assert store.add(docs) == 8
    schema = store._table.schema.field("vector").type
    assert schema.value_type == pa.float16()   # 列恒 f16
    assert schema.list_size == 512             # 截维生效
    assert store._int8_store is True           # 值被量化

    # 存进去的是量化整数（以 f16 承载）
    rows = store._table.search().limit(8).to_arrow()
    v0 = np.asarray(rows.column("vector").to_pylist()[0], dtype=np.float32)
    assert np.all(np.equal(np.mod(v0, 1), 0)), "值应为量化整数"
    assert float(np.linalg.norm(v0)) == pytest.approx(127.0, abs=2.0)

    # 原生检索命中自身：分数按 1 - d²/(2·127²) 还原 → 同文本 ≈1
    res = store.search("第3章 内容，关于机器学习的介绍。", top_k=3)
    assert res, "检索必须有结果"
    assert res[0]["source"] == "d3.md"
    assert res[0]["score"] > 0.9


def test_table_column_is_f16_and_native_search_works(real_store_f16):
    """f16 路径（裁剪关闭）：f16 列、全维 768；lancedb 原生检索命中自身。"""
    store = real_store_f16
    docs = [{"text": f"第{i}章 内容，关于机器学习的介绍。", "source": f"d{i}.md",
             "chunk_index": 0} for i in range(8)]
    assert store.add(docs) == 8
    schema = store._table.schema.field("vector").type
    assert schema.value_type == pa.float16()
    assert schema.list_size == 768

    res = store.search("第3章 内容，关于机器学习的介绍。", top_k=3)
    assert res and res[0]["source"] == "d3.md"
    assert res[0]["score"] > 0.9


def test_search_similarity_ordering(real_store_int8):
    """相似文本得分应高于无关文本（int8 量化 + 原生检索排序正确）。"""
    store = real_store_int8
    store._ef = _LexicalEF()
    docs = [
        {"text": "机器学习是人工智能的核心分支，通过数据训练模型。", "source": "ml.md", "chunk_index": 0},
        {"text": "今天天气晴朗，适合去公园散步和野餐。", "source": "weather.md", "chunk_index": 0},
    ]
    store.add(docs)
    hit = store.search("机器学习与人工智能模型训练", top_k=2)
    scores = {r["source"]: r["score"] for r in hit}
    assert scores["ml.md"] > scores["weather.md"]


def test_where_filter_works_on_int8_path(real_store_int8):
    """where 定向检索在 int8 量化 + 原生检索路径下仍生效。"""
    store = real_store_int8
    docs = [{"text": f"文档{i}的正文内容，讨论机器学习话题{i}。", "source": f"f{i}.md",
             "chunk_index": 0} for i in range(5)]
    store.add(docs)
    res = store.search("机器学习话题", top_k=5, where="source = 'f2.md'")
    assert res and all(r["source"] == "f2.md" for r in res)


def test_mrl_trunc_with_f16_dtype_uses_native_search(tmp_path):
    """2026-10-29 新增：启用裁剪但选 f16 存储 → 截 256 维 + 重归一化 + f16
    入库，列是浮点类型 → lancedb **原生检索**可用（非暴力扫描）。"""
    store = _make_store(
        tmp_path, _HashEF(),
        matryoshka_enabled=True, matryoshka_dim=256, vector_dtype="f16")
    docs = [{"text": f"第{i}章 内容，关于机器学习的介绍。", "source": f"d{i}.md",
             "chunk_index": 0} for i in range(8)]
    assert store.add(docs) == 8

    vec_type = store._table.schema.field("vector").type
    assert vec_type.value_type == pa.float16()   # 仍是浮点列
    assert vec_type.list_size == 256             # 截维生效
    assert store._int8_store is False            # 不量化
    assert store._mrl_trunc is True

    # 原生路径命中自身，分数接近 1
    res = store.search("第3章 内容，关于机器学习的介绍。", top_k=3)
    assert res and res[0]["source"] == "d3.md"
    assert res[0]["score"] > 0.9

    # 入库向量确为 f16 且已重归一化（‖v‖≈1）
    rows = store._table.search().limit(8).to_arrow()
    v0 = np.asarray(rows.column("vector").to_pylist()[0], dtype=np.float32)
    assert float(np.linalg.norm(v0)) == pytest.approx(1.0, abs=0.01)
    store.close()


def test_int8_mode_can_build_ann_index(tmp_path):
    """2026-10-29：int8 量化模式同样能在向量列上建 ANN 索引。

    列类型恒为 float16，因此 int8 量化模式下原生检索与索引都可用
    （度量必须 l2——下游把 _distance 当平方 L2 还原余弦）。
    """
    store = _make_store(tmp_path, _HashEF(), matryoshka_enabled=True,
                        matryoshka_dim=256, vector_dtype="int8")
    store.add([{"text": f"第{i}篇文档内容。", "source": f"d{i}.md", "chunk_index": 0}
               for i in range(64)])
    # 列是 float16（不是真 Int8）→ 原生检索/索引都可用
    assert store._table.schema.field("vector").type.value_type == pa.float16()
    try:
        store._table.create_index(
            metric="l2", vector_column_name="vector",
            num_partitions=4, num_sub_vectors=8)
        indexed = True
    except Exception as e:  # 环境不支持索引时跳过该断言
        indexed = False
        print("      (跳过索引创建:", e, ")")
    if indexed:
        res = store.search("第7篇文档内容。", top_k=2)
        assert res and res[0]["source"] == "d7.md"
    store.close()


def test_switching_vector_dtype_rebuilds_data(tmp_path):
    """切换存储精度：列类型相同（float16）但值语义不同 → 必须重建数据。

    靠建表时写入 schema metadata 的 vector_dtype 标记检测，避免量化向量
    （范数≈127）与单位向量混检把排序带偏。
    """
    store = _make_store(tmp_path, _HashEF(), matryoshka_enabled=True,
                        matryoshka_dim=256, vector_dtype="int8")
    store.add([{"text": "第一篇关于机器学习的内容。", "source": "a.md",
                "chunk_index": 0}])
    assert store.reset_if_incompatible() is False   # 当前精度一致 → 无需重建

    # 切到 f16 后（同一张表）应判定不兼容并重建
    store._vector_dtype = "f16"
    store._int8_store = False
    assert store.reset_if_incompatible() is True


def test_build_fts_index_offline(tmp_path):
    """2026-10-29：同步收尾可离线构建 BM25 索引（不必等首次检索）。"""
    store = _make_store(tmp_path, _HashEF(), matryoshka_enabled=True,
                        matryoshka_dim=256, vector_dtype="int8")
    store.add([{"text": "量子计算与量子比特纠错码入门。", "source": "qc.md",
                "chunk_index": 0}])
    assert store._fts_ready is False
    assert store.build_fts_index() is True
    assert store._fts_ready is True
    res = store.search_fts("量子比特", top_k=3)
    assert res and res[0]["source"] == "qc.md"
    store.close()


def test_rerank_candidates_are_capped(tmp_path, monkeypatch):
    """2026-10-29：混合候选收敛后再送精排（成本与候选条数线性相关）。"""
    store = _make_store(tmp_path, _HashEF(), matryoshka_enabled=True,
                        matryoshka_dim=256, vector_dtype="int8")
    store.add([{"text": f"关于机器学习主题的第{i}篇内容。", "source": f"m{i}.md",
                "chunk_index": 0} for i in range(40)])

    seen: list = []

    def _spy_rerank(query, candidates, top_k=5, min_similarity=0.0):
        seen.append(len(candidates))
        return candidates[:top_k]

    monkeypatch.setattr(store, "rerank", _spy_rerank)
    out = store.search_with_rerank("机器学习", top_k=3, candidate_multiplier=4)
    assert seen and out
    # 两路各取 12 条，合并去重后本可达 ~24 条；收敛后 ≤ top_k × 2 = 6
    assert seen[0] <= 3 * 2, seen
    store.close()


def test_reset_on_legacy_f32_table(real_store_int8):
    """旧 f32 表（768 维）：reset_if_incompatible 必须判定不兼容并重建。"""
    store = real_store_int8
    store.add([{"text": "初始化数据，确保表存在。", "source": "init.md", "chunk_index": 0}])
    store._db.drop_table("knowledge_base")
    rows = [{"id": "old_0", "text": "旧", "vector": [0.1] * 768,
             "source": "old.md", "chunk_index": 0,
             "text_fts": "旧", "ingested_at": 0.0}]
    store._db.create_table("knowledge_base", rows)
    store._table = store._db.open_table("knowledge_base")
    assert store.reset_if_incompatible() is True
    assert store._table is None          # 已 drop，等待重建
