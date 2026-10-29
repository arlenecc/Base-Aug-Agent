"""全链路端到端回归（每日）：切片 → 嵌入 → 裁剪 → 重归一化 → int8 → 入库
→ 查询对称 → 检索效果 → where 过滤。

使用**真实组件**：chunker（800 字符/语义）、JinaMLXEmbeddingFunction（真实
模型 weights.safetensors）、VectorStore（真实 LanceDB）。任何一环坏了
（前缀丢失、task 混用、归一化缺失、量化错位、检索退化）都会在此暴露。

2026-10-28 由人工端到端验证脚本固化而来；模型文件缺失时整组跳过
（CI/无权重环境不影响其余单测）。
"""

import os
import sys

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.config import AgentConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JINA_DIR = os.path.join(ROOT, "models", "jina-embeddings-v3-mlx-int4")
_HAS_MODEL = os.path.isfile(os.path.join(JINA_DIR, "weights.safetensors"))

pytestmark = pytest.mark.skipif(
    not _HAS_MODEL, reason="缺少本地 jina-embeddings-v3 MLX 模型")

DOC_TEXT = (
    "人工智能正在重塑知识工作的每一个环节。从文档解析到语义检索，从辅助写作到自动摘要，"
    "本地大模型让这些能力可以脱离云端、在个人隐私边界内运行。\n\n"
    "知识库问答系统通过把文档切片、向量化并建立检索索引，使模型能够在回答时引用真实资料"
    "而非凭空生成。检索增强生成（RAG）的核心是：先用嵌入模型把用户问题与文档片段映射到同一"
    "向量空间，召回最相关的若干片段，再交给生成模型压缩、归纳成自然语言答案。\n\n"
    "切片策略决定了召回的粒度——过大的切片会引入噪声，过小的切片会割裂上下文。语义切片在"
    "句边界的基础上进一步按主题聚合，使每个片段内部高度相关。重排序模型则在召回之后对候选"
    "片段做精细打分，把真正相关的内容排在前面。\n\n"
    "对于中文场景，嵌入模型的选择尤为关键：多语种强模型往往比专为中文训练的小模型在跨域检索"
    "上更稳健。向量维度也是一个工程权衡点——Matryoshka 表示允许在存储与精度之间灵活取舍。\n\n"
    "与此同时，解析引擎决定了原始文档能被还原成多少结构化信息：版面理解模型能保留标题层级与"
    "表格，OCR 决定了扫描件的可读性。整个链路的每一步都可能在速度与质量之间做取舍，而本地"
    "部署让这些取舍能够针对具体硬件与隐私要求定制。\n\n"
    "部署层面，量化技术让大模型能在消费级设备上运行：int4 权重把内存需求压到原来的四分之一，"
    "而精度损失在多数任务上可以忽略。苹果硅芯片的统一内存架构特别适合这种本地推理范式。\n\n"
    "最终，一个设计良好的本地知识库应当做到：数据不出设备、检索毫秒级响应、答案有据可查。"
    "这三点构成了个人隐私优先的 AI 基础设施的基石，也是本项目持续优化的方向。"
)

DOCS = {
    "geo.md": "北京故宫是明清两代的皇家宫殿，有大小宫殿七十多座。中轴线上的太和殿是核心建筑，游览从午门进神武门出。",
    "code.md": "Python 的列表推导式可以用一行代码生成列表，写法是 [表达式 for 变量 in 可迭代对象]。它比等价的 for 循环更简洁也更快。",
    "db.md": "数据库查询慢时先看执行计划，再给 WHERE 和 JOIN 涉及的列建索引。复合索引要遵循最左前缀原则。",
    "health.md": "长时间近距离用眼是近视加深的主要原因。每天保证两小时以上户外阳光活动，能有效延缓近视发生。",
    "finance.md": "基金定投是定期定额买入基金，摊平成本、弱化择时。适合波动较大的指数基金，长期坚持能享受复利。",
    "cook.md": "红烧肉要选肥瘦相间的五花肉，先焯水再炒糖色，小火慢炖一小时以上，肉质软糯肥而不腻。",
    "history.md": "丝绸之路是古代连接中国与中亚、欧洲的陆上贸易通道。张骞两次出使西域开辟了这条商路。",
    "space.md": "黑洞是引力强到连光都无法逃逸的天体，由大质量恒星坍缩形成。事件视界是它的边界。",
}
QUERIES = [
    ("去北京想看明清皇家宫殿", "geo.md"),
    ("一行代码生成列表怎么写", "code.md"),
    ("孩子看手机多了会近视吗", "health.md"),
    ("每月剩点钱做什么理财", "finance.md"),
    ("五花肉怎么炖才软糯不腻", "cook.md"),
    ("古代通往欧洲的商贸通道", "history.md"),
]


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def cfg():
    return AgentConfig()


@pytest.fixture(scope="module")
def jina_ef(cfg):
    """真实 jina 嵌入后端（模块级只加载一次，~7s）。"""
    from agent.rag.model_backends import create_embedding_backend

    return create_embedding_backend(
        cfg.rag_embedding_backend, cfg.rag_embedding_model,
        cfg.rag_embedding_model_path)


@pytest.fixture(scope="module")
def retrieval_store(tmp_path_factory, cfg, jina_ef):
    """默认配置（裁剪 256 → int8）的 VectorStore，已入库 8 篇多领域文档。"""
    from agent.rag.vector_store import VectorStore

    vs = VectorStore(
        persist_dir=str(tmp_path_factory.mktemp("e2e-lancedb")),
        embedding_model=cfg.rag_embedding_model,
        embedding_model_path=cfg.rag_embedding_model_path,
        embedding_function=jina_ef,
        embedding_backend=cfg.rag_embedding_backend,
        matryoshka_enabled=cfg.rag_matryoshka_enabled,
        matryoshka_dim=cfg.rag_matryoshka_dim,
    )
    vs.add([{"text": t, "source": s, "chunk_index": 0} for s, t in DOCS.items()])
    yield vs
    vs.close()


# ---------------------------------------------------------------- 1. 切片

def test_stage1_chunker_zero_loss_and_limits(cfg):
    """切片：语义切片产物每片 ≤ chunk_size，字符总量覆盖原文（无丢失）。"""
    from agent.rag.chunker import chunk_documents

    chunks = chunk_documents(
        [{"path": "rag_notes.md", "text": DOC_TEXT}],
        chunk_size=cfg.rag_chunk_size, chunk_overlap=cfg.rag_chunk_overlap,
        use_semantic=cfg.rag_semantic_chunking,
    )
    assert chunks, "切片不能为空"
    assert all(len(c["text"]) <= cfg.rag_chunk_size for c in chunks), \
        f"切片超限: max={max(len(c['text']) for c in chunks)}"
    assert sum(len(c["text"]) for c in chunks) >= len(DOC_TEXT), \
        "切片总字符 < 原文：内容丢失"


# ------------------------------------------------ 2-5. 入库链路逐级核对

def test_stage2_to_5_ingest_chain_int8(cfg, jina_ef, tmp_path, monkeypatch):
    """入库链路：原始 1024 归一化 → 截 256（范数必降）→ 重归一化 → int8
    ×127 → 库内存储逐字节一致。"""
    from agent.rag.vector_store import VectorStore

    raw_capture = {}
    cls = type(jina_ef)
    real_call = cls.__call__

    def _spy(self, texts, batch_size=None):
        out = real_call(self, texts, batch_size=batch_size)
        for t, v in zip(texts, out):
            raw_capture[t] = np.asarray(v, dtype=np.float32)
        return out

    monkeypatch.setattr(cls, "__call__", _spy)  # dunder 只能打在类上，自动恢复

    vs = VectorStore(
        persist_dir=str(tmp_path / "lancedb"),
        embedding_model=cfg.rag_embedding_model,
        embedding_model_path=cfg.rag_embedding_model_path,
        embedding_function=jina_ef,
        embedding_backend=cfg.rag_embedding_backend,
        matryoshka_enabled=cfg.rag_matryoshka_enabled,
        matryoshka_dim=cfg.rag_matryoshka_dim,
    )
    chunk_texts = [t for t in DOCS.values()]
    assert vs.add([{"text": t, "source": "e2e.md", "chunk_index": i}
                   for i, t in enumerate(chunk_texts)]) == len(chunk_texts)

    vec_type = vs._ensure_table().schema.field("vector").type
    # 列类型恒 float16（int8 = 值量化，整数以 f16 承载 → 原生检索/索引可用）
    assert vec_type.value_type == pa.float16()
    assert vec_type.list_size == cfg.rag_matryoshka_dim

    rows = vs._table.search().limit(len(chunk_texts)).to_arrow()
    stored = dict(zip(rows.column("text").to_pylist(),
                      rows.column("vector").to_pylist()))
    for text in chunk_texts:
        raw = raw_capture[text]
        assert abs(float(np.linalg.norm(raw)) - 1.0) < 1e-3, \
            "模型原始输出未归一化（task LoRA 管线变了？）"
        trunc = raw[:cfg.rag_matryoshka_dim]
        assert float(np.linalg.norm(trunc)) < 0.99, \
            "截断后范数未下降——截断没有生效？"
        expect = np.clip(np.round(trunc / np.linalg.norm(trunc) * 127),
                         -127, 127).astype(np.float16)
        assert np.allclose(expect, np.asarray(stored[text], dtype=np.float16)), \
            f"入库向量与「截断→重归一化→量化」手算不一致: {text[:20]}"
    vs.close()


def test_stage6_query_side_symmetric(cfg, jina_ef, retrieval_store):
    """查询侧与入库侧走同一管线：embed_query → 截256 → 重归一化 → int8。"""
    q_raw = np.asarray(
        jina_ef.embed_query("检索增强生成的核心流程是什么？"), dtype=np.float32)
    assert abs(float(np.linalg.norm(q_raw)) - 1.0) < 1e-3
    q = retrieval_store._prepare_query(q_raw)
    # 查询侧 dtype=float32，值同样是量化整数（与入库侧同一管线）
    assert q.dtype == np.float32 and q.shape == (cfg.rag_matryoshka_dim,)
    manual = np.clip(
        np.round((q_raw[:cfg.rag_matryoshka_dim]
                  / np.linalg.norm(q_raw[:cfg.rag_matryoshka_dim])) * 127),
        -127, 127).astype(np.float32)
    assert np.allclose(q, manual), "查询侧处理与入库侧管线不一致"


# ---------------------------------------------------------------- 7. 检索效果

def test_stage7_retrieval_effectiveness(retrieval_store):
    """6 个口语化中文查询全部 Top-1 命中（int8 暴力扫描路径）。"""
    for q, target in QUERIES:
        res = retrieval_store.search(q, top_k=3)
        assert res, f"无结果: {q}"
        assert res[0]["source"] == target, \
            f"{q!r} 期望 {target}, 实际 {res[0]['source']} (score={res[0]['score']:.3f})"
        assert res[0]["score"] > res[1]["score"], \
            f"{q!r} 首名与次名并列，区分度异常"


def test_stage7_where_filter_on_int8_path(retrieval_store):
    """where 过滤在 int8 暴力扫描路径下定向正确。"""
    res = retrieval_store.search("黑洞的边界叫什么", top_k=3,
                                 where="source = 'space.md'")
    assert res and all(r["source"] == "space.md" for r in res)
