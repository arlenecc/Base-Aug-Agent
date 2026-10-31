"""真实模型（jina v5-nano MLX 嵌入 + jina reranker ONNX 重排）RAG 全链路端到端测试。

区别于 ``tests/test_rag_e2e.py``（使用确定性假嵌入，仅验证管线结构与格式），
本文件加载**真实本地模型**，验证默认配置下整条链路：

  1. 建库 + 语义切片（长文档按 800 字硬限拆分）；
  2. 向量入库为 128 维、int8 量化（L2 重归一化后整体缩放到 ~127）；
  3. 混合检索 + jina 重排的**真实召回质量**（top1 命中正确主题文档）；
  4. status 查询、无关查询不崩溃。

模型缺失时整文件 skip（不阻塞常规测试 / CI）。本测试较重（加载 MLX 嵌入 +
ONNX 重排模型，单次 ingest 约数十秒），故只在真实模型就绪时运行。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

MLX_EMBED_DIR = os.path.join(ROOT, "models", "jina-embeddings-v5-text-nano-mlx-int4")
ONNX_RERANK_DIR = os.path.join(ROOT, "models", "jina-reranker-v2-base-multilingual-onnx")
_HAS_MLX = os.path.isfile(os.path.join(MLX_EMBED_DIR, "model.safetensors"))
_HAS_ONNX = os.path.isfile(os.path.join(ONNX_RERANK_DIR, "model_quantized.onnx"))
_HAS_REAL = _HAS_MLX and _HAS_ONNX

pytestmark = pytest.mark.skipif(
    not _HAS_REAL,
    reason="缺少本地 jina 模型（MLX 嵌入 models/jina-embeddings-v5-text-nano-mlx-int4 "
           "/ ONNX 重排 models/jina-reranker-v2-base-multilingual-onnx）",
)


def _abs_model(p: str) -> str:
    """相对路径按仓库根解析，避免依赖运行 cwd。"""
    if not p or os.path.isabs(p):
        return p
    return os.path.join(ROOT, p)


# 6 篇多主题中文文档；quantum 为长文（>800 字）用于触发语义切片拆分。
DOCS = {
    "apple.md": """# 苹果公司

苹果公司（Apple Inc.）是一家总部位于美国加利福尼亚州库比蒂诺的科技公司。
苹果公司由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于 1976 年创立。
公司最早的产品是 Apple I 个人电脑，随后推出的 Apple II 取得了巨大商业成功。
1984 年苹果发布了采用图形用户界面的 Macintosh 电脑。
2007 年苹果推出第一代 iPhone 智能手机，彻底改变了手机行业。
苹果公司目前的主要产品包括 iPhone、iPad、Mac 电脑、Apple Watch 和 AirPods。
蒂姆·库克在 2011 年接替乔布斯出任苹果公司首席执行官。""",

    "ml_basics.md": """# 机器学习基础

机器学习是人工智能的一个分支，研究如何让计算机从数据中自动学习规律。
监督学习使用带标签的数据训练模型，常见任务包括分类和回归。
无监督学习则处理无标签数据，典型算法有聚类和降维。
过拟合是指模型在训练数据上表现很好，但在未见过的测试数据上表现很差的现象。
为防止过拟合，常用方法包括正则化、交叉验证和早停。
梯度下降是一种通过迭代调整参数来最小化损失函数的优化算法。
神经网络由多层神经元组成，深度学习使用包含很多隐藏层的神经网络。""",

    "tea_culture.md": """# 中国茶文化

中国是世界上最早发现和利用茶叶的国家，茶文化源远流长。
绿茶是不发酵茶，代表有西湖龙井和碧螺春。
红茶是全发酵茶，正山小种和祁门红茶都属于红茶。
普洱茶主要产自云南，属于后发酵茶，有生普和熟普之分。
乌龙茶是半发酵茶，铁观音和大红袍都是著名的乌龙茶。
泡茶讲究水温、投茶量和出汤时间，不同茶类有不同要求。
茶道精神强调和、敬、清、寂，体现了中国人的生活美学。""",

    "solar_system.md": """# 太阳系行星

太阳系以太阳为中心，包含八大行星。
按照离太阳的距离从近到远，依次是水星、金星、地球、火星、木星、土星、天王星、海王星。
木星是太阳系中体积和质量最大的行星，是一颗气态巨行星。
木星拥有众多卫星，其中木卫三（盖尼米德）是太阳系中最大的卫星。
土星以其壮观的行星环系统而闻名，环主要由冰和岩石碎块组成。
火星表面呈现红色，是因为其土壤富含氧化铁（铁锈）。
地球是目前已知唯一存在生命的行星，约 71% 的表面被液态水覆盖。""",

    "protein.md": """# 蛋白质结构与功能

蛋白质是生命活动的主要承担者，由氨基酸通过肽键连接而成。
蛋白质的一级结构是指氨基酸的线性排列顺序，由基因编码决定。
二级结构是指多肽链局部的规则构象，常见有 α 螺旋和 β 折叠。
三级结构是指整条多肽链在三维空间中的折叠方式。
四级结构是指由多条多肽链（亚基）组装形成的空间结构。
蛋白质的特定空间构象决定了它的生物学功能，构象改变常导致功能变化。
酶是一类具有催化功能的蛋白质，能显著降低化学反应的活化能。""",

    "quantum.md": """# 量子计算入门

量子计算是一种基于量子力学原理的全新计算范式，它与经典计算有本质区别。
在经典计算机中，信息的基本单元是比特，只能处于 0 或 1 两种状态之一。
量子计算机使用量子比特（qubit）作为信息单元，由于量子叠加原理，
一个量子比特可以同时处于 0 和 1 的叠加态，而非确定地取其中一个值。
当多个量子比特相互纠缠时，整个系统的状态空间会随比特数指数级增长，
这正是量子计算潜在算力的来源。量子门是对量子比特进行操作的基本逻辑单元，
常见的单比特门包括泡利 X、Y、Z 门和阿达马（Hadamard）门。
阿达马门可以把一个确定的基态变成等幅叠加态，是实现量子并行的重要工具。
量子算法方面，秀尔算法能够在多项式时间内完成大整数质因数分解，
对现有的 RSA 公钥密码体系构成了潜在威胁。格罗弗算法则提供了对无序数据库
进行平方根加速搜索的能力。量子纠错是实用化量子计算的关键挑战，
因为量子态极其脆弱，极易受到环境噪声的干扰而发生退相干。
为了对抗退相干，研究者提出了表面码等多种量子纠错码，用多个物理量子比特
编码一个逻辑量子比特，从而检测和纠正错误。当前主流的超导量子计算路线
使用接近绝对零度的稀释制冷机来保持量子比特的相干性，谷歌和 IBM 等公司
都已经构建了包含数十到上百个量子比特的原型机。尽管取得了显著进展，
通用容错量子计算机距离真正实用化仍有相当长的一段路要走。量子计算的发展也
带来了新的密码学挑战，后量子密码学（PQC）因此成为研究热点，NIST 已经
标准化了多种基于格密码的算法来应对量子攻击。除了超导路线，离子阱量子计算
利用囚禁的带电原子作为量子比特，通过激光脉冲实现量子门操作，具有较长的
相干时间。光量子计算则直接用光子作为信息载体，在玻色采样等特定问题上展示了
量子优越性。拓扑量子计算试图利用任意子（anyon）的非阿贝尔统计来实现天然
抗错的量子比特，虽然理论优美但工程实现极为困难。量子模拟是量子计算机最具
前景的应用方向之一，可用于模拟复杂分子的化学反应过程，有望在材料科学和
药物研发中发挥重要作用。总体而言，量子计算正在从实验室演示走向工程化探索，
未来十年将是决定其实用化进程的关键窗口期。""",
}


@pytest.fixture(scope="module")
def rag_real(tmp_path_factory):
    """构建真实模型引擎并入库一次（模型懒加载，整模块复用）。"""
    from agent.config import AgentConfig
    from agent.rag.engine import RAGEngine

    ws = tmp_path_factory.mktemp("rag_real_ws")
    kb = tmp_path_factory.mktemp("rag_real_kb")
    for name, text in DOCS.items():
        with open(os.path.join(kb, name), "w", encoding="utf-8") as f:
            f.write(text)

    cfg = AgentConfig()
    engine = RAGEngine(
        workspace=str(ws),
        knowledge_base=str(kb),
        embedding_model=cfg.rag_embedding_model,
        embedding_backend=cfg.rag_embedding_backend,
        embedding_model_path=_abs_model(cfg.rag_embedding_model_path),
        embedding_threads=getattr(cfg, "rag_embedding_threads", 6),
        embedding_batch_size=getattr(cfg, "rag_embedding_batch_size", 8),
        embedding_max_length=getattr(cfg, "rag_embedding_max_length", 8192),
        chunk_size=cfg.rag_chunk_size,
        chunk_overlap=cfg.rag_chunk_overlap,
        semantic_chunking=cfg.rag_semantic_chunking,
        rerank_model=cfg.rag_rerank_model,
        rerank_backend=cfg.rag_rerank_backend,
        rerank_model_path=_abs_model(cfg.rag_rerank_model_path),
        rerank_mlx_model_path=_abs_model(cfg.rag_rerank_mlx_model_path)
        if cfg.rag_rerank_mlx_model_path else "",
        rerank_jina_model_path=_abs_model(cfg.rag_rerank_jina_model_path),
        rerank_threads=getattr(cfg, "onnx_threads", 4),
        rerank_batch_size=cfg.rag_rerank_batch_size,
        rerank_max_length=cfg.rag_rerank_max_length,
        rerank_enabled=cfg.rag_rerank_enabled,
        matryoshka_enabled=cfg.rag_matryoshka_enabled,
        matryoshka_dim=cfg.rag_matryoshka_dim,
        vector_dtype=cfg.rag_vector_dtype,
        pdf_engine=cfg.rag_pdf_engine,
    )
    stats = engine.ingest(force=True)
    yield engine, stats
    engine.close()


def test_real_default_wiring_jina(rag_real):
    """默认流程应接到 jina 嵌入(MLX) + jina 重排(ONNX)。"""
    engine, _ = rag_real
    assert engine._embedding_backend == "mlx"
    assert "jina" in engine._embedding_model.lower()
    assert engine._rerank_backend == "onnx"
    assert "jina" in engine._rerank_model.lower()


def test_real_ingest_and_semantic_split(rag_real):
    """入库无错误；长文档(>800字)被语义切片拆成多片。"""
    engine, stats = rag_real
    assert "error" not in stats, stats.get("error")
    assert stats["files_found"] == len(DOCS)
    assert stats["chunks"] > 0
    assert not stats["errors"]
    # 短文档各 1 片、长文档 quantum.md 超阈值拆成多片 → 总切片数 > 文件数。
    assert stats["chunks"] > stats["files_found"], (
        f"语义切片未生效：应存在被拆分的长文档（总切片 {stats['chunks']} "
        f"应 > 文件数 {stats['files_found']}）")


def test_real_vector_store_128_int8(rag_real):
    """入库向量为 128 维、int8 量化（L2 重归一化后整体缩放至 ~127）。"""
    from agent.rag.vector_store import _TABLE_NAME

    engine, _ = rag_real
    import lancedb
    db = lancedb.connect(engine._vector_dir)
    tbl = db.open_table(_TABLE_NAME)

    vfield = next(f for f in tbl.schema if "vector" in f.name.lower())
    assert vfield.type.list_size == 128, f"向量维度应 128，实际 {vfield.type.list_size}"

    arrow = tbl.to_arrow()
    vec = np.asarray(arrow.column("vector")[0].as_py(), dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    # 入库前先做 128 维 L2 重归一化（单位向量），再做 int8 值量化会整体放大到
    # int8 区间（约 ×127.5），故存储范数≈127 而非 1——余弦相似度得以保留。
    steps = np.unique(np.round(vec / (2 / 255)).astype(int))
    assert len(vec) == 128
    assert 100 < norm < 140, f"int8 量化后存储范数应≈127，实际 {norm:.4f}"
    assert vec.max() <= 127 and vec.min() >= -127, "向量值应落在 int8 区间 [-127,127]"
    assert len(steps) > 10, f"应呈 int8 离散阶梯（>10 级），实际 {len(steps)} 级"


def test_real_search_quality(rag_real):
    """真实召回质量：各主题查询 top1 命中正确文档。"""
    engine, _ = rag_real
    cases = [
        ("苹果公司是谁创立的？", "apple.md"),
        ("什么是过拟合？", "ml_basics.md"),
        ("普洱茶属于什么茶类？", "tea_culture.md"),
        ("木星有哪些卫星？", "solar_system.md"),
        ("蛋白质的一级结构指什么？", "protein.md"),
        ("量子比特是什么？", "quantum.md"),
    ]
    for q, expect in cases:
        res = engine.search(q, top_k=3)
        assert res, f"查询无结果: {q}"
        top = os.path.basename(res[0]["source"])
        assert top == expect, f"查询 {q!r} 期望 {expect} 实际 {top}"


def test_real_search_irrelevant_no_crash(rag_real):
    """无关查询不编造、不崩溃，返回结构良好的结果。"""
    engine, _ = rag_real
    res = engine.search("核电站的工作原理是什么？", top_k=3)
    assert isinstance(res, list)
    for r in res:
        assert r.get("text")
        assert r.get("source")
        assert "score" in r


def test_real_status(rag_real):
    """status 正确反映知识库状态。"""
    engine, _ = rag_real
    status = engine.status()
    assert status["chunks_stored"] > 0
    assert len(status["sources"]) == len(DOCS)
    assert status["has_knowledge_base"] is True
