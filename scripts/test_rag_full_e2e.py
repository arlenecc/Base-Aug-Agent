#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 全流程端到端验证（真实本地模型：MLX 嵌入 + ONNX Reranker）。

覆盖：
  1. 多格式入库：md / txt / html / csv / docx（含 HTML 清洗、碎片过滤）。
  2. 真实语义嵌入（jina v5 nano MLX）+ 真实 Reranker（jina v2 ONNX）。
  3. 混合检索（向量 + BM25）+ Reranker 精排。
  4. 召回验证：每个主题的「金标准文档」必须出现在检索结果中（无遗漏）。
  5. 两段式 BQ 粗筛：关闭 / 开启两条路径，召回一致（开启不丢相关切片）；
     并直接验证 search_binary_coarse 机制本身可用、kb_binary 表已建且行数=切片数。

运行（仓库根目录）：
    PYTHONPATH=src python scripts/test_rag_full_e2e.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EMB = str(ROOT / "models" / "jina-embeddings-v5-text-nano-mlx-int4")
RER = str(ROOT / "models" / "jina-reranker-v2-base-multilingual-onnx")

# ── 测试文档（每个明显 > 100 字，避免触发碎片过滤；主题彼此区分）─────────────
DOCS = {
    "rag_intro.md": """# RAG 技术介绍

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的技术架构。
它通过在生成回答前先从外部知识库中检索相关文档片段，然后将检索结果作为上下文提供给大语言模型，
从而显著提高回答的准确性和时效性。

RAG 的核心组件包括：文档解析器负责从 PDF、Word、Markdown 等格式提取文本；文本切片器将长文档
分割成适合模型处理的短片段；向量化引擎将文本片段转换为向量表示；向量数据库负责存储和检索向量；
重排序器（Reranker）对初步检索结果进行精确排序，进一步提升相关性。相比传统的关键词搜索，
RAG 使用语义向量搜索，能够理解查询的深层含义。
""",
    "vector_db.txt": """向量数据库对比分析

向量数据库是专门用于存储和检索高维向量数据的数据库系统，在 AI 应用中扮演着关键角色。

1. ChromaDB：开源、轻量级，支持多种嵌入模型，内置持久化存储，适合中小规模应用。
2. FAISS：由 Facebook AI 开发，提供高性能向量检索，支持 GPU 加速，适合大规模检索场景。
3. Pinecone：云原生向量数据库，全托管服务，自动扩缩容，适合生产环境部署。
4. Weaviate：开源向量搜索引擎，支持 GraphQL 查询，内置多种向量化模块，适合复杂查询场景。

选择向量数据库时需要综合考虑数据规模、查询延迟、部署复杂度与成本等因素。不同场景下的取舍
差异很大，没有放之四海皆准的最优方案。
""",
    "best_practices.html": """<html><head><title>知识库最佳实践</title></head><body>
<header>导航栏（应被清洗掉）</header>
<nav>菜单（应被清洗掉）</nav>
<main>
<h1>知识库构建最佳实践</h1>
<p>构建高质量的知识库是 RAG 系统成功的关键。以下是一些经过验证的最佳实践：</p>
<h2>文本切片策略</h2>
<p>切片大小应根据模型上下文窗口合理设置。一般建议 500 token 一个切片，保留 10% 的重叠区域，
避免信息在切片边界处丢失。</p>
<h2>元数据管理</h2>
<p>为每个文档添加标题、来源、日期等元数据信息，方便检索时进行过滤和溯源。</p>
<h2>定期更新</h2>
<p>知识库需要定期更新，确保信息的时效性。建议设置自动同步机制，对新增和修改的文档增量处理。</p>
</main>
<footer>页脚信息（应被清洗掉）</footer>
<script>console.log("script should be removed")</script>
<style>body { color: red; }</style>
</body></html>
""",
    "embedding_models.csv": """模型名称,维度,语言支持,特点
bge-large-zh-v1.5,1024,中文,中文优化检索模型
text-embedding-3-small,1536,多语言,OpenAI 官方嵌入模型
m3e-base,768,中英文,中文社区流行的嵌入模型
jina-embeddings-v5-text-nano,512,多语言,体积小速度快的嵌入模型
all-MiniLM-L6-v2,384,多语言,轻量快速的句子嵌入模型
bge-m3,1024,多语言,支持多语言与多粒度检索
""",
}

# docx 内容（用 python-docx 生成）
DOCX_TEXT = """Token 与上下文窗口管理

Token 是大语言模型处理文本的基本单位。在英文中，一个 token 大约等于 3 到 4 个字符；
在中文中，一个 token 大约等于 1.5 到 2 个字符。Token 化方式直接影响上下文窗口的占用。

上下文窗口决定了模型在一次推理中能看到的的最大 token 数量。常见的上下文窗口大小有 4K、
8K、32K、128K 等。窗口越大，模型能同时处理的文档越多，但推理成本也越高。

Token 消耗优化建议：使用较短的系统提示词；合理设置切片大小避免过长的上下文；使用重排序
只保留最相关的结果；压缩历史对话移除冗余信息。这些手段能显著降低 Token 消耗与延迟。
"""

# 查询 -> 金标准文档（basename）
QUERIES = [
    ("什么是 RAG 检索增强生成 向量库检索 重排序", "rag_intro.md"),
    ("向量数据库有哪些 ChromaDB FAISS Pinecone Weaviate 对比", "vector_db.txt"),
    ("知识库构建最佳实践 切片大小 重叠 元数据 定期更新", "best_practices.html"),
    ("常见的嵌入模型有哪些 bge m3e text-embedding jina MiniLM", "embedding_models.csv"),
    ("Token 上下文窗口管理 消耗优化 切片大小", "knowledge_docx.docx"),
]


def _build_kb(kb_dir: str) -> None:
    os.makedirs(kb_dir, exist_ok=True)
    for name, text in DOCS.items():
        with open(os.path.join(kb_dir, name), "w", encoding="utf-8") as f:
            f.write(text)
    # docx
    try:
        from docx import Document
        doc = Document()
        for para in DOCX_TEXT.split("\n\n"):
            para = para.strip()
            if para:
                doc.add_paragraph(para)
        doc.save(os.path.join(kb_dir, "knowledge_docx.docx"))
    except Exception as e:
        raise RuntimeError(f"无法生成 docx 测试文档: {e}")


def _gold_recalled(results, gold_basename) -> int:
    """返回金标准文档在结果中的 rank（1-based），未出现返回 -1。"""
    for i, r in enumerate(results, 1):
        if os.path.basename(r.get("source", "")) == gold_basename:
            return i
    return -1


def _run_phase(label, binary_coarse):
    print(f"\n{'=' * 64}\n  阶段：{label}  (binary_coarse_enabled={binary_coarse})\n{'=' * 64}")
    ws = tempfile.mkdtemp(prefix="rag_e2e_")
    kb = os.path.join(ws, "kb")
    _build_kb(kb)

    from agent.rag.engine import RAGEngine
    eng = RAGEngine(
        workspace=ws,
        knowledge_base=kb,
        embedding_backend="mlx",
        embedding_model_path=EMB,
        rerank_backend="jina",
        rerank_jina_model_path=RER,
        binary_coarse_enabled=binary_coarse,
        rerank_enabled=True,
        pdf_engine="ovis",
    )
    stats = eng.ingest(force=True)
    print(f"  入库统计: files_found={stats['files_found']} extracted={stats['files_extracted']} "
          f"chunks={stats['chunks']} errors={len(stats['errors'])}")

    store = eng._get_store()
    total = store.count()
    print(f"  向量库切片总数: {total}")

    checks = []

    # 基础断言
    checks.append(("入库切片 > 0", total > 0))
    checks.append(("入库文件数 = 5", stats["files_extracted"] == 5))
    checks.append(("bing_coarse 标志正确", store._binary_coarse_enabled == binary_coarse))

    if binary_coarse:
        btbl = store._open_binary_table()
        bcount = btbl.count_rows() if btbl is not None else 0
        print(f"  kb_binary 表: {'存在' if btbl is not None else '缺失'} 行数={bcount}")
        checks.append(("kb_binary 表已建", btbl is not None))
        checks.append(("kb_binary 行数 == 切片数", bcount == total))

    # 召回验证：每个主题金标准文档必须出现在 top5
    recalled = {}
    for q, gold in QUERIES:
        res = eng.search(q, top_k=5)
        rank = _gold_recalled(res, gold)
        recalled[gold] = rank
        ok = rank > 0
        checks.append((f"召回[{gold}] query={q[:18]}…", ok))
        print(f"    Q: {q[:30]}… -> 金标准 {gold} rank={rank} "
              f"{'✅' if ok else '❌'}  (top1={os.path.basename(res[0]['source']) if res else '无'})")
        # 顺带确认 Reranker 已生效（结果经精排、分数在(0,1)且降序）
        if res:
            scores = [r.get("score", 0) for r in res]
            checks.append((f"Reranker 分数合法[{gold}]", all(0 < s <= 1 for s in scores)))
            checks.append((f"结果按分降序[{gold}]", scores == sorted(scores, reverse=True)))

    if binary_coarse:
        # 直接验证 BQ 粗筛机制本身
        for q, gold in QUERIES:
            bq = store.search_binary_coarse(q, refine_top_k=5)
            ok = bq is not None and _gold_recalled(bq, gold) > 0
            checks.append((f"BQ粗筛命中[{gold}]", ok))
            print(f"    BQ粗筛: {gold} -> {'✅' if ok else '❌'}")

        # Reranker 模型实际加载（对象非空即证明已加载并可用于精排）
        checks.append(("Reranker 模型已加载", store._reranker is not None))

    shutil.rmtree(ws, ignore_errors=True)
    return checks, recalled


def main() -> int:
    if not (os.path.isdir(EMB) and os.path.isdir(RER)):
        print("❌ 未找到本地模型：请确认 models/ 下存在 jina-embeddings-v5-text-nano-mlx-int4 "
              "与 jina-reranker-v2-base-multilingual-onnx")
        return 2

    all_checks = []
    phase_a, _ = _run_phase("BQ 关闭（默认原生检索路径）", binary_coarse=False)
    phase_b, _ = _run_phase("BQ 开启（两段式二进制粗筛）", binary_coarse=True)
    all_checks.extend(phase_a)
    all_checks.extend(phase_b)

    # 跨阶段召回一致性：两阶段金标准召回集合应完全相同（BQ 不丢相关切片）
    # （已在各阶段内逐条断言；此处汇总）
    print(f"\n{'=' * 64}\n  汇总：共 {len(all_checks)} 项检查\n{'=' * 64}")
    failed = 0
    for name, ok in all_checks:
        mark = "✅" if ok else "❌"
        if not ok:
            failed += 1
        print(f"  {mark} {name}")
    print(f"\n结果：{'全部通过 ✅' if failed == 0 else f'{failed} 项失败 ❌'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
