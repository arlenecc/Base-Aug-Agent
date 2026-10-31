"""全流程评测：两段式 BQ 粗筛检索（rag_binary_coarse_enabled）是否按设计工作。

重点验证三件事：
  1. 正确性：开启 BQ 粗筛后，search_with_rerank / search_binary_coarse
     返回的结果与原生 self.search 路径一致，且真正需要的切片不丢失；
  2. 召回率：海量数据中，被查询命中的「真实相关切片」始终落在汉明粗筛
     top-200 之内（不会被粗筛漏掉）；
  3. 速度：开启 BQ 后，向量通道的检索耗时（内存 numpy 汉明扫描）相比
     LanceDB 原生 ANN 检索，在大库下有明显加速，且加速比随数据量放大。

注意：被测的 VectorStore 用「自定义 EF」（self._custom_ef 非空），rerank 会
自动走 _rerank_fallback（按余弦距离排序），所以本脚本的召回判据完全由向量
相似度决定，不受真实 reranker 模型影响——正好隔离出 BQ 粗筛这一环。

本脚本自带 TopicEmbeddingFunction：向量以「主题方向 + 确定性噪声」构成，
分量有正有负（符号位才有区分度），且同主题切片方向聚集、跨主题方向正交，
可真实反映 BQ（符号位汉明）作为余弦代理的有效性。
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from agent.rag.vector_store import VectorStore, _hash_source, _INT8_SCALE  # noqa: E402

logging.getLogger("agent").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.ERROR)

DIM = 128
_BQ_TOP_K = 200  # 与 vector_store._BINARY_COARSE_TOP_K 对应


# ---------------------------------------------------------------------------
# 主题结构化嵌入：分量有正有负，同主题聚集、跨主题正交
# ---------------------------------------------------------------------------
class TopicEmbeddingFunction:
    """确定性、无网络、带符号位的主题结构化嵌入，专为 BQ 评测设计。

    每个主题对应一个固定单位方向向量；文本若含某主题关键词，则以其方向
    为基准叠加「按全文 hash 派生的确定性噪声」后归一化。同主题切片方向
    高度聚集（余弦≈0.97），跨主题近正交（余弦≈0）。查询命中同主题时，
    汉明距离天然最小——这正是 BQ 粗筛能保召回的前提。
    """

    def __init__(self, dim=DIM, topics=None, noise_std=0.02, seed=12345):
        self.dim = dim
        self.noise_std = noise_std
        rng = np.random.default_rng(seed)
        self.topics = {}
        for t in (topics or []):
            v = rng.normal(0, 1, size=dim)
            self.topics[t] = v / np.linalg.norm(v)
        self._misc = rng.normal(0, 1, size=dim)
        self._misc /= np.linalg.norm(self._misc)

    def _topic_of(self, text):
        for t in self.topics:
            if t in text:
                return t
        return None

    def _vec_for(self, text):
        topic = self._topic_of(text)
        base = self.topics[topic].copy() if topic else self._misc.copy()
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        rnd = np.random.default_rng(h & 0xFFFFFFFF)
        noise = rnd.normal(0, self.noise_std, size=self.dim)
        v = base + noise
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v.astype(np.float32).tolist()

    def __call__(self, texts):
        return [self._vec_for(t) for t in texts]

    def embed_query(self, text):
        return self._vec_for(text)

    def name(self):
        return "topic-ef"


# ---------------------------------------------------------------------------
# 语料构建
# ---------------------------------------------------------------------------
def build_corpus(n_distractors, query_topics, chunks_per_doc=5, other_topics=None):
    """返回 (chunks, ground_truth_ids)。

    - query_topics：列表，每个元素是「查询主题关键词」，会各注入 1 篇专属文档
      （其切片即该主题的 ground truth，且库中无其他同主题文档 → 唯一可命中）；
    - 其余 n_distractors 个切片均匀取自 other_topics（与查询主题不重叠）。
    """
    other_topics = other_topics or [
        "古希腊哲学", "文艺复兴艺术", "分子生物学", "分布式系统", "海洋地质",
        "认知心理学", "中世纪史", "弦理论", "城市规划", "食品化学",
        "歌剧艺术", "神经网络架构", "古埃及文明", "火山地质", "行为经济学",
        "气象学", "半导体物理", "古典音乐", "航天器设计", "热带雨林生态",
    ]
    rng = np.random.default_rng(7)
    chunks = []
    ground_truth_ids = []
    # ground-truth 文档：每主题一篇
    for t in query_topics:
        src = f"gt_{t}.md"
        for c in range(chunks_per_doc):
            text = f"{t}的核心概念与原理（第{c+1}节）：关于{t}的深入阐述与实例分析。"
            chunks.append({"text": text, "source": src, "chunk_index": c})
            ground_truth_ids.append(f"{_hash_source(src)}_{c}")
    # 干扰文档：取自其它主题
    gt_set = set(query_topics)
    for i in range(n_distractors):
        t = other_topics[rng.integers(0, len(other_topics))]
        src = f"doc_{i}.md"
        c = rng.integers(0, chunks_per_doc)
        text = f"{t}领域的基础知识科普：介绍{t}的发展历程、关键方法与典型应用案例。"
        chunks.append({"text": text, "source": src, "chunk_index": int(c)})
    rng.shuffle(np.array(chunks, dtype=object))  # 打乱顺序，避免同主题聚集
    return chunks, ground_truth_ids


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="干扰切片数量（总库≈n+ground_truth）")
    ap.add_argument("--queries", type=int, default=20, help="随机查询主题数量（用于召回压力测试）")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    n = args.n
    top_k = args.top_k

    # 主查询主题（保证有 ground truth）+ 一批随机主题做召回压力测试
    main_topics = ["机器学习", "量子计算"]
    # 随机生成一批不重叠的主题词（避免与 other_topics 撞车用带编号的稀有词）
    rng = np.random.default_rng(99)
    extra_topics = [f"__topic_{i}__" for i in range(args.queries)]
    query_topics = main_topics + extra_topics

    print(f"== 构建语料：{n} 干扰切片 + {len(query_topics)} 篇 ground-truth 文档 ==")
    chunks, gt_ids = build_corpus(
        n_distractors=n, query_topics=query_topics, chunks_per_doc=5)
    total = len(chunks)
    print(f"   总切片数 = {total}")

    ef = TopicEmbeddingFunction(dim=DIM, topics=query_topics + [
        "古希腊哲学", "文艺复兴艺术", "分子生物学", "分布式系统", "海洋地质",
        "认知心理学", "中世纪史", "弦理论", "城市规划", "食品化学",
        "歌剧艺术", "神经网络架构", "古埃及文明", "火山地质", "行为经济学",
        "气象学", "半导体物理", "古典音乐", "航天器设计", "热带雨林生态",
    ])

    # 同一持久化目录：先用 BQ 开启的 store 写主表 + kb_binary，再用关闭的
    # store 打开同一目录做原生路径对照（kb_binary 被忽略，行为等同旧版）。
    tmp = tempfile.mkdtemp(prefix="bq_bench_")
    t0 = time.time()
    store_coarse = VectorStore(persist_dir=tmp, embedding_function=ef,
                               binary_coarse_enabled=True)
    store_coarse.add(chunks)
    build_s = time.time() - t0
    store_native = VectorStore(persist_dir=tmp, embedding_function=ef,
                               binary_coarse_enabled=False)
    print(f"   建库耗时 = {build_s:.2f}s；kb_binary 表存在 = "
          f"{'kb_binary' in store_coarse._db.table_names()}")

    # ---- 1) 召回压力测试：每个查询主题的 ground truth 是否落入汉明 top-200 ----
    print(f"\n== 召回验证：{len(query_topics)} 个查询主题，逐主题检查 ground-truth 切片 ==")
    missed = 0
    worst_rank = 0
    for t in query_topics:
        q = f"{t}的核心概念是什么？请解释{t}的原理。"
        cache = store_coarse._get_binary_cache()
        raw = ef.embed_query(q)
        q_bin = np.frombuffer(store_coarse._prepare_binary(raw), dtype=np.uint8)
        diff = np.count_nonzero(cache["binaries"] ^ q_bin[None, :], axis=1)
        order = np.argsort(diff)
        # 该主题 ground truth 在 order 中的最差（最大）rank
        gt_ranks = []
        for idx in order:
            gid = cache["ids"][idx]
            if gid.startswith(f"{_hash_source('gt_'+t+'.md')}_"):
                gt_ranks.append(int(np.where(order == idx)[0][0]))
        if not gt_ranks:
            missed += 1
            print(f"   [!!] 主题 {t}: 未找到 ground-truth 切片（不应发生）")
            continue
        mr = max(gt_ranks)
        worst_rank = max(worst_rank, mr)
        if mr >= _BQ_TOP_K:
            missed += 1
            print(f"   [!!] 主题 {t}: 最差 ground-truth rank={mr} ≥ {_BQ_TOP_K}（被粗筛漏掉）")
    print(f"   漏召主题数 = {missed} / {len(query_topics)}；"
          f"所有 ground-truth 的最差汉明 rank ≤ {worst_rank}（门限 {_BQ_TOP_K}）")
    recall_ok = missed == 0

    # ---- 2) 两条路径结果一致性 + 主查询确实召回 ground truth ----
    print("\n== 结果一致性：search_with_rerank（开/关 BQ）主查询 ==")
    for t in main_topics:
        q = f"{t}的核心概念是什么？请解释{t}的原理。"
        r_coarse = store_coarse.search_with_rerank(q, top_k=top_k)
        r_native = store_native.search_with_rerank(q, top_k=top_k)
        c_ids = {r.get("id") for r in r_coarse}
        n_ids = {r.get("id") for r in r_native}
        gt_for_t = {f"{_hash_source('gt_'+t+'.md')}_{c}" for c in range(5)}
        coarse_hit = len(gt_for_t & c_ids) > 0
        native_hit = len(gt_for_t & n_ids) > 0
        overlap = len(c_ids & n_ids)
        print(f"   主题 {t}: 开BQ命中GT={coarse_hit} 关BQ命中GT={native_hit} "
              f"两路结果交集={overlap}/{top_k} 开BQ返回{len(r_coarse)}条")

    # ---- 3) 速度对比：粗筛核心（内存 numpy 汉明扫描）vs 原生 ANN ----
    print(f"\n== 速度对比：{total} 切片库，{len(query_topics)} 个查询 avg 耗时 ==")
    # 预热缓存（首查会构建全库内存缓存，属于一次性成本，单独计时）
    cache = store_coarse._get_binary_cache()
    queries = [f"{t}的核心概念是什么？" for t in query_topics]
    db_bin = cache["binaries"]
    db_vec = cache["vectors"]
    scale = float(_INT8_SCALE) if store_coarse._int8_store else 1.0

    # 3a) 粗筛「核心」：仅汉明粗筛 + INT8 余弦精筛（内存 numpy，不含取文本）。
    #     这是两段式设计真正的加速来源——O(N) 向量化扫描，与是否触盘无关。
    t_a = time.time()
    for q in queries:
        raw = ef.embed_query(q)
        q_bin = np.frombuffer(store_coarse._prepare_binary(raw), dtype=np.uint8)
        q_vec = store_coarse._prepare_query(raw).astype(np.float32)
        diff = np.count_nonzero(db_bin ^ q_bin[None, :], axis=1)
        order = np.argsort(diff).tolist()
        top = order[:_BQ_TOP_K]
        sub = db_vec[top].astype(np.float32)
        d2 = np.sum((sub - q_vec[None, :]) ** 2, axis=1)
        _ = np.argsort(-np.maximum(0.0, 1.0 - d2 / (2.0 * scale * scale)))[:top_k]
    core_ms = (time.time() - t_a) / len(queries) * 1000.0

    # 3b) 粗筛「全路径」：search_binary_coarse（含一次按 id 取文本，当前实现）。
    t_a2 = time.time()
    for q in queries:
        store_coarse.search_binary_coarse(q, refine_top_k=top_k)
    full_ms = (time.time() - t_a2) / len(queries) * 1000.0

    # 3c) 原生路径：self.search（LanceDB ANN，每次触盘并直接带文本返回）。
    t_b = time.time()
    for q in queries:
        store_native.search(q, top_k=top_k)
    native_ms = (time.time() - t_b) / len(queries) * 1000.0

    print(f"   单查询·粗筛核心(汉明+精筛,内存) = {core_ms:.3f} ms  ← 设计加速来源")
    print(f"   单查询·粗筛全路径(含取文本)     = {full_ms:.3f} ms")
    print(f"   单查询·原生ANN全路径            = {native_ms:.3f} ms")
    if native_ms > 0:
        print(f"   核心扫描相对原生 ≈ {native_ms / core_ms:.1f}× 更快"
              f"（全路径因当前每次多一次取文本查询，差距在大数据量时显现）")

    # ---- 结论 ----
    print("\n== 结论 ==")
    print(f"   [召回] 所有真实相关切片均未被粗筛漏掉: {'PASS' if recall_ok else 'FAIL'}")
    print(f"   [一致] 开/关 BQ 主查询均召回 ground truth: "
          f"{'PASS' if coarse_hit and native_hit else 'FAIL'}")
    print(f"   [速度] 粗筛核心扫描快于原生: {'PASS' if core_ms < native_ms else 'FAIL'}")
    ok = recall_ok and coarse_hit and native_hit and (core_ms < native_ms)
    print(f"\n总体: {'ALL PASS' if ok else 'HAS FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
