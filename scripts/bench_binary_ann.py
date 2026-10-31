"""对比实验：BIN 表的「ANN 粗筛」vs「二进制暴力汉明粗筛」在各数据量级的速度。

当前 VectorStore 的二段式粗筛是**内存暴力汉明扫描**（见 search_binary_coarse）：
把 kb_binary 全库载入内存，XOR+popcount 算汉明距离选 top-200。本脚本在此之外，
额外在 kb_binary.vector（int8/128 维）上建 IVF_PQ ANN 索引，用 ANN 选 top-200，
对比两条粗筛路径的**单查询选择耗时**与**召回**。

注意：ANN 索引只能建在向量列（int8），不能建在 16 字节 binary 列（lancedb 0.40
不支持对 binary 类型建 ANN）。所以「ANN 粗筛」本质是用全向量 ANN 代替二进制汉明
粗筛——这正是用来验证「BQ 二进制暴力粗筛是否已经够快 / 是否值得上 ANN」的实验。

用法：
  python scripts/bench_binary_ann.py --ns 5000,20000,50000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)  # 以便 import bench_binary_coarse

from agent.rag.vector_store import VectorStore, _hash_source, _INT8_SCALE  # noqa: E402
from bench_binary_coarse import TopicEmbeddingFunction, DIM, _BQ_TOP_K  # noqa: E402

logging.getLogger("agent").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.ERROR)

_OTHER_TOPICS = [
    "古希腊哲学", "文艺复兴艺术", "分子生物学", "分布式系统", "海洋地质",
    "认知心理学", "中世纪史", "弦理论", "城市规划", "食品化学",
    "歌剧艺术", "神经网络架构", "古埃及文明", "火山地质", "行为经济学",
    "气象学", "半导体物理", "古典音乐", "航天器设计", "热带雨林生态",
]


def build_corpus_shared(n_distractors, query_topics, chunks_per_doc=20):
    """与 bench_binary_coarse 类似，但每个文档的 chunk 共享 source，避免
    _sync_binary_for_records 对海量唯一 source 做删除（否则建库极慢）。
    """
    rng = np.random.default_rng(7)
    chunks = []
    gt_ids = []
    for t in query_topics:
        src = f"gt_{t}.md"
        for c in range(chunks_per_doc):
            text = f"{t}的核心概念与原理（第{c+1}节）：关于{t}的深入阐述与实例分析。"
            chunks.append({"text": text, "source": src, "chunk_index": c})
            gt_ids.append(f"{_hash_source(src)}_{c}")
    n_docs = max(1, n_distractors // chunks_per_doc)
    for d in range(n_docs):
        t = _OTHER_TOPICS[rng.integers(0, len(_OTHER_TOPICS))]
        src = f"doc_{d}.md"
        for c in range(chunks_per_doc):
            text = f"{t}领域的基础知识科普：介绍{t}的发展历程、关键方法与典型应用案例。"
            chunks.append({"text": text, "source": src, "chunk_index": c})
    rng.shuffle(np.array(chunks, dtype=object))
    return list(chunks), gt_ids


def _brute_selection(store, ef, q, db_bin, db_vec, scale):
    """复刻 search_binary_coarse 的选择阶段（汉明粗筛 + INT8 精筛），不含取文本。"""
    raw = ef.embed_query(q)
    qb = np.frombuffer(store._prepare_binary(raw), dtype=np.uint8)
    qv = store._prepare_query(raw).astype(np.float32)
    diff = np.count_nonzero(db_bin ^ qb[None, :], axis=1)
    order = np.argsort(diff).tolist()
    top = order[:_BQ_TOP_K]
    sub = db_vec[top].astype(np.float32)
    d2 = np.sum((sub - qv[None, :]) ** 2, axis=1)
    _ = np.argsort(-np.maximum(0.0, 1.0 - d2 / (2.0 * scale * scale)))[:20]
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5000,20000,50000", help="逗号分隔的数据量级")
    ap.add_argument("--queries", type=int, default=15)
    ap.add_argument("--cpd", type=int, default=20, help="每个文档的 chunk 数（共享 source）")
    args = ap.parse_args()
    Ns = [int(x) for x in args.ns.split(",")]

    query_topics = ["机器学习", "量子计算"] + [f"__t{i}__" for i in range(max(0, args.queries - 2))]
    ef = TopicEmbeddingFunction(dim=DIM, topics=query_topics + _OTHER_TOPICS, noise_std=0.02)

    print(f"对比：ANN 粗筛（kb_binary.vector 建 IVF_PQ） vs 二进制暴力汉明粗筛")
    print(f"{'N':>8} | {'建库s':>6} | {'暴力粗筛µs':>10} | {'ANN粗筛µs':>10} | "
          f"{'原生ANNµs':>10} | {'加速(BQ/ANN)':>12} | {'召回BQ':>6} | {'召回ANN':>7}")
    print("-" * 92)

    for N in Ns:
        chunks, gt_ids = build_corpus_shared(N, query_topics, chunks_per_doc=args.cpd)
        total = len(chunks)
        tmp = tempfile.mkdtemp(prefix="bqann_")
        t0 = time.time()
        store = VectorStore(persist_dir=tmp, embedding_function=ef, binary_coarse_enabled=True)
        store.add(chunks)
        build_s = time.time() - t0

        # 在 kb_binary.vector 上建 ANN 索引（IVF_PQ）
        btbl = store._open_binary_table()
        ann_ok = False
        ann_build_s = 0.0
        try:
            num_partitions = max(2, min(total // 100, 256))
            ta = time.time()
            btbl.create_index(
                metric="cosine", index_type="IVF_PQ",
                vector_column_name="vector",
                num_partitions=num_partitions, num_sub_vectors=16)
            ann_build_s = time.time() - ta
            ann_ok = True
        except Exception as e:
            print(f"  [WARN] N={total} ANN 建索引失败: {e}")

        # 预热
        cache = store._get_binary_cache()
        db_bin = cache["binaries"]
        db_vec = cache["vectors"]
        scale = float(_INT8_SCALE) if store._int8_store else 1.0
        queries = [f"{t}的核心概念是什么？" for t in query_topics]

        # 1) 暴力二进制汉明粗筛（选择阶段）
        t = time.time()
        for q in queries:
            _brute_selection(store, ef, q, db_bin, db_vec, scale)
        brute_us = (time.time() - t) / len(queries) * 1_000_000.0

        # 2) ANN 粗筛（选择阶段）
        if ann_ok:
            t = time.time()
            for q in queries:
                raw = ef.embed_query(q)
                qv = store._prepare_query(raw)
                qi = np.clip(np.round(qv * 127), -127, 127).astype(np.int8)
                _ = btbl.search(qi.tolist()).limit(_BQ_TOP_K).to_list()
            ann_us = (time.time() - t) / len(queries) * 1_000_000.0
        else:
            ann_us = float("nan")

        # 3) 原生 ANN（主表，关 BQ 的对照）
        store_native = VectorStore(persist_dir=tmp, embedding_function=ef, binary_coarse_enabled=False)
        t = time.time()
        for q in queries:
            store_native.search(q, top_k=_BQ_TOP_K)
        native_us = (time.time() - t) / len(queries) * 1_000_000.0

        # 召回：统计每个主题有多少 GT 切片进入 top-200（粗筛后还要经 INT8 精筛
        # 到 top-k，所以只要每个主题有足量 GT 切片进入 top-200 即视为不丢）。
        # 注意：12 主题×20 切片=240 个 GT 切片 > top-200，不可能“全部”进 top-200，
        # 故此处按“每主题至少 5 个”判定，与原生/ANN 的“任一在列”语义一致地反映
        # “真正需要的切片没被漏掉”。
        recall_bq = True
        min_cnt = 10**9
        for t_ in query_topics:
            raw = ef.embed_query(f"{t_}的核心概念是什么？")
            qb = np.frombuffer(store._prepare_binary(raw), dtype=np.uint8)
            diff = np.count_nonzero(db_bin ^ qb[None, :], axis=1)
            order = np.argsort(diff)
            top_ids = set(cache["ids"][i] for i in order[:_BQ_TOP_K])
            prefix = f"{_hash_source('gt_' + t_ + '.md')}_"
            cnt = sum(1 for gid in top_ids if gid.startswith(prefix))
            min_cnt = min(min_cnt, cnt)
            if cnt < 5:
                recall_bq = False
        if ann_ok:
            recall_ann = True
            for t_ in query_topics:
                raw = ef.embed_query(f"{t_}的核心概念是什么？")
                qv = store._prepare_query(raw)
                qi = np.clip(np.round(qv * 127), -127, 127).astype(np.int8)
                res = btbl.search(qi.tolist()).limit(_BQ_TOP_K).to_list()
                ids = {r.get("id") for r in res}
                prefix = f"{_hash_source('gt_' + t_ + '.md')}_"
                if not any(i.startswith(prefix) for i in ids):
                    recall_ann = False
        else:
            recall_ann = False

        speedup = (ann_us / brute_us) if ann_ok and ann_us == ann_us else float("nan")
        print(f"{total:>8} | {build_s:>5.1f}s | {brute_us:>9.1f} | "
              f"{ann_us if ann_ok else float('nan'):>10.1f} | {native_us:>9.1f} | "
              f"{speedup:>11.2f}× | {str(recall_bq):>6} | {str(recall_ann):>7}")

    print("\n说明：暴力粗筛=内存汉明扫描(全库O(N)但常数极小)；ANN粗筛=IVF_PQ索引查询；"
          "原生ANN=主表 self.search 对照。三者选出的都是 top-200 候选，精筛成本相同。")


if __name__ == "__main__":
    main()
