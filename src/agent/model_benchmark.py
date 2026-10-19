"""Repeatable local-model benchmarks used to compare selectable RAG backends.

The measured numbers are persisted so future backend-switching logic (and the
user) can consult them instead of re-guessing: 每次测速结果按 ``kind`` 覆盖写入
``~/.base-agent/model_benchmark.json``，UI 的「测速对比」按钮会展示建议。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

# 与 config.json 同级：~/.base-agent/model_benchmark.json
RESULTS_PATH = os.path.expanduser("~/.base-agent/model_benchmark.json")


def _timed(call: Callable[[], Any], rounds: int) -> tuple[List[float], Any]:
    durations: List[float] = []
    result = None
    for _ in range(max(1, rounds)):
        started = time.perf_counter()
        result = call()
        durations.append(time.perf_counter() - started)
    return durations, result


def _summary(durations: List[float], item_count: int) -> Dict[str, float]:
    mean = statistics.fmean(durations)
    return {
        "mean_seconds": mean,
        "min_seconds": min(durations),
        "items_per_second": item_count / mean if mean > 0 else 0.0,
    }


def benchmark_embeddings(
    *,
    mlx_path: str = "models/nomic-embed-text-v1.5-mlx",
    fastembed_model: str = "nomic-ai/nomic-embed-text-v1.5-Q",
    batch_size: int = 8,
    rounds: int = 3,
    threads: int = 4,
) -> Dict[str, Any]:
    import numpy as np

    from .rag.model_backends import MLXEmbeddingFunction
    from .rag.vector_store import FastEmbedEmbeddingFunction

    seeds = [
        "人工智能是研究如何让机器表现出智能行为的技术领域。",
        "苹果芯片的统一内存架构允许 CPU 与 GPU 共享数据。",
        "向量数据库通过近邻搜索召回语义相关的文档片段。",
        "ONNX Runtime can execute quantized transformer models on CPU.",
        "MLX is an array framework optimized for Apple silicon.",
        "Rerankers score a query and document jointly with a cross encoder.",
        "Python applications can load local models without an HTTP server.",
        "知识库检索通常结合向量相似度、关键词检索和重排序。",
    ]
    texts = [seeds[i % len(seeds)] + f" 样本 {i}" for i in range(max(1, batch_size))]

    mlx = MLXEmbeddingFunction(mlx_path, batch_size=batch_size, max_length=512)
    started = time.perf_counter()
    mlx_warm = mlx.embed_documents(texts)
    mlx_load = time.perf_counter() - started
    mlx_times, mlx_values = _timed(lambda: mlx.embed_documents(texts), rounds)

    fastembed = FastEmbedEmbeddingFunction(fastembed_model, threads=threads)
    started = time.perf_counter()
    fast_warm = fastembed.embed_documents(texts)
    fast_load = time.perf_counter() - started
    fast_times, fast_values = _timed(lambda: fastembed.embed_documents(texts), rounds)

    mlx_array = np.asarray(mlx_values or mlx_warm, dtype=np.float32)
    fast_array = np.asarray(fast_values or fast_warm, dtype=np.float32)
    mlx_array /= np.maximum(np.linalg.norm(mlx_array, axis=1, keepdims=True), 1e-12)
    fast_array /= np.maximum(np.linalg.norm(fast_array, axis=1, keepdims=True), 1e-12)
    alignment = np.sum(mlx_array * fast_array, axis=1)

    mlx_summary = _summary(mlx_times, len(texts))
    fast_summary = _summary(fast_times, len(texts))
    return {
        "kind": "embedding",
        "batch_size": len(texts),
        "rounds": rounds,
        "mlx": {"load_and_warmup_seconds": mlx_load, **mlx_summary},
        "fastembed_cpu": {"load_and_warmup_seconds": fast_load, **fast_summary},
        "speedup_mlx_over_fastembed": (
            mlx_summary["items_per_second"] / fast_summary["items_per_second"]
            if fast_summary["items_per_second"] > 0 else 0.0
        ),
        "mean_vector_cosine_alignment": float(alignment.mean()),
        "min_vector_cosine_alignment": float(alignment.min()),
    }


def benchmark_embedding_batch_sizes(
    *,
    mlx_path: str = "models/nomic-embed-text-v1.5-mlx",
    fastembed_model: str = "nomic-ai/nomic-embed-text-v1.5-Q",
    sizes=(2, 4, 6, 8, 10, 12, 16, 18, 20),
    corpus_size: int = 48,
    rounds: int = 3,
    threads: int = 4,
    text_chars: int = 0,
) -> Dict[str, Any]:
    """扫描子批数对 Embedding 吞吐的影响，用于确定批大小上下限。

    每档喂同一批语料（固定 corpus_size 条），按该档子批数切块后计时——
    小批的每调用开销和大批的 padding 浪费都会体现在最终 条/s 上，比只测
    「单批耗时」更接近真实建库/检索场景。
    """
    from .rag.model_backends import MLXEmbeddingFunction
    from .rag.vector_store import FastEmbedEmbeddingFunction

    seeds = [
        "人工智能是研究如何让机器表现出智能行为的技术领域。",
        "苹果芯片的统一内存架构允许 CPU 与 GPU 共享数据。",
        "向量数据库通过近邻搜索召回语义相关的文档片段。",
        "ONNX Runtime can execute quantized transformer models on CPU.",
        "MLX is an array framework optimized for Apple silicon.",
        "Rerankers score a query and document jointly with a cross encoder.",
        "知识库检索通常结合向量相似度、关键词检索和重排序。",
        "本地推理能够减少网络依赖并保护数据隐私。",
    ]
    base = [seeds[i % len(seeds)] + f" 样本 {i}" for i in range(max(1, corpus_size))]
    if text_chars > 0:
        # 长文本档：嵌入激活 ∝ 条数 × 批内最长序列²，真实长切片下形状预算
        # 会钳制子批条数，短文本语料测不出这个拐点。
        corpus = []
        for text in base:
            repeated = (text + " ")
            while len(repeated) < text_chars:
                repeated += text + " "
            corpus.append(repeated[:text_chars])
    else:
        corpus = base

    backends = [("mlx", lambda: MLXEmbeddingFunction(mlx_path, batch_size=8, max_length=512))]
    try:
        backends.append(
            ("fastembed_cpu", lambda: FastEmbedEmbeddingFunction(model_name=fastembed_model, threads=threads))
        )
    except Exception:  # pragma: no cover - fastembed 缺失时只测 MLX
        pass

    rows: List[Dict[str, Any]] = []
    for label, factory in backends:
        started = time.perf_counter()
        try:
            backend = factory()
        except Exception as e:
            rows.append({"backend": label, "error": f"{type(e).__name__}: {e}"})
            continue
        backend.embed_documents(corpus[:2])  # 预热（含权重加载）
        load = time.perf_counter() - started
        for size in sizes:
            durations, _ = _timed(
                lambda s=size: backend.embed_documents(corpus, batch_size=s), rounds
            )
            summary = _summary(durations, len(corpus))
            rows.append({
                "backend": label,
                "batch_size": int(size),
                "batches_per_round": math.ceil(len(corpus) / int(size)),
                "mean_seconds": summary["mean_seconds"],
                "min_seconds": summary["min_seconds"],
                "items_per_second": summary["items_per_second"],
                "ms_per_item": 1000 / summary["items_per_second"] if summary["items_per_second"] else 0.0,
            })
        rows.append({"backend": label, "load_and_warmup_seconds": load})
        closer = getattr(backend, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    return {
        "kind": "embedding_batch_sweep",
        "corpus_size": len(corpus),
        "text_chars": int(text_chars),
        "rounds": rounds,
        "sizes": [int(s) for s in sizes],
        "rows": rows,
    }


def _rerank_consistency(scores, reference, top_k: int) -> tuple:
    """Pearson correlation + top-k 排序重合度，用于判断「能否等价替换」。"""
    import numpy as np

    values = np.asarray(scores, dtype=np.float64)
    base = np.asarray(reference, dtype=np.float64)
    correlation = float(np.corrcoef(values, base)[0, 1]) if values.size > 1 else 1.0
    rank_a = np.argsort(-values)
    rank_b = np.argsort(-base)
    return correlation, len(set(rank_a[:top_k]) & set(rank_b[:top_k])) / top_k


def benchmark_rerankers(
    *,
    mlx_path: str = "models/bge-reranker-v2-m3-mlx-mxfp8",
    onnx_path: str = "models/bge-reranker-v2-m3-ONNX/model_quantized.onnx",
    fastembed_model: str = "BAAI/bge-reranker-v2-m3-int8",
    batch_size: int = 16,
    rounds: int = 3,
    threads: int = 4,
) -> Dict[str, Any]:
    """三方对比：MLX 直载 / ONNX 直载 / fastembed CPU。

    任一后端不可用（未装运行时或路径缺失）只记 error 不中断，保证其余两方
    仍能产出可比数据。
    """
    from .rag.model_backends import FastEmbedReranker, MLXReranker, ONNXReranker

    query = "如何在应用内直接加载本地模型，并避免网络 API 调用？"
    seeds = [
        "应用可以通过 MLX、ONNX Runtime 或 llama.cpp 直接加载本地模型权重。",
        "远程 API 需要通过 HTTP 请求服务端模型。",
        "本地推理能够减少网络依赖并保护数据隐私。",
        "今天的天气晴朗，适合在公园散步。",
        "量化模型通常能降低内存占用并提高 CPU 推理速度。",
        "数据库事务用于维护结构化数据的一致性。",
        "Transformers tokenizer 可以从 tokenizer.json 离线构造模型输入。",
        "苹果的 Metal GPU 可以加速兼容的 MLX 模型。",
    ]
    documents = [seeds[i % len(seeds)] + f" 样本 {i}" for i in range(max(1, batch_size))]

    def _measure(factory):
        started = time.perf_counter()
        try:
            backend = factory()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        warm = backend.rerank(query, documents, batch_size=batch_size)
        load = time.perf_counter() - started
        durations, scores = _timed(
            lambda: backend.rerank(query, documents, batch_size=batch_size), rounds
        )
        closer = getattr(backend, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
        return {
            "load_and_warmup_seconds": load,
            "scores": scores or warm,
            **_summary(durations, len(documents)),
        }

    mlx = _measure(lambda: MLXReranker(mlx_path, batch_size=batch_size, max_length=512))
    direct = _measure(lambda: ONNXReranker(onnx_path, threads=threads, max_length=512))
    fast = _measure(lambda: FastEmbedReranker(fastembed_model, threads=threads))

    result: Dict[str, Any] = {
        "kind": "reranker",
        "batch_size": len(documents),
        "rounds": rounds,
    }
    for key, measure in (("mlx", mlx), ("onnx_direct_cpu", direct), ("fastembed_cpu", fast)):
        result[key] = {k: v for k, v in measure.items() if k != "scores"}

    top_k = min(5, len(documents))
    reference = fast.get("scores")
    if reference is not None and "error" not in fast:
        fast_rate = fast.get("items_per_second", 0.0)
        for key, prefix in (("mlx", "mlx"), ("onnx_direct_cpu", "")):
            measure = mlx if key == "mlx" else direct
            if measure.get("scores") is None:
                continue
            correlation, overlap = _rerank_consistency(measure["scores"], reference, top_k)
            rate = measure.get("items_per_second", 0.0)
            speedup = rate / fast_rate if fast_rate > 0 else 0.0
            result[f"{prefix}_pearson_correlation" if prefix else "score_pearson_correlation"] = correlation
            result[f"{prefix}_top5_overlap" if prefix else "top5_overlap"] = overlap
            result[f"speedup_{prefix}_over_fastembed" if prefix else "speedup_direct_over_fastembed"] = speedup
            # 同时写回各方自己的小节，summarize / recommend 直接读即可，
            # 不必再靠 key 名反推。
            result[key]["speedup_vs_fastembed"] = speedup
            result[key]["pearson_correlation"] = correlation
            result[key]["top5_overlap"] = overlap
    return result


def load_results(path: str = RESULTS_PATH) -> List[Dict[str, Any]]:
    """Return previously stored benchmark results (empty list when absent)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def save_result(result: Dict[str, Any], path: str = RESULTS_PATH) -> str:
    """Persist one result per ``kind`` (embedding / reranker), atomically.

    同 kind 只保留最后一次：切换逻辑只需要「当前这台上跑出来最快的是谁」，
    历史序列会让读取方被迫做排序，且旧结果在换机/换模型后具有误导性。
    """
    kind = str(result.get("kind", ""))
    results = [item for item in load_results(path) if str(item.get("kind", "")) != kind]
    stamped = dict(result)
    stamped["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results.append(stamped)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def sweep_table(result: Dict[str, Any]) -> str:
    """Render a batch-size sweep as a readable table."""
    rows = [row for row in result.get("rows", []) if "batch_size" in row]
    if not rows:
        return "（无扫描数据）"
    backends = []
    for row in rows:
        if row["backend"] not in backends:
            backends.append(row["backend"])
    lines = [f"{'子批数':>6} " + " ".join(f"{b:>22}" for b in backends)]
    for size in result.get("sizes", []):
        cells = []
        for backend in backends:
            match = next(
                (r for r in rows if r["backend"] == backend and r["batch_size"] == size), None
            )
            cells.append(
                f"{match['items_per_second']:>10.1f} 条/s {match['ms_per_item']:>6.2f}ms" if match else " " * 22
            )
        lines.append(f"{size:>6} " + " ".join(cells))
    return "\n".join(lines)


def result_for(kind: str, path: str = RESULTS_PATH) -> Optional[Dict[str, Any]]:
    for item in load_results(path):
        if str(item.get("kind", "")) == kind:
            return item
    return None


def summarize(result: Dict[str, Any]) -> str:
    """One-line human readable summary of a benchmark result."""
    kind = str(result.get("kind", ""))
    if kind == "embedding":
        mlx = result.get("mlx", {})
        fast = result.get("fastembed_cpu", {})
        return (
            f"Embedding: MLX {mlx.get('items_per_second', 0):.1f} 条/s "
            f"(加载 {mlx.get('load_and_warmup_seconds', 0):.2f}s) vs "
            f"fastembed CPU {fast.get('items_per_second', 0):.1f} 条/s "
            f"(加载 {fast.get('load_and_warmup_seconds', 0):.2f}s) → "
            f"{result.get('speedup_mlx_over_fastembed', 0):.2f}x，"
            f"向量余弦 {result.get('mean_vector_cosine_alignment', 0):.3f}"
        )
    if kind == "reranker":
        fast = result.get("fastembed_cpu", {})
        parts = [
            f"fastembed CPU {fast.get('items_per_second', 0):.1f} 条/s "
            f"(加载 {fast.get('load_and_warmup_seconds', 0):.2f}s)"
        ]
        for key, label in (("mlx", "MLX"), ("onnx_direct_cpu", "ONNX")):
            item = result.get(key) or {}
            if not item:
                continue
            if item.get("error"):
                parts.append(f"{label} 不可用（{item['error'][:60]}）")
                continue
            parts.append(
                f"{label} {item.get('items_per_second', 0):.1f} 条/s "
                f"({item.get('speedup_vs_fastembed', 0):.2f}x, "
                f"加载 {item.get('load_and_warmup_seconds', 0):.2f}s, "
                f"相关 {item.get('pearson_correlation', 0):.3f})"
            )
        return "Reranker: " + " ｜ ".join(parts)
    return f"{kind}: {result}"


def recommend(result: Dict[str, Any]) -> str:
    """Suggest the backend the switching logic should prefer.

    判据是「速度 + 一致性」双门槛：只快但向量/分数不一致会让检索结果悄悄
    变差，属于必须人工回归的情形，所以单快时给出的是带条件而非直接建议。
    """
    kind = str(result.get("kind", ""))
    if kind == "embedding":
        speedup = float(result.get("speedup_mlx_over_fastembed", 0) or 0)
        alignment = float(result.get("min_vector_cosine_alignment", 0) or 0)
        if speedup >= 1.2 and alignment >= 0.9:
            return f"建议切换到 mlx（{speedup:.2f}x 更快，向量一致性 {alignment:.3f}）"
        if speedup >= 1.2:
            return (
                f"mlx 更快（{speedup:.2f}x），但向量一致性仅 {alignment:.3f}，"
                "切换前需回归检索质量"
            )
        return f"保持 fastembed（MLX 未体现速度优势，{speedup:.2f}x）"
    if kind == "reranker":
        fast_rate = float((result.get("fastembed_cpu") or {}).get("items_per_second", 0) or 0)
        # 一致性不达标（与 fastembed 排序不一致）的后端直接出局：换过去会静默
        # 改变召回顺序，比慢一点危险得多。
        candidates = []
        for key, label in (("mlx", "mlx"), ("onnx_direct_cpu", "onnx")):
            item = result.get(key) or {}
            if item.get("error") or not item.get("items_per_second"):
                continue
            if float(item.get("pearson_correlation", 0) or 0) < 0.95:
                continue
            candidates.append((float(item["items_per_second"]), label, item))
        if not candidates:
            return "保持 fastembed（本地后端不可用或分数一致性不足）"
        rate, label, item = max(candidates, key=lambda entry: entry[0])
        if fast_rate > 0 and rate >= fast_rate * 1.05:
            return (
                f"建议切换到 {label}（{rate / fast_rate:.2f}x 更快，"
                f"分数相关 {float(item['pearson_correlation']):.3f}）"
            )
        return (
            f"{label} 与 fastembed 速度相当（{rate / fast_rate:.2f}x）且分数相关 "
            f"{float(item['pearson_correlation']):.3f}，可作为免联网的等价本地方案"
            if fast_rate > 0 else f"建议切换到 {label}（本地直载，免联网）"
        )
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark selectable local model backends")
    parser.add_argument(
        "--kind", choices=("embedding", "reranker", "all", "sweep"), default="all"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--sizes", default="2,4,6,8,10,12,16,18,20",
        help="子批数扫描档位，逗号分隔（仅 --kind sweep）",
    )
    parser.add_argument(
        "--corpus", type=int, default=48,
        help="子批数扫描每档喂入的文本条数（仅 --kind sweep）",
    )
    parser.add_argument(
        "--text-chars", type=int, default=0,
        help="把每条语料填充到该字符数，用于测长切片下的形状预算拐点",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="把结果写入 ~/.base-agent/model_benchmark.json 作为切换逻辑参考",
    )
    parser.add_argument("--show", action="store_true", help="只打印已保存的测速结果与建议")
    args = parser.parse_args()

    if args.show:
        stored = load_results()
        if not stored:
            print(f"尚无测速记录: {RESULTS_PATH}")
            return
        for item in stored:
            print(f"[{item.get('measured_at', '?')}] {summarize(item)}\n  → {recommend(item)}")
        return

    results = []
    if args.kind == "sweep":
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
        sweep = benchmark_embedding_batch_sizes(
            sizes=sizes, corpus_size=args.corpus, rounds=args.rounds,
            threads=args.threads, text_chars=args.text_chars,
        )
        results.append(sweep)
        if args.save:
            save_result(sweep)
        print(sweep_table(sweep))
        return
    if args.kind in ("embedding", "all"):
        results.append(benchmark_embeddings(
            batch_size=args.batch_size or 8,
            rounds=args.rounds,
            threads=args.threads,
        ))
    if args.kind in ("reranker", "all"):
        results.append(benchmark_rerankers(
            batch_size=args.batch_size or 16,
            rounds=args.rounds,
            threads=args.threads,
        ))
    if args.save:
        for item in results:
            save_result(item)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    for item in results:
        print(f"{summarize(item)}\n  → {recommend(item)}")


if __name__ == "__main__":
    main()
