"""Vector store — embedding + similarity search using LanceDB, with BGE reranker.

Embeddings are generated locally via FastEmbed (ONNX Runtime) using
nomic-ai/nomic-embed-text-v1.5 — no API server or HuggingFace dependency.
Vectors are stored in LanceDB (Lance columnar format on local disk).
"""
from __future__ import annotations

import functools
import logging
import math
import os
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default embedding model — quantized nomic-embed-text-v1.5 via FastEmbed (ONNX Runtime).
# Using the -Q variant: model_quantized.onnx (~137MB) instead of model.onnx (~548MB).
# Same 768-dim embeddings, same quality, 4x smaller download.
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
# 重排序走 fastembed TextCrossEncoder（ONNX，无 PyTorch 依赖）。选官方 INT8
# 量化版：~570MB（fp32 全量 2.3GB 的一半不到），且 v2-m3 的多语种/中文精排
# 能力优于 bge-reranker-base。旧配置里的 bge-reranker-base 仍被 fastembed
# 支持（fp32 onnx ~1.1GB），无需迁移。
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"
_TABLE_NAME = "knowledge_base"

# 文档元数据表：存储每个文档的缩略版本（目录结构 + 章节标题）与完整 Markdown，
# 用于 Meta-context + Targeted RAG（先返回全文档结构概览，再按需检索细节）。
_DOCUMENTS_TABLE_NAME = "documents"

# nomic-embed-text-v1.5 requires query/document prefixes for best results
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

# BM25 全文检索的预分词列名（jieba 分词后空格连接，供原生 FTS 的 simple
# tokenizer 按空格切分索引，解决中文无空格无法分词的问题）。
_FTS_COLUMN = "text_fts"

# 入库时间戳列（epoch 秒）：向量去重按它「保留入库最早的一份」。旧表没有
# 这一列，schema 迁移时回填 0.0（视为最老——历史数据天然占先，不会被后来
# 重新入库的副本顶掉）。
_INGESTED_AT = "ingested_at"

# jieba 分词器缓存（进程级单例，避免每次调用重复初始化词典）。
_jieba = None
_jieba_lock = threading.Lock()
_jieba_tried = False


def _get_jieba():
    """Lazily import and return jieba, or None if unavailable."""
    global _jieba, _jieba_tried
    if _jieba_tried:
        return _jieba
    with _jieba_lock:
        if _jieba_tried:
            return _jieba
        _jieba_tried = True
        try:
            import jieba
            _jieba = jieba
            logger.debug("jieba 分词器已加载（用于 BM25 中文全文检索）")
        except ImportError:
            _jieba = None
            logger.info("jieba 未安装，BM25 中文分词退化为空白切分")
    return _jieba


def _tokenize_for_fts(text: str) -> str:
    """Tokenize text for the BM25 full-text index.

    Uses jieba for CJK word segmentation, then joins tokens with spaces so the
    native FTS ``simple`` tokenizer can index each word.  English words are
    kept intact (jieba segments them on whitespace/punctuation).  When jieba
    is unavailable, returns the original text (English still works via
    whitespace; CJK degrades to whole-sentence tokens).
    """
    jieba = _get_jieba()
    if jieba is None or not text:
        return text or ""
    tokens = [t.strip() for t in jieba.cut(text) if t.strip()]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# FastEmbed-based embedding function (local ONNX Runtime, no API needed)
# ---------------------------------------------------------------------------


class FastEmbedEmbeddingFunction:
    """Embedding function using FastEmbed (ONNX Runtime) for local inference.

    Loads nomic-ai/nomic-embed-text-v1.5 via ONNX Runtime — no GPU, no
    HuggingFace network calls, no PyTorch dependency. The model is cached
    after first download (~130MB quantized).

    Thread-safety: ONNX Runtime InferenceSession.run() is thread-safe,
    so a single instance can be shared across RAG, graph_memory, etc.
    Use get_or_create_embedding_function() to get a shared singleton.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, threads: int = 4):
        self._model_name = model_name
        self._threads = max(1, int(threads))
        self._model = None  # lazy-init
        # 懒加载的双检锁：InferenceSession.run() 的线程安全只覆盖推理，不覆盖
        # 这里的懒初始化——两个线程同时首次调用会各自加载一份 ~130MB 模型，
        # 后写覆盖先写，先建的那份成为纯泄漏。
        self._model_lock = threading.Lock()
        # 跨模块共享同一 ONNX session（RAG 入库 + graph_memory + skill_retriever
        # 同时飞）。fastembed 的 embed() 自管线程池复用 session，并发调用在个别
        # 版本下偶发 Ort::Exception / 结果错位。用一把锁串行所有 embed 调用，
        # 代价仅是跨模块并发退化为顺序，但消除了真实竞态。
        self._embed_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            from fastembed import TextEmbedding
            logger.info("  🔧 正在加载嵌入模型: %s (首次加载, 约 130MB)...", self._model_name)
            import time as _time
            _time.sleep(0.01)  # 释放 GIL 让 UI 更新
            try:
                # Use a persistent cache dir instead of the default system
                # temp dir (/var/folders/.../T/) which macOS cleans regularly.
                # This prevents the model from being re-downloaded after
                # every system cleanup.
                cache_dir = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "fastembed",
                )
                os.makedirs(cache_dir, exist_ok=True)
                # Suppress FastEmbed's "model has been updated on HuggingFace"
                # UserWarning — it fires on every instantiation and is not
                # actionable for the user.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self._model = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=cache_dir,
                        # 显式锁死 CPU EP（不依赖 fastembed 的默认行为——它
                        # 只自动选 CUDA、从不选 CoreML，行为变更时会静默漂移）。
                        # nomic 走 CoreML EP 已实测失败（41% 节点不支持、
                        # 分桶小批次复现 SIGKILL×3），CPU EP + 长度分桶才是
                        # 本机最优路径（#126 验证依据之一），见项目记忆「失败路」。
                        providers=["CPUExecutionProvider"],
                        # ORT intra-op 线程数：默认值在本机实测与 4 相当
                        # （57 vs 63 tok/s），留一半核心给 OCR/解析 worker。
                        # 嵌入吞吐的主要杠杆是长度分桶（见 _embed_chunks），
                        # 线程数影响很小（#122）。
                        threads=self._threads,
                    )
            except Exception as e:
                # 模型加载失败最常见的原因是首次使用时需要从 HuggingFace
                # 下载 ONNX 模型文件（约 130MB），但网络无法访问 huggingface.co。
                # 给用户提供明确的错误信息和解决方案。
                msg = (
                    f"无法加载嵌入模型 '{self._model_name}'：{e}\n\n"
                    f"可能原因及解决方案：\n"
                    f"1. 首次使用需从 HuggingFace 下载模型（~130MB），网络无法访问\n"
                    f"   → 设置镜像站: export HF_ENDPOINT=https://hf-mirror.com\n"
                    f"2. 新版 huggingface_hub 默认使用 Xet 传输，镜像站可能不支持\n"
                    f"   → 禁用 Xet: export HF_HUB_DISABLE_XET=1\n"
                    f"3. 模型缓存目录权限不足\n"
                    f"   → 检查 ~/.cache/huggingface/ 目录权限\n"
                    f"4. 也可手动下载模型到本地缓存：\n"
                    f"   HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \\\n"
                    f"     python3 -c \"from huggingface_hub import snapshot_download; \\\n"
                    f"     snapshot_download('nomic-ai/nomic-embed-text-v1.5')\"\n\n"
                    f"原始错误: {type(e).__name__}: {e}"
                )
                logger.error(msg)
                raise RuntimeError(msg) from e
            logger.info("  ✅ 嵌入模型加载完成: %s", self._model_name)
            _time.sleep(0.01)  # 释放 GIL 让 UI 更新

    def __call__(self, input: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Embed a list of document texts. Returns a list of embedding vectors."""
        if not input:
            return []
        self._ensure_model()
        # Add document prefix for nomic-embed-text-v1.5
        prefixed = [_DOC_PREFIX + t if not t.startswith(_DOC_PREFIX) else t for t in input]
        if batch_size:
            with self._embed_lock:
                embeddings = list(self._model.embed(prefixed, batch_size=int(batch_size)))
        else:
            with self._embed_lock:
                embeddings = list(self._model.embed(prefixed))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text with the query prefix."""
        self._ensure_model()
        prefixed = _QUERY_PREFIX + text if not text.startswith(_QUERY_PREFIX) else text
        with self._embed_lock:
            embeddings = list(self._model.embed([prefixed]))
        return embeddings[0].tolist()

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Embed a list of document texts (with document prefix).

        Batched: a single ONNX forward pass for the whole list, much faster
        than calling embed_query() per text.
        """
        return self(texts, batch_size=batch_size)

    def name(self) -> str:
        return f"fastembed:{self._model_name}"


# ---------------------------------------------------------------------------
# Process-level embedding function singleton
# ---------------------------------------------------------------------------

_EF_SINGLETONS: Dict[str, Any] = {}
_EF_SINGLETONS_LOCK = threading.Lock()


def _store_op(func):
    """装饰 VectorStore 的公开方法：登记在途操作，与 clear()/close() 互斥。

    不装饰内部方法（``_*``）：它们由已登记的操作调用，重复计数会让
    clear() 永远等不到排空（自己等自己）。
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self._begin_op()
        try:
            return func(self, *args, **kwargs)
        finally:
            self._end_op()

    return wrapper


def _new_embedding_function(
    model_name: str,
    *,
    backend: str,
    model_path: str,
    threads: int,
    batch_size: int,
    max_length: int,
):
    """Create a *new* (non-cached) embedding adapter for the selected backend.

    与 ``get_or_create_embedding_function`` 的区别：永远新建、不做单例缓存。
    供需要独立实例的调用方使用（如测试注入）。
    """
    backend = (backend or "fastembed").lower()
    if backend == "fastembed":
        return FastEmbedEmbeddingFunction(model_name=model_name, threads=threads)
    from .model_backends import create_embedding_backend
    return create_embedding_backend(
        backend,
        model_name,
        model_path,
        threads=threads,
        batch_size=batch_size,
        max_length=max_length,
    )


def get_or_create_embedding_function(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    backend: str = "fastembed",
    model_path: str = "",
    threads: int = 4,
    batch_size: int = 8,
    max_length: int = 8192,
):
    """Return a shared embedding adapter for the selected local backend.

    单实例适配器（与旧版一致）。Embedding 后端（MLX / fastembed / ONNX /
    llama_cpp）内部已各自并行（MLX 走全局计算流；fastembed/ONNX 用 intra-op
    线程），多实例并发不仅无收益（实测 MLX 仅 ~1.04x、fastembed 还因批次依赖
    破坏嵌入一致性），反而多占模型权重内存，故不提供多实例路径。
    """
    backend = (backend or "fastembed").lower()
    key = "|".join((
        backend,
        model_name or "",
        os.path.abspath(os.path.expanduser(model_path)) if model_path else "",
        str(max(1, int(threads))),
        str(max(1, int(batch_size))),
        str(max(8, int(max_length))),
    ))
    with _EF_SINGLETONS_LOCK:
        ef = _EF_SINGLETONS.get(key)
        if ef is not None:
            return ef
        ef = _new_embedding_function(
            model_name,
            backend=backend,
            model_path=model_path,
            threads=threads,
            batch_size=batch_size,
            max_length=max_length,
        )
        _EF_SINGLETONS[key] = ef
        return ef


# ---------------------------------------------------------------------------
# 自适应嵌入子批大小控制器
# ---------------------------------------------------------------------------

# 子批大小探索范围与步长（#125 引入，#126-#129 生产验证修订，2026-10 MLX
# 默认后端后按新实测重新标定）。
#
# 旧标定「吞吐随子批增大单调变差、size2 双优」（size2 = 2071 tok/s @
# 1.28GB，size12 = 1163 @ 4.99GB）是 **fastembed/ONNX CPU** 时代的结论。
# 换成 MLX 直载后曲线反转：短切片 2→16 吞吐从 88 涨到 178 条/s（+102%），
# 16→32 再 +5~8%，32 以上基本持平、128 反而回落；长切片（~1200 字符）
# 则与子批数无关（13.2 条/s 从 1 到 16 平坦）。因此：
#   - 起步 16：直接落在 MLX 的拐点上，省掉从 2 一路爬 24 档的试探开销；
#   - 上限 64：32-96 已无收益，64 是收益与安全的水位（未测更大批的内存）；
#   - 下限 2：低于 4 吞吐掉到最优的 50-69%，但保留 2 作为极端内存压力下的
#     避让位（形状预算同样以此为止，见 _shape_cap_size，不再夹到 1）。
# 步长 2：范围扩大到 [2,64] 后步长 1 要爬 31 档才到顶，爬山期过长。
# 自适应子批范围（2026-10-26 按 bge-small-zh @500 字切片实测重标定，
# 终值 64/6/16/2 由用户拍板）：初测以为 6-8 是平台顶（~97 条/s），但三轮
# 交错复测显示小尺寸（1-8）极差达 15-39% 且第 1 轮系统性偏快 ~15%，
# 该"平台顶"是噪声；均值口径 6-64 全部平坦（78-85 条/s），size≥16 极差
# <9% 才可信。故 START=16（测量最稳的平坦区起点），MAX=64（无收益上限，
# +2.5% 在抖动范围内），MIN=6（低于 6 单条吞吐塌陷、固定成本摊不薄，
# 见 #139），STEP=2。调参器爬坡阈值 3% 应与 ≥16 段的方差量级比较。
_SUBBATCH_MIN = 6
_SUBBATCH_MAX = 64
_SUBBATCH_START = 16
_SUBBATCH_STEP = 2
# RSS 软墙（#129，用户定参 8GB）：内存偶尔冲高是正常的（OCR 缓存、压缩
# 计费），只要求「超墙时降档避让、回墙下即停」——无闩锁。旧 5GB 闩锁版
# 会在真实同步基线（6-10GB）下首个子批就触发且永不解除，是「降到 1 再
# 也不回去」的根因之一。注意：wall 只影响运行中的进程，改这里不会让
# 已启动的应用生效，需要完全重启。
_SUBBATCH_RSS_LIMIT = 8.0  # GB

# 逐子批内存打点文件（与 engine 的 RAG_MEM_LOG 同一开关）：footprint 口径。
_MEM_LOG_PATH = os.environ.get("RAG_MEM_LOG") or ""

# MLX 后端缓存清理间隔（每 N 个嵌入子批一次）。
#
# 为什么必须显式清理：默认嵌入后端就是 mlx（config.py: rag_embedding_backend
# = "mlx"），而 mx.clear_cache() 原本只存在于 MLXEmbeddingFunction.close()
# 与 MLXReranker.close() 两处。而共享单例 EF 根本走不到 close()——
# VectorStore.close() 里 `is_shared` 为真时整段跳过释放，只做 `self._ef = None`
# （共享 EF 不能 close，否则会砸掉 graph_memory / 其它 VectorStore）。
# 于是整轮同步（数千子批）mx.clear_cache() 一次都不执行：子批大小在 [2,64]
# 间不断变化（_SubBatchTuner + _shape_cap_size），MLX 按形状缓存 buffer/graph，
# 缓存在同步期间单调增长 → 系统内存压力 → 「同步跑一会就崩」。
# 每 N 个子批清一次即可把缓存压在一代内，清掉的只是不再引用的缓存数组，
# 不影响仍在使用的模型权重。
_MLX_CACHE_CLEAR_EVERY = 16
# 恢复阈值（#129）：monitor 下速率回到历史最优 95% 且当前尺寸低于最优
# 尺寸时，回到最优尺寸重新爬升——消除「降档单向棘轮」（降需连续 2 窗口
# <75%，回却需 >110%，一次短暂内存尖峰会永久降档）。低于 100% 是安全
# 的：回到最优尺寸后 probe 会重新验证，验证不过自然回退。
_SUBBATCH_RECOVER = 0.95
# 形状预算（#126 修订）：嵌入激活 ∝ 子批条数 × 批内最长序列²（attention 项
# 主导，对会物化注意力的 ONNX/CPU 路径成立）。预算按后端取（见
# model_backends._EMBEDDING_SHAPE_BUDGET）：mlx 走 flash attention（O(S)）
# 放到 B=4@8192，onnx 物化注意力（O(S²)）放到 B=2@8192，其它沿用 6M。不用
# 绝对 RSS 阈值：真实同步进程基线本就有 6-10GB（OCR 原生缓存、LanceDB 等），
# 绝对阈值会在首个子批就触发降档并永久卡死在下限（实测「子批很快降到 1」的
# 根因）。形状预算只对超长文本钳制条数，与进程里其它组件的内存正交；但它
# **不再把子批夹到 1**——最低保 2 条（见 _shape_cap_size），避免单条长切片
# 的吞吐塌陷与固定成本摊不薄。


def _shape_cap_size(size: int, max_len: int, backend: Optional[str] = None) -> int:
    """按形状预算收缩子批条数：超长文本下自动变小，短文本不受影响。

    这是**内存硬约束**，但收缩有下限：最低保 ``_SUBBATCH_MIN`` 条
    （2026-10-26 终值 6），与调优器的探索下限 ``_SUBBATCH_MIN`` 对齐，两者
    不再互相打架。单条长切片会把吞吐打到最优的 50-69%，且模型加载 / ONNX
    会话初始化的固定成本被单条摊得极薄；本项目切片硬限 500 字符，超长文本
    仅出现在异常输入，超预算部分由 MLX 内存策略（cache 1GB / 系统可用 60%）
    兜底。请求本身不足下限时按请求返回，不会为了凑下限而放大批次。

    预算按后端取：mlx 走 flash attention（O(S)）可放到 B=4@8192；onnx(CPU)
    物化注意力（O(S²)）放到 B=2@8192；其它后端沿用 6M。
    """
    if max_len <= 0:
        return size
    from .model_backends import (
        _EMBEDDING_SHAPE_BUDGET,
        _EMBEDDING_SHAPE_BUDGET_DEFAULT,
    )
    budget = _EMBEDDING_SHAPE_BUDGET.get(backend, _EMBEDDING_SHAPE_BUDGET_DEFAULT)
    return min(size, max(_SUBBATCH_MIN, budget // (max_len * max_len)))


# 每个窗口聚合的子批数：太小易被单批噪声带偏，太大响应迟钝。
_SUBBATCH_WINDOW = 2
# 爬升判定：候选子批的算力速率需比基线高 3% 才采纳——低于该阈值的波动
# 视为噪声，避免在吞吐平坦段来回抖动。
_SUBBATCH_IMPROVE = 1.03
# 降档判定：连续 2 个窗口的速率低于历史最优的 75% 视为环境劣化（OCR
# worker 抢占 CPU、热降频等），降 2 档避让，直到 4 为下限。
_SUBBATCH_DEGRADE = 0.75
_SUBBATCH_DEGRADE_STREAK = 2
# 历史最优速率的逐窗口衰减：环境永久变慢后旧纪录缓慢失效，不至于永远
# 拿旧峰值当标尺。
_SUBBATCH_BEST_DECAY = 0.999
# 重新爬升判定：monitor 阶段实测速率比历史最优高出 10%（如 OCR worker
# 结束、CPU 空出来）→ 环境已变，重新从小尺寸向上爬山，否则会永远停在
# 忙时降下来的小尺寸上。
_SUBBATCH_RE_CLIMB = 1.10


class _SubBatchTuner:
    """在 [6, 64] 内自动寻找吞吐最优的嵌入子批大小（爬山 + 降档）。

    度量：不用原始 tok/s——长度分桶后各批序列长度天然不同，直接比
    tok/s 有系统性偏差。改用算力速率 ``ops/s = n × max_len² / dt``（以
    字符数近似 token 数，单调代理即可），对批大小与批内最长长度同时
    归一，跨窗口、跨尺寸可比。

    状态机（#125 用户规格，2026-10-26 按 bge-small-zh @500 字切片实测重标定）：
      climb   初始 6（实测平台区，size 6-8 吞吐 ~97 条/s）：基线窗口（2 子批）
              → 试探 +2 → 更快（≥3%）则采纳并继续上调（上限 8），变慢则
              回退到上一个数字，进入 monitor。起步即落在实测最优区；
      monitor 持续盯速率：连续 2 窗口低于历史最优 75% → 降 1 档（步长 2）
              避让（下限 2），速率恢复即停；速率回到历史最优 95% 且尺寸低于
              最优尺寸 → 尖峰已过去，回到最优尺寸重新爬升（#129，消除降档
              单向棘轮）；实测速率比历史最优高 10% → 环境变快，重新爬升。

    内存安全双保险（#126/#129）：
      ① 形状预算：_embed_chunks 组装每个子批时按 ``_shape_cap_size`` 钳制
         条数（激活 ∝ 条数 × 批内最长序列²），超长文本自动变小，但最低保
         2 条、不夹到 1（吞吐保护，见 #139）。绝对 RSS 阈值在真实同步进程
         （基线 6-10GB）必然误触发，曾导致永久卡死。
      ② RSS 软墙 8GB（无闩锁，2026-10 用户定参保留 8GB）：超墙每子批
         force_down 降 1 档（步长 2），回墙下即停，恢复交给 monitor 恢复
         分支——旧 5GB 闩锁版是「降到下限再也不回去」的根因。调优器 report()
         收到的是钳制后的实际条数，状态机在可行域内正常爬山。

    预热丢弃（#126 修订）：首个完整窗口包含模型加载 / ONNX 会话初始化
    （可达 10s+），会把爬升基线拉到虚低，让热身的试探窗口显得「暴涨」
    被误采纳——第一个窗口只用于预热，丢弃不计入判定。

    线程安全：内部状态由锁保护（add() 与 add_streaming() 可能来自不同
    线程，窗口聚合不会交错损坏）。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.size = max(_SUBBATCH_MIN, min(_SUBBATCH_MAX, _SUBBATCH_START))
        self._phase = "climb"  # climb | monitor
        self._climb_state = "baseline"  # baseline | probe
        self._prev_size = self.size  # 爬升失败时回退到此处
        self._baseline_score: Optional[float] = None
        self._win_ops = 0.0
        self._win_dt = 0.0
        self._win_n = 0
        self._ema: Optional[float] = None
        self._best: Optional[float] = None
        # 历史最优速率对应的尺寸（#129）：恢复判定的参照——降档棘轮的解药
        self._best_size = self.size
        self._degrade_streak = 0
        self._warmed = False  # 预热窗口是否已丢弃

    def propose(self) -> int:
        """返回下一个子批应使用的大小。"""
        with self._lock:
            return self.size

    def report(self, n: int, max_len: int, dt: float) -> None:
        """上报一个子批的实测结果（条数、批内最长字符数、耗时）。"""
        ops = float(n) * float(max(max_len, 1)) ** 2
        with self._lock:
            self._win_ops += ops
            self._win_dt += max(dt, 1e-6)
            self._win_n += 1
            if self._win_n < _SUBBATCH_WINDOW:
                return
            score = self._win_ops / self._win_dt
            self._win_ops = 0.0
            self._win_dt = 0.0
            self._win_n = 0
            if not self._warmed:
                # 预热窗口丢弃：含模型加载/会话初始化，基线会虚低
                self._warmed = True
                return
            if self._phase == "climb":
                self._on_climb_window(score)
            else:
                self._on_monitor_window(score)

    def _on_climb_window(self, score: float) -> None:
        """爬升阶段窗口结果。调用时已持锁。"""
        if self._climb_state == "baseline":
            self._baseline_score = score
            nxt = self.size + _SUBBATCH_STEP
            if nxt > _SUBBATCH_MAX:
                self._phase = "monitor"
                self._best_size = self.size  # 操作点已重新验证（#129）
                return
            self._prev_size = self.size
            self.size = nxt
            self._climb_state = "probe"
            logger.info(
                "  ├─ 子批自适应: %d → %d (试探吞吐)", self._prev_size, nxt
            )
            return
        # probe 窗口结果：与基线比较
        assert self._baseline_score is not None
        ratio = score / self._baseline_score
        if ratio >= _SUBBATCH_IMPROVE:
            logger.info(
                "  ├─ 子批自适应: %d 吞吐 %+.0f%%, 采纳并继续上调",
                self.size, (ratio - 1) * 100,
            )
            self._baseline_score = score
            self._best_size = self.size  # 采纳 = 已验证（#129）
            nxt = self.size + _SUBBATCH_STEP
            if nxt > _SUBBATCH_MAX:
                self._phase = "monitor"
            else:
                self._prev_size = self.size
                self.size = nxt
        else:
            logger.info(
                "  ├─ 子批自适应: %d 吞吐 %+.0f%%, 回退到 %d",
                self.size, (ratio - 1) * 100, self._prev_size,
            )
            self.size = self._prev_size
            self._phase = "monitor"
            # 回退 = 大尺寸被否决，操作点锚在此处，防止恢复循环（#129）
            self._best_size = self.size

    def _on_monitor_window(self, score: float) -> None:
        """监控阶段窗口结果：盯劣化降档，盯变快重新爬升。调用时已持锁。"""
        self._ema = score if self._ema is None else self._ema * 0.5 + score * 0.5
        prev_best = self._best
        # 恢复（#129）：当前尺寸低于最优尺寸、且速率回到历史最优 95% →
        # 尖峰（内存压力 / 抢 CPU）已过去，直接回到最优尺寸（不走路爬山：
        # plateau 上试探必然持平 <3% 被回退，回退锚会把最优尺寸永久改小，
        # 恰好复现「降到 1 再也不回去」）。留在 monitor：若最优尺寸在新
        # 环境下确实过慢，降档路径会再次接管，自校正无循环。放在 best
        # 更新之前，避免恢复窗口把 best/best_size 就地改写成小尺寸。
        if (
            prev_best is not None
            and prev_best > 0
            and self._ema is not None
            and self.size < self._best_size
            and self._ema >= prev_best * _SUBBATCH_RECOVER
        ):
            logger.info(
                "  ├─ 子批自适应: 速率恢复到最优的 %.0f%%, %d → %d 回到最优尺寸",
                self._ema / prev_best * 100, self.size, self._best_size,
            )
            self.size = self._best_size
            self._ema = None
            self._degrade_streak = 0
            return
        if self._best is None or self._ema > self._best:
            self._best = self._ema
        else:
            self._best *= _SUBBATCH_BEST_DECAY
        # 环境明显变快（新 EMA 比旧纪录高 10%）→ 重新爬升
        if (
            prev_best is not None
            and self._ema is not None
            and prev_best > 0
            and self._ema >= prev_best * _SUBBATCH_RE_CLIMB
        ):
            logger.info(
                "  ├─ 子批自适应: 速率比历史最优高 %.0f%%, 环境变快, 重新爬升",
                (self._ema / prev_best - 1) * 100,
            )
            self._phase = "climb"
            self._climb_state = "baseline"
            self._baseline_score = None
            self._ema = None
            self._degrade_streak = 0
            return
        if self._ema is not None and self._best and self._ema < self._best * _SUBBATCH_DEGRADE:
            self._degrade_streak += 1
        else:
            self._degrade_streak = 0
        if self._degrade_streak >= _SUBBATCH_DEGRADE_STREAK and self.size > _SUBBATCH_MIN:
            new_size = max(_SUBBATCH_MIN, self.size - _SUBBATCH_STEP)
            logger.info(
                "  ├─ 子批自适应: 吞吐降至最优的 %.0f%%, %d → %d 避让",
                self._ema / self._best * 100, self.size, new_size,
            )
            self.size = new_size
            self._ema = None
            self._degrade_streak = 0

    def force_down(self) -> None:
        """RSS 软墙（#129）：降 1 档并丢弃半满窗口。

        与速率状态机独立，**无闩锁**：超墙时每个子批调一次（激活随之变小、
        RSS 自然回落），回墙下即停，之后由恢复分支 / 重爬把尺寸带回来。
        旧 5GB 闩锁版（mem_pressed 永不解除）会在真实同步基线 6-10GB 下
        首个子批就触发并永久卡死在下限，是「降到 1 再也不回去」的根因。
        """
        with self._lock:
            self._win_ops = 0.0
            self._win_dt = 0.0
            self._win_n = 0
            if self.size <= _SUBBATCH_MIN:
                return
            new_size = max(_SUBBATCH_MIN, self.size - _SUBBATCH_STEP)
            logger.info(
                "  ├─ 子批自适应: RSS 超 %.0fGB, %d → %d 降档避让",
                _SUBBATCH_RSS_LIMIT, self.size, new_size,
            )
            self.size = new_size
            self._phase = "monitor"
            self._ema = None
            self._degrade_streak = 0


# ---------------------------------------------------------------------------
# Vector store backed by LanceDB
# ---------------------------------------------------------------------------


class VectorStore:
    """Local vector store backed by LanceDB with FastEmbed embeddings.

    Uses nomic-ai/nomic-embed-text-v1.5 via ONNX Runtime for local inference.
    Supports optional BGE reranker for precise post-retrieval scoring.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        embedding_function: Optional[Any] = None,
        embedding_backend: str = "fastembed",
        embedding_model_path: str = "",
        embedding_threads: int = 4,
        embedding_batch_size: int = 8,
        embedding_max_length: int = 8192,
        rerank_backend: str = "fastembed",
        rerank_model_path: str = "",
        rerank_mlx_model_path: str = "",
        rerank_jina_model_path: str = "",
        rerank_threads: int = 4,
        rerank_batch_size: int = 16,
        rerank_max_length: int = 8192,
        rerank_enabled: bool = True,
    ):
        self._persist_dir = persist_dir
        self._embedding_model = embedding_model
        self._embedding_backend = embedding_backend
        self._embedding_model_path = embedding_model_path
        self._embedding_threads = max(1, int(embedding_threads))
        self._embedding_batch_size = max(1, int(embedding_batch_size))
        self._embedding_max_length = max(8, int(embedding_max_length))
        self._rerank_model = rerank_model
        self._rerank_backend = rerank_backend if rerank_enabled else "disabled"
        self._rerank_model_path = rerank_model_path
        self._rerank_mlx_model_path = rerank_mlx_model_path
        self._rerank_jina_model_path = rerank_jina_model_path
        self._rerank_threads = max(1, int(rerank_threads))
        self._rerank_batch_size = max(1, int(rerank_batch_size))
        self._rerank_max_length = max(8, int(rerank_max_length))
        self._custom_ef = embedding_function  # for testing injection
        self._db = None
        self._table = None
        self._documents_table = None  # 文档元数据表（缩略版本 + markdown）
        # 保护 documents 表懒创建 + 写操作。用 RLock：upsert_document /
        # delete_document 会「持锁后再调用 _ensure_documents_table()」，普通
        # Lock 会在此自锁死（见 _ensure_documents_table 的注释）。
        self._documents_lock = threading.RLock()
        # 保护整个 store 的懒初始化（连接/建表/schema 迁移只允许一次）。
        self._init_lock = threading.RLock()
        # 保护 reranker 状态机（避免并发加载两份 ~500MB 模型）。
        self._reranker_lock = threading.RLock()
        # close() 后置位：后台加载线程完成时不得把模型写回本实例。
        self._closed = False
        # ---- 在途操作计数：与 clear()/close() 互斥 ----
        # clear() 会 drop_table、close() 会把 _db 置 None，二者若与在途的
        # add/search 并发，在途操作会打到已删除的表或 None 连接上（表现是
        # 随机 AttributeError / 检索到已删数据）。这里用在途计数 + 条件变量：
        # 业务操作进出各记一次，clear/close 先排空在途操作再动手。
        self._op_cv = threading.Condition(threading.Lock())
        self._inflight = 0
        self._closing = False
        # documents 表内容缓存：避免 get_document_digest / list_documents 每次
        # 都全表 to_pylist()（表内含完整 markdown，反复全量扫描既慢又占内存）。
        # 写入（upsert/delete/clear）时置 None 失效，下次读取时重新加载。
        self._documents_cache: Optional[List[Dict[str, Any]]] = None
        self._ef = None  # embedding function
        self._subbatch_tuner = _SubBatchTuner()  # 自适应子批大小（#125）
        self._initialized = False
        self._table_checked = False  # 是否已检查过 table_names（避免重复磁盘扫描）
        self._fts_ready = False  # BM25 全文索引是否已就绪
        self._fts_checked = False  # 是否已尝试创建过 FTS 索引
        self._fts_failed = False  # 建索引曾失败 → 不再重试（见 _ensure_fts_index）

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 在途操作计数（与 clear/close 互斥）
    # ------------------------------------------------------------------

    def _begin_op(self) -> None:
        """登记一次在途操作。store 已关闭/正在关闭时抛明确的错误。

        ``_closing`` 必须一起检查：``clear()`` 排空完在途操作后会执行
        drop_table，若此时新操作仍能进来，就正好撞上这次改动想消除的竞态
        （读到/写到一张已被删除的表）。旧实现只检查 ``_closed``，漏了它。
        """
        with self._op_cv:
            if self._closed:
                raise RuntimeError("VectorStore is closed")
            if self._closing:
                raise RuntimeError("VectorStore is shutting down (clear in progress)")
            self._inflight += 1

    def _end_op(self) -> None:
        with self._op_cv:
            self._inflight = max(0, self._inflight - 1)
            if self._inflight == 0:
                self._op_cv.notify_all()

    def _drain_ops(self, timeout: float = 10.0) -> bool:
        """排空在途操作，期间拒绝新的操作。返回 True 表示已排空。

        超时后仍会继续：关闭/清空不能因为某个卡住的操作（例如慢 embedding）
        而永久挂起，宁可冒一次竞态也不能把 UI 卡死。
        """
        deadline = time.monotonic() + timeout
        with self._op_cv:
            self._closing = True
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "VectorStore: %d 个在途操作未在 %.0fs 内结束，将继续关闭",
                        self._inflight, timeout,
                    )
                    return False
                self._op_cv.wait(remaining)
            return True

    def _resume_ops(self) -> None:
        """clear() 结束后解除「拒绝新操作」状态。"""
        with self._op_cv:
            self._closing = False
            self._op_cv.notify_all()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        # close() 之后 _db 已被置 None：这里给出明确错误，而不是让后面的
        # lancedb.connect / create_table 撞上 None。
        if getattr(self, "_closed", False):
            raise RuntimeError("VectorStore is closed")
        # ingest 的 4 个 worker 会并发走到这里（upsert_document / add_streaming
        # 都会触发），若不加锁，多个线程会同时执行 lancedb.connect /
        # create_table / schema 迁移，轻则重复建表报错，重则 schema 迁移互相
        # 覆盖。与 _ensure_documents_table 相同的双重检查模式。
        with self._init_lock:
            if self._initialized:
                return
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        """``_ensure_initialized`` 的实现，调用方必须已持有 _init_lock。"""
        logger.info("  正在初始化向量库...")
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb is required for RAG vector storage. pip install lancedb"
            )

        os.makedirs(self._persist_dir, exist_ok=True)

        logger.info("  连接 LanceDB: %s", self._persist_dir)
        self._db = lancedb.connect(self._persist_dir)

        # Use injected embedding function (for testing) or the configured shared backend.
        if self._custom_ef is not None:
            self._ef = self._custom_ef
        else:
            self._ef = get_or_create_embedding_function(
                self._embedding_model,
                backend=self._embedding_backend,
                model_path=self._embedding_model_path,
                threads=self._embedding_threads,
                batch_size=self._embedding_batch_size,
                max_length=self._embedding_max_length,
            )

        # Open existing table or defer creation until first add
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            self._check_schema_migration()

        # Open the documents metadata table (if it already exists).
        if _DOCUMENTS_TABLE_NAME in self._db.table_names():
            self._documents_table = self._db.open_table(_DOCUMENTS_TABLE_NAME)

        self._initialized = True
        count = self._table.count_rows() if self._table is not None else 0
        logger.info("  向量库初始化完成: 当前存储 %d 条向量记录", count)

    def _check_schema_migration(self) -> None:
        """Migrate an existing table to include ``text_fts`` / ``ingested_at``.

        The BM25 full-text index lives on a pre-tokenized ``text_fts`` column;
        向量去重按 ``ingested_at`` 保留最早入库的一份。Tables created before
        these features lack the columns; since LanceDB schemas are immutable,
        we rebuild the table (data survives — it is re-read from the old table
        and re-written with the new columns). 旧行没有 ingested_at，回填
        0.0（视为「最老」——历史数据天然占先，不会被重新入库的副本顶掉）。

        #140 内存事故修正（2026-10-26，22GB 卡死）：
        1. 旧表向量维度与当前嵌入模型不一致时**跳过迁移**——那些数据对当前
           模型毫无意义（反正要清库重建），而全表迁移曾把 57 万行 × 768 维
           向量读成 Python 对象（每 float 一个 PyHeap 对象 ≈ 20GB）直接把
           机器卡死。清库交给 reset_if_incompatible()。
        2. 迁移本身分批转换（每批 ≤2000 行），且**保留旧表已有列**——旧实现
           的 records 只重建 5 个基础列，旧表只缺 ingested_at 时会把已有的
           text_fts 整个丢掉，产出一必然写入失败的坏表（用户看到的
           "text_fts does not exist" 正是这么来的）。
        """
        if self._table is None:
            return
        try:
            schema = self._table.schema
            missing = [c for c in (_FTS_COLUMN, _INGESTED_AT)
                       if c not in schema.names]
            if not missing:
                return
        except Exception:
            return

        # 维度守卫：无法确认「旧表维度 == 当前嵌入维度」就绝不做全表迁移。
        try:
            table_dim = schema.field("vector").type.list_size
        except Exception:
            table_dim = None
        ef_dim = self._probe_ef_dim()
        if table_dim is None or ef_dim is None or table_dim != ef_dim:
            logger.info(
                "  ⚙ 旧表向量维度(%s)与当前嵌入模型(%s)不一致或无法确认，"
                "跳过 schema 迁移——旧数据将由兼容性检查清库重建",
                table_dim, ef_dim,
            )
            return

        logger.info("  ⚙ 检测到旧版向量表（缺少 %s 列），自动迁移...", "/".join(missing))
        try:
            import pyarrow as pa

            # 目标 schema = 现有列 + 缺失列。
            target_names = set(schema.names) | set(missing)
            target_fields = list(schema)
            if _FTS_COLUMN in missing:
                target_fields.append(pa.field(_FTS_COLUMN, pa.string()))
            if _INGESTED_AT in missing:
                target_fields.append(pa.field(_INGESTED_AT, pa.float64()))
            target_schema = pa.schema(target_fields)

            # 分批转换：Python 对象只在单批（≤2000 行）内瞬时存在，累计
            # 数据用紧凑的 Arrow batch（float32 4 字节/值，全表 ~2GB 而非
            # ~20GB）。
            arrow_table = self._table.to_arrow()
            out_batches = []
            for batch in arrow_table.to_batches(max_chunksize=2000):
                records = []
                for row in batch.to_pylist():
                    # 保留旧表已有列（含 text_fts）——迁移只负责补缺，不丢数据。
                    rec = {k: row[k] for k in target_names if k in row}
                    if _FTS_COLUMN in missing:
                        rec[_FTS_COLUMN] = _tokenize_for_fts(rec["text"])
                    if _INGESTED_AT in missing:
                        rec[_INGESTED_AT] = 0.0
                    records.append(rec)
                out_batches.append(
                    pa.RecordBatch.from_pylist(records, schema=target_schema))
            del arrow_table  # 建表前先释放源数据的 Arrow 副本

            # LanceDB OSS 无 rename_table，只能 drop 后重建。数据已在内存
            # 的紧凑 batches 中，drop 旧表后 create 失败可再次尝试；即使最终
            # 失败，RAG 数据也可从源文件重新同步恢复（幂等）。
            self._db.drop_table(_TABLE_NAME)
            self._table = None
            if out_batches:
                self._table = self._db.create_table(
                    _TABLE_NAME,
                    pa.Table.from_batches(out_batches, schema=target_schema))
            else:
                self._table = self._db.create_table(
                    _TABLE_NAME, schema=target_schema)
            del out_batches
            self._table_checked = True
            logger.info("  ⚙ 向量表迁移完成（新增 %s 列，%d 条记录）",
                        "/".join(missing), self._table.count_rows())
        except Exception as e:
            logger.warning("  ⚠ 向量表迁移失败（BM25 索引将不可用）: %s", e)
            self._table = None
            self._table_checked = False

    def _probe_ef_dim(self) -> Optional[int]:
        """探测当前 embedding function 的输出维度。

        返回 None 表示探测失败（EF 不可用）——调用方不应据此清库。
        """
        try:
            if hasattr(self._ef, "embed_query"):
                vec = self._ef.embed_query("维度探测")
            else:
                vec = self._ef(["维度探测"])[0]
            return len(vec)
        except Exception as e:
            logger.warning("  ⚠ 嵌入维度探测失败: %s", e)
            return None

    def reset_if_incompatible(self) -> bool:
        """向量库与当前嵌入模型不兼容时，自愈式清空重建。

        典型场景：用户从 nomic（768 维）切换到 bge-small（512 维）后，旧表
        里的 768 维向量与新写入的 512 维向量无法共存（LanceDB schema 固定
        向量宽度），写入报错。此处检查表是否存在且维度匹配，不匹配则
        drop 主表并复位内部状态，由调用方负责作废 manifest（全量重建）。

        返回 True 表示执行了重置（调用方需作废同步记录）。
        """
        self._ensure_initialized()
        if self._table is None:
            return False
        # 与 clear() 相同的并发安全姿势：先排空在途操作。
        try:
            self._drain_ops(timeout=10.0)
        except Exception:
            pass
        try:
            schema = self._table.schema
            names = schema.names
            expected_dim = self._probe_ef_dim()
            current_dim = None
            if "vector" in names:
                try:
                    current_dim = schema.field("vector").type.list_size
                except Exception:
                    current_dim = None

            incompatible = False
            reasons = []
            if expected_dim is None:
                # EF 不可用：无法判断，宁可保守不清库
                logger.warning("  ⚠ 无法探测嵌入维度，跳过向量库兼容性检查")
                return False
            if "vector" not in names:
                incompatible = True
                reasons.append("缺少 vector 列")
            elif current_dim != expected_dim:
                incompatible = True
                reasons.append(
                    f"向量维度不匹配（库内 {current_dim} 维 vs 当前嵌入 {expected_dim} 维）")
            for col in ("id", _FTS_COLUMN, _INGESTED_AT):
                if col not in names:
                    incompatible = True
                    reasons.append(f"缺少 {col} 列")

            if not incompatible:
                return False

            logger.warning(
                "  ⚠ 向量库与当前嵌入模型不兼容（%s），执行自愈重建...", "；".join(reasons))
            self._db.drop_table(_TABLE_NAME)
            self._table = None
            self._table_checked = False
            # FTS 状态一并复位（表都没了，旧标记毫无意义）。
            self._fts_ready = False
            self._fts_checked = False
            self._fts_failed = False
            return True
        except Exception as e:
            logger.warning("  ⚠ 向量库兼容性检查/重置失败: %s", e)
            return False
        finally:
            self._resume_ops()

    def _ensure_table(self) -> Optional[Any]:
        """Open existing table if it exists, return None if not yet created.

        Re-checks ``table_names()`` on every call when ``_table`` is None,
        because the table may have been created by another RAGEngine instance
        (e.g. SyncKnowledgeWorker) after this VectorStore was first initialized.
        Caching "table doesn't exist" was a premature optimization that caused
        a real bug: after sync wrote data via a separate engine, this engine's
        search() still returned empty because it never re-checked.

        LanceDB's ``table_names()`` is cheap (directory listing), so the
        per-call cost is acceptable for correctness.
        """
        if self._table is not None:
            return self._table
        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
            self._table_checked = True
        return self._table

    def _create_table_atomic(self, records: List[Dict[str, Any]]):
        """并发安全建表。

        返回 ``(table, created)``：
          * ``created=True`` —— 本次调用用 ``records`` 新建了表；
          * ``created=False`` —— 表已存在（可能刚被并发线程创建），调用方
            必须走正常的 upsert 路径而不是提前返回。

        修复竞态：ingest worker / 多线程首次写入时，旧实现里两个线程同时
        看到「表不存在」→ 都执行 create_table → 一方抛 "Table already
        exists"，该 worker 的 ingest 直接崩掉。
        """
        with self._init_lock:  # RLock，内部可再进 _ensure_table
            table = self._ensure_table()
            if table is not None:
                return table, False
            self._table = self._db.create_table(_TABLE_NAME, records)
            self._table_checked = True
            return self._table, True

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_vector_dim_compat(table: Any, records: List[Dict[str, Any]]) -> None:
        """写入前校验表向量维度与数据一致，不一致立刻抛明确错误。

        #140 防御兜底：旧表（如 nomic 768 维）未被清掉时，首批写入就拿到
        可行动的错误提示，而不是先执行完 delete 再在 add 上报晦涩的
        schema mismatch。
        """
        if table is None or not records:
            return
        try:
            table_dim = table.schema.field("vector").type.list_size
        except Exception:
            return
        vec = records[0].get("vector")
        if vec is not None and len(vec) != table_dim:
            raise RuntimeError(
                f"向量库维度不匹配：库内向量 {table_dim} 维，当前嵌入模型输出 "
                f"{len(vec)} 维（嵌入模型切换后旧库未清空）。"
                f"请重新启动同步，将自动清库重建。")

    @_store_op
    def add(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Add document chunks to the vector store. Returns number of chunks added.

        Upsert 语义：先按 source 删除该来源的全部旧记录（处理文档缩减后
        残留旧 chunks 的情况），再插入新记录。embedding 分批进行以避免 OOM。
        """
        self._ensure_initialized()
        if not chunks:
            return 0

        # Build records (embed in batches to avoid OOM on large knowledge bases)
        records: List[Dict[str, Any]] = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch_chunks]
            vectors = self._ef(texts)
            for j, chunk in enumerate(batch_chunks):
                idx = i + j
                source = chunk.get("source", "unknown")
                chunk_idx = chunk.get("chunk_index", idx)
                chunk_id = f"{_hash_source(source)}_{chunk_idx}"
                records.append({
                    "id": chunk_id,
                    "text": chunk["text"],
                    "vector": _normalize(vectors[j]),
                    "source": source,
                    "chunk_index": chunk_idx,
                    _FTS_COLUMN: _tokenize_for_fts(chunk["text"]),
                    _INGESTED_AT: time.time(),
                })

        table = self._ensure_table()
        self._assert_vector_dim_compat(table, records)

        if table is None:
            # First add: create table with all records（并发安全）
            table, created = self._create_table_atomic(records)
            if created:
                self._fts_ready = False  # 推迟到下次 search 重建 BM25 索引
                logger.info("LanceDB table '%s' created with %d records", _TABLE_NAME, len(records))
                return len(records)
            # 并发线程刚建了表 → 继续走下面的 upsert 路径

        # Upsert: 按 source 删除该来源的全部旧记录。
        # 用 source 的 md5 hash 作为 LIKE 模式（hash 是 hex 字符串，无 SQL 注入风险），
        # 既能覆盖 chunk_index 0..N 的所有旧记录，也能清理文档缩减后多出的旧 chunks。
        sources_to_replace = sorted({c.get("source", "unknown") for c in chunks})
        for source in sources_to_replace:
            h = _hash_source(source)
            try:
                # id 格式为 "{hash}_{chunk_index}"，LIKE '{hash}_%' 匹配该 source 的所有 chunk
                table.delete(f"id LIKE '{h}_%'")
            except Exception as e:
                logger.debug("VectorStore: delete by source '%s' failed: %s", source, e)

        # Add records in batches（单批失败不中断，最大限度保留数据）
        total = 0
        failed_batches = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                table.add(batch)
                total += len(batch)
            except Exception as e:
                failed_batches += 1
                logger.error(
                    "VectorStore: add batch %d/%d failed: %s",
                    i // batch_size + 1, (len(records) + batch_size - 1) // batch_size, e,
                )
                if failed_batches >= 3:
                    # 连续失败太多，可能系统性问题，中断避免持续报错
                    raise
        if failed_batches > 0:
            logger.warning("VectorStore: %d batch(es) failed during add", failed_batches)

        # 数据写入后**不立即**重建 BM25 全文索引：一次同步会多次调 add()，
        # 每次都全表重建是 O(N) 的浪费。置 _fts_ready=False 把重建推迟到
        # 下一次 search_fts()（那里已有惰性建索引路径），一次同步只付一次
        # 建索引成本。
        self._fts_ready = False

        logger.info("VectorStore: added %d chunks", total)
        return total

    @_store_op
    def delete_by_source(self, source: str) -> int:
        """删除指定 source 的全部记录。返回删除前的记录数（若可查）。

        用于 force 重新 ingest 时清理已从知识库中删除的文件对应的旧向量。
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return 0
        h = _hash_source(source)
        # 删除失败必须抛出而不是吞掉：engine._cleanup_deleted_files 依赖
        # 「无异常 = 删除成功」决定是否从 manifest 移除条目——吞掉异常会让
        # 失败文件被无条件 pop，向量永久残留且下次同步没有重试机会。
        try:
            table.delete(f"id LIKE '{h}_%'")
            # 向量删了，但 BM25 倒排索引里的 term 还在——必须失效索引，
            # 否则已删除文档的内容仍会被关键词检索召回并返回给 LLM。
            self._fts_ready = False
            logger.info("VectorStore: deleted records for source '%s'", source)
            return 1
        except Exception as e:
            logger.warning("VectorStore: delete_by_source '%s' failed: %s", source, e)
            raise

    @_store_op
    def add_streaming(
        self,
        chunk_iter,
        batch_size: int = 100,
        on_batch: Optional[Any] = None,
        is_cancelled: Optional[Any] = None,
    ) -> int:
        """流式增量写入：接受 chunk 迭代器，每批 embed 完立即写入并释放。

        用于大知识库 ingest，避免 H3 隐患（全部 records 含 vector 同时在内存）。
        内存占用 = O(batch_size) 而非 O(总 chunks 数)。

        Args:
            chunk_iter: 产出 chunk dict 的迭代器（必须有 source / chunk_index / text）
            batch_size: 每批 embed + 写入的 chunk 数
            on_batch: 可选回调 (batch_added, total_added) -> None
            is_cancelled: 可选回调 () -> bool，返回 True 时中断写入并返回当前计数

        Returns:
            成功写入的 chunk 总数（取消时返回已写入数）
        """
        self._ensure_initialized()

        # 第一批特殊处理：用于判断是否需要 create_table
        first_batch: List[Dict[str, Any]] = []
        exhausted = False
        for _ in range(batch_size):
            try:
                first_batch.append(next(chunk_iter))
            except StopIteration:
                exhausted = True
                break

        if not first_batch:
            logger.info("  向量化写入: 无切片需要写入（空迭代器）")
            return 0

        # 第一批处理前检查取消标志
        if is_cancelled is not None and is_cancelled():
            return 0

        # Embed 第一批
        logger.info("  ├─ 向量化: 首批 %d 个切片正在嵌入...", len(first_batch))
        first_records = self._embed_chunks(first_batch, 0)
        logger.info("  ├─ 向量化完成: 首批 %d 个切片", len(first_records))

        table = self._ensure_table()
        self._assert_vector_dim_compat(table, first_records)
        # 跟踪已删除过旧数据的 source，避免后续批次重复写入时新旧数据共存。
        sources_seen = {r["source"] for r in first_records}
        created = False
        if table is None:
            # 首次：创建表（用第一批 records），无需删除旧数据。并发安全。
            table, created = self._create_table_atomic(first_records)
        if created:
            logger.info("  ├─ 向量库: 新建 LanceDB 表 '%s', 写入 %d 条记录", _TABLE_NAME, len(first_records))
            total = len(first_records)
            if on_batch:
                on_batch(len(first_records), total)
            if exhausted:
                logger.info("  └─ 向量化写入完成: 共 %d 个切片 (仅一批)", total)
                # 置脏即可——BM25 索引推迟到下次 search_fts 惰性重建。
                self._fts_ready = False
                return total
            # 未耗尽：继续后续批次（table 已建好）
        else:
            # 已有表（或并发线程刚建）：先按 source 删除旧记录
            existing_count = table.count_rows()
            logger.info(
                "  ├─ 向量库: 已有表 (%d 行), 将替换 %d 个来源的旧数据",
                existing_count, len(sources_seen),
            )
            for source in sources_seen:
                h = _hash_source(source)
                try:
                    table.delete(f"id LIKE '{h}_%'")
                except Exception as e:
                    logger.debug("VectorStore: delete by source '%s' failed: %s", source, e)
            # 写入第一批
            try:
                table.add(first_records)
                total = len(first_records)
                logger.info("  ├─ 向量库写入: 首批 %d 条记录", total)
                if on_batch:
                    on_batch(len(first_records), total)
            except Exception as e:
                logger.error("  向量库写入失败(首批): %s", e)
                self._fts_ready = False  # 已删旧数据但新数据未写入 → 索引必须重建
                raise
            if exhausted:
                logger.info("  └─ 向量化写入完成: 共 %d 个切片 (仅一批)", total)
                # 置脏即可——BM25 索引推迟到下次 search_fts 惰性重建。
                self._fts_ready = False
                return total

        # 后续批次
        chunk_idx = len(first_batch)
        batch_buf: List[Dict[str, Any]] = []
        batch_num = 1
        deleted_sources: set = sources_seen.copy() if table is not None else set()
        for chunk in chunk_iter:
            batch_buf.append(chunk)
            if len(batch_buf) >= batch_size:
                if is_cancelled is not None and is_cancelled():
                    # 已写入的行必须让 BM25 索引重建（置脏），否则这批新数据
                    # 对关键词检索不可见，混合检索静默退化成纯向量检索。
                    self._fts_ready = False
                    return total
                batch_num += 1
                logger.info("  ├─ 向量化: 第 %d 批 (%d 切片) 正在嵌入...", batch_num, len(batch_buf))
                if table is not None:
                    new_sources = {c["source"] for c in batch_buf} - deleted_sources
                    if new_sources:
                        for source in new_sources:
                            h = _hash_source(source)
                            try:
                                table.delete(f"id LIKE '{h}_%'")
                            except Exception:
                                pass
                        deleted_sources.update(new_sources)
                records = self._embed_chunks(batch_buf, chunk_idx)
                try:
                    table.add(records)
                    total += len(records)
                    logger.info("  ├─ 向量库写入: 第 %d 批 %d 条 (累计 %d)", batch_num, len(records), total)
                    if on_batch:
                        on_batch(len(records), total)
                except Exception as e:
                    logger.error("  向量库写入失败(第 %d 批): %s", batch_num, e)
                    raise
                chunk_idx += len(batch_buf)
                batch_buf.clear()
        # 处理剩余
        if batch_buf:
            if is_cancelled is not None and is_cancelled():
                # 同取消分支：已写入的行要让 BM25 索引置脏重建。
                self._fts_ready = False
                return total
            batch_num += 1
            logger.info("  ├─ 向量化: 最后批次 (%d 切片) 正在嵌入...", len(batch_buf))
            if table is not None:
                new_sources = {c["source"] for c in batch_buf} - deleted_sources
                if new_sources:
                    for source in new_sources:
                        h = _hash_source(source)
                        try:
                            table.delete(f"id LIKE '{h}_%'")
                        except Exception:
                            pass
            records = self._embed_chunks(batch_buf, chunk_idx)
            try:
                table.add(records)
                total += len(records)
                logger.info("  ├─ 向量库写入: 最后批次 %d 条 (累计 %d)", len(records), total)
                if on_batch:
                    on_batch(len(records), total)
            except Exception as e:
                logger.error("  向量库写入失败(最后批次): %s", e)
                raise

        logger.info("  └─ 向量化写入完成: 共 %d 个切片 (%d 批次)", total, batch_num)

        # 数据写入后只置脏 BM25 全文索引：旧实现每次 add_streaming 都全表
        # rebuild（replace=True），一次同步 N 批 = N 次全库建索引。重建推迟
        # 到下一次 search_fts()（惰性路径已存在），一次同步只付一次成本。
        # 注：LanceDB 的 FTS 索引不会在 add() 后自动增量更新。
        self._fts_ready = False

        return total

    def _embed_chunks(
        self, chunks: List[Dict[str, Any]], start_idx: int,
    ) -> List[Dict[str, Any]]:
        """Embed 一批 chunks，返回 records（含 vector）。内部辅助方法。

        将每批拆分为更小的子批次进行 embedding，每个子批次完成后主动
        time.sleep() 释放 GIL，让主线程有机会处理 Qt 事件（日志渲染、进度
        更新）。ONNX Runtime 的 embed() 在 C++ 计算期间持有 GIL，整批一次
        性嵌入会导致 UI 长时间冻结。

        子批大小不固定（历史值 12）：由 :class:`_SubBatchTuner` 在 1-5 内
        自适应（#125，#126/#127/#128 生产验证修订：真实切片上吞吐与内存都
        随子批增大单调变差，size2 双优）。长度分桶后 padding 已归零。

        形状预算（#126 修订）：每个子批组装时按 ``_shape_cap_size`` 钳制
        条数（激活 ∝ 条数 × 批内最长序列²），超长文本自动变小。预算按后端
        取——mlx 走 flash attention 放到 B=4@8192，onnx 物化注意力放到
        B=2@8192（详见 ``_EMBEDDING_SHAPE_BUDGET``）；但**最低保 2 条、不
        夹到 1**（#139，单条长切片吞吐塌陷且固定成本摊不薄）。不再使用绝对
        RSS 阈值——真实同步进程基线 6-10GB，绝对阈值会误触发并永久卡死在下限。

        关键优化：**按文本长度排序分桶**（#122）。fastembed 的批内所有序列
        padding 到批内最长——长度混杂的批 padding 浪费 2-4 倍算力（实测
        32 条混杂批 309s ≈ 40 tok/s，而同模型「长度排序 + 8-12 条/批」
        ~2000 tok/s，差 30 倍）。排序后每批长度相近、padding 归零，向量按
        原顺序回填，结果与混杂批完全一致。
        """
        import time as _time
        # 让出 GIL 的累计耗时阈值：不是「每子批都 sleep」，而是累计占用 GIL
        # 超过 50ms 才让出一次，保证 UI 响应又不为 sleep 付出固定开销。
        _YIELD_INTERVAL = 0.05
        texts = [c["text"] for c in chunks]
        # 长度降序分桶：批内长度相近，padding 浪费最小（降序让最长批最先
        # 出现，提早把激活水位顶到峰值，后续批更轻）。
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        vectors_by_idx: List[Optional[List[float]]] = [None] * len(texts)
        tuner = self._subbatch_tuner
        _last_yield = _time.monotonic()
        _t_embed = _time.monotonic()
        sub_idx = 0
        sub_start = 0
        while sub_start < len(order):
            size = tuner.propose()
            # 形状预算钳制（#126 修订）：激活 ∝ 条数 × 批内最长序列²，对
            # 超长文本收缩本子批条数，把瞬时激活压在 ~2GB（常量注释含标定）。
            # 降序分桶下切片首条即批内最长，钳制后 report() 收到实际条数。
            size = _shape_cap_size(size, len(texts[order[sub_start]]), self._embedding_backend)
            idxs = order[sub_start:sub_start + size]
            _t_sub = _time.monotonic()
            sub_vectors = self._ef([texts[i] for i in idxs])
            _dt = _time.monotonic() - _t_sub
            for i, v in zip(idxs, sub_vectors):
                vectors_by_idx[i] = v
            sub_idx += 1
            sub_start += len(idxs)
            # 子批级进度日志：嵌入一个子批要数十秒且中间完全静默，用户会
            # 误判为「卡死」（#122）。每个子批一条，给出速率与当前子批大小。
            n_chars = sum(len(texts[i]) for i in idxs)
            logger.info(
                "  ├─ 嵌入子批 #%d [size=%d]: %d 条 %.1fk字符, %.1fs (%.0f k字符/s, 累计 %.0fs)",
                sub_idx, size, len(idxs), n_chars / 1000, _dt,
                n_chars / max(_dt, 1e-6) / 1000,
                _time.monotonic() - _t_embed,
            )
            tuner.report(len(idxs), max(len(texts[i]) for i in idxs), _dt)
            # 逐子批内存打点（RAG_MEM_LOG 开启时）：footprint 口径（RSS+swap
            # +MLX 统计）。RSS 不含 Metal 缓冲与换页——实测同一进程 RSS 2.2GB
            # vs footprint 12.1GB，只记 RSS 会看不到嵌入期的真实增长。
            if _MEM_LOG_PATH:
                try:
                    import psutil as _psutil

                    _mi = _psutil.Process().memory_info()
                    _swap = 0
                    try:
                        _swap = _psutil.Process().memory_swap()
                    except Exception:
                        pass
                    _mx_a = _mx_c = 0
                    try:
                        import mlx.core as _mx

                        _mx_a = _mx.get_active_memory()
                        _mx_c = _mx.get_cache_memory()
                    except Exception:
                        pass
                    with open(_MEM_LOG_PATH, "a") as _f:
                        _f.write(
                            f"{time.strftime('%H:%M:%S')} EMBED sub#{sub_idx} "
                            f"size={size} max_chars={len(texts[order[sub_start - 1]]) if idxs else 0} "
                            f"fp={(_mi.rss + _swap + _mx_a + _mx_c) / 1e9:.2f}GB rss={_mi.rss / 1e9:.2f}GB "
                            f"swap={_swap / 1e9:.2f}GB mx_active={_mx_a / 1e9:.2f}GB "
                            f"mx_cache={_mx_c / 1e9:.2f}GB\n"
                        )
                except Exception:
                    pass
            # RSS 软墙（#129，8GB）：内存偶尔冲高是正常波动，超墙只要求降档
            # 避让——每子批检查一次，降 1 档、激活变小、RSS 自然回落，回墙
            # 下即停。无闩锁（旧 5GB 闩锁版是「卡死在 1」的根因），恢复交给
            # 调优器自身的恢复分支 / 重爬。psutil 缺失时静默跳过。
            try:
                import psutil as _psutil

                if (
                    _psutil.Process().memory_info().rss / 1024**3
                    > _SUBBATCH_RSS_LIMIT
                ):
                    tuner.force_down()
            except Exception:
                pass
            # 周期性清理 MLX 缓存（见 _MLX_CACHE_CLEAR_EVERY 注释）：默认后端
            # 就是 mlx，而共享单例 EF 走不到 close()，这里是唯一能冲刷 Metal
            # buffer/graph 缓存的地方。
            if (self._embedding_backend == "mlx"
                    and sub_idx % _MLX_CACHE_CLEAR_EVERY == 0):
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except Exception:
                    pass
            # 主动释放 GIL，让 Qt 事件循环有机会处理排队的日志/进度信号。
            # 这是关键：ONNX Runtime 在 embed() 期间持有 GIL，只有在 Python
            # 层主动 sleep 时才释放。
            if _time.monotonic() - _last_yield >= _YIELD_INTERVAL:
                _time.sleep(0.01)
                _last_yield = _time.monotonic()
        vectors = vectors_by_idx
        records: List[Dict[str, Any]] = []
        # 批次级入库时间戳：向量去重按它保留最早入库的一份（见 _INGESTED_AT）。
        _now = time.time()
        # 一批 chunk 通常来源相同 → 缓存 source→hash 映射，避免每个 chunk
        # 都重算一次 md5（写入侧纯浪费）。
        _hash_cache: Dict[str, str] = {}
        for j, chunk in enumerate(chunks):
            idx = start_idx + j
            source = chunk.get("source", "unknown")
            chunk_idx = chunk.get("chunk_index", idx)
            h = _hash_cache.get(source)
            if h is None:
                h = _hash_source(source)
                _hash_cache[source] = h
            chunk_id = f"{h}_{chunk_idx}"
            records.append({
                "id": chunk_id,
                "text": chunk["text"],
                "vector": _normalize(vectors[j]),
                "source": source,
                "chunk_index": chunk_idx,
                _FTS_COLUMN: _tokenize_for_fts(chunk["text"]),
                _INGESTED_AT: _now,
            })
        return records

    # ------------------------------------------------------------------
    # BM25 full-text search (hybrid retrieval)
    # ------------------------------------------------------------------

    def _ensure_fts_index(self, table) -> bool:
        """Ensure a BM25 full-text index exists on the ``text_fts`` column.

        Creates the index on first use and after each ingest (the index must
        be refreshed to cover newly written rows).  The ``text_fts`` column is
        pre-tokenized with jieba (space-joined).  We use LanceDB's *native*
        FTS (Tantivy support was removed in lancedb 0.37); the native ``simple``
        tokenizer splits on whitespace, so each space-separated jieba token is
        indexed as its own term — sidestepping the lack of a compiled CJK
        tokenizer in the native FTS backend.

        Returns True if the index is available for querying.
        """
        if self._fts_ready:
            return True
        # 建索引失败过就不再重试：旧实现只把 _fts_checked 置 True，而
        # _fts_checked 全项目没有任何读取点 —— 于是每次检索都会重跑一遍
        # 昂贵的全库建索引，然后再次失败。
        if getattr(self, "_fts_failed", False):
            return False

        # 原生 FTS 配置（两种 lancedb 版本共用同一语义）：
        #  - simple base tokenizer 按空格/标点切分，配合 jieba 预分词列，
        #    每个空格分隔的 jieba token 即为一个独立 term；
        #  - ascii_folding=False 保留 CJK 码点；
        #  - stem=False / remove_stop_words=False 避免破坏中文词形。
        fts_kwargs = dict(
            base_tokenizer="simple",
            lower_case=True,
            stem=False,
            remove_stop_words=False,
            ascii_folding=False,
            with_position=True,
        )

        # lancedb >= 0.37：create_index(config=FTS(...))（Tantivy 已移除）。
        # lancedb 0.25.x：create_fts_index(use_tantivy=False, ...)。
        # 注意：ImportError 必须单独捕获——0.25.x 上根本没有 lancedb.index.FTS
        # 这个模块，裸 import 会抛 ImportError；若掉进通用 except Exception 分支，
        # _fts_failed 被置 True 后 0.25.x 的回退路径就永远走不到了（BM25 永久禁用）。
        try:
            from lancedb.index import FTS
        except ImportError:
            FTS = None
        try:
            if FTS is not None:
                table.create_index(
                    _FTS_COLUMN,
                    config=FTS(**fts_kwargs),
                    replace=True,
                )
            else:
                raise TypeError  # 走 0.25.x 回退分支
        except TypeError:
            # 0.25.x 的 create_index 无 config 参数（或 FTS 不可用），回退到
            # create_fts_index。
            table.create_fts_index(
                _FTS_COLUMN,
                use_tantivy=False,
                replace=True,
                **fts_kwargs,
            )
        except Exception as e:
            logger.warning("  ⚠ BM25 全文索引创建失败 (回退纯向量检索): %s", e)
            self._fts_checked = True
            self._fts_ready = False
            self._fts_failed = True
            return False

        self._fts_ready = True
        self._fts_checked = True
        logger.info("  ├─ BM25 全文索引已创建 (原生 FTS simple tokenizer + jieba 预分词)")
        return True

    @_store_op
    def search_fts(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Keyword (BM25) full-text search over the ``text_fts`` column.

        Returns list of {text, source, chunk_index, score, fts_score} dicts,
        where ``score`` is the BM25 score (higher is better).  Used together
        with vector search for hybrid retrieval.

        ``where`` is an optional SQL filter (e.g. ``source LIKE '%book.md'``)
        applied to the FTS search as well, so targeted retrieval restricts both
        channels consistently.
        """
        self._ensure_initialized()
        # 空查询直接短路：否则会白白触发一次全库 FTS 建索引 + 一次空查询。
        if not (query or "").strip():
            return []
        table = self._ensure_table()
        if table is None:
            return []

        if not self._ensure_fts_index(table):
            return []

        try:
            # Tokenize the query the same way as the indexed text.
            tokenized_query = _tokenize_for_fts(query)
            q = (
                table.search(tokenized_query, query_type="fts", fts_columns=[_FTS_COLUMN])
            )
            if where:
                q = q.where(where)
            results = q.limit(top_k).to_list()
        except Exception as e:
            logger.warning("  ⚠ BM25 全文检索失败: %s", e)
            return []

        output: List[Dict[str, Any]] = []
        for r in results:
            # Tantivy/原生 FTS 返回的 BM25 分在 `_score`（越大越好），是无上界
            # 的原始分（常见 5~30+）。绝不能把它直接塞进 `score`：
            # `score` 在有向量候选的场合被当作 [0,1] 余弦相似度使用，
            # _rerank_fallback 会对其做 `1 - s²/2` 换算 —— BM25=15 会被算成
            # 1-112.5 → clamp 到 0.0，纯关键词候选集体排到末尾，混合检索
            # 在 reranker 不可用时静默退化成纯向量检索。
            # 这里用 s/(1+s) 归一化到 [0,1)，原始分另存 fts_score。
            raw = float(r.get("_score", 0.0))
            norm = raw / (1.0 + raw) if raw > 0 else 0.0
            output.append({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": norm,
                "fts_score": raw,
            })
        return output

    @_store_op
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the vector store for relevant chunks.

        Returns list of {text, source, chunk_index, score} dicts.
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            logger.info("  向量检索: 向量表不存在 (查询=%r)", query[:60])
            return []
        row_count = table.count_rows()
        if row_count == 0:
            logger.info("  向量检索: 向量表为空 (查询=%r)", query[:60])
            return []

        # Embed query (use embed_query for proper prefix handling)
        logger.info("  ├─ 查询向量化: 将查询文本转为向量...")
        if hasattr(self._ef, 'embed_query'):
            query_vec = _normalize(self._ef.embed_query(query))
        else:
            query_vec = _normalize(self._ef([query])[0])

        # Vector search
        logger.info("  ├─ 向量相似度搜索: 在 %d 条向量中检索 top %d...", row_count, top_k)
        search = table.search(query_vec).limit(top_k)
        if where:
            # prefilter=True：先按条件过滤再做 ANN。默认的后置过滤是「先取
            # ANN top-k 再过滤」，定向检索（rag_search(source=...)）下候选
            # 会被先截断再筛，可能漏召回甚至返回空。
            try:
                search = search.where(where, prefilter=True)
            except TypeError:
                # 旧版 lancedb 的 where() 不接受 prefilter 关键字。
                search = search.where(where)
        results = search.to_list()
        logger.info("  ├─ 向量检索完成: 获得 %d 个候选切片", len(results))

        output: List[Dict[str, Any]] = []
        for r in results:
            # LanceDB 返回 _distance（L2 距离，越小越相似）。
            # 转换为相似度分数 [0,1]（越高越相似），统一下游接口语义。
            if "_distance" in r:
                distance = float(r["_distance"])
                # 对归一化向量，L2 距离 d ∈ [0,2]，余弦相似度 cos = 1 - d²/2
                similarity = max(0.0, 1.0 - (distance * distance) / 2.0)
            else:
                # _distance 字段缺失（罕见：表 schema 不兼容或旧表迁移）。
                # 使用中等分数避免将无效结果排到最前或最后。
                logger.debug("VectorStore.search: _distance field missing for result, using default score 0.5")
                similarity = 0.5
            output.append({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "score": similarity,
            })

        return output

    # ------------------------------------------------------------------
    # reranking with BGE
    # ------------------------------------------------------------------

    def _get_reranker(self):
        """Lazy-load the BGE reranker model with timeout protection.

        Returns the reranker instance, or None if loading fails (network
        timeout, missing dependency, etc.). Callers must handle None by
        falling back to distance-based ranking.

        Uses ``_reranker_state`` to track loading state so concurrent callers
        don't each spawn their own load thread (which would leak ~500MB of
        PyTorch weights per extra thread).  On timeout, the background thread
        is NOT abandoned — when it eventually finishes it stores the result
        (or error) on ``self`` so a later call can pick it up without reloading.
        """
        # close() 之后的实例不允许再拉起 reranker：后台加载线程看到 _closed
        # 会直接 return（状态置 failed），避免状态机卡死或「复活」持有权重。
        if getattr(self, "_closed", False):
            return None
        if self._rerank_backend == "disabled":
            self._reranker = None
            self._reranker_state = "failed"
            return None
        state = getattr(self, "_reranker_state", "idle")
        if state == "done":
            return self._reranker
        if state in ("failed", "loading"):
            # failed → 不可用；loading → 另一个线程已在加载，本次调用回退到
            # 距离排序，加载完成后的调用自然能拿到模型。
            return None

        # 状态转换必须串行：import fastembed + 校验模型缓存可耗时数秒，
        # 旧实现从「读到 idle」到「写入 loading」之间没有任何锁，两个并发
        # 调用方都会各自起一个加载线程——各加载一份数百 MB 模型，后写者覆盖
        # 前者，先建的那份成为纯泄漏。
        with self._reranker_lock:
            state = getattr(self, "_reranker_state", "idle")
            if state == "done":
                return self._reranker
            if state in ("failed", "loading"):
                return None

            from .model_backends import create_reranker_backend

            logger.info(
                "  🔧 正在加载 Reranker: backend=%s model=%s path=%s...",
                self._rerank_backend,
                self._rerank_model,
                self._rerank_model_path or "(catalog)",
            )
            import time as _time
            _time.sleep(0.01)  # 释放 GIL 让 UI 更新

            def _load():
                try:
                    reranker = create_reranker_backend(
                        self._rerank_backend,
                        self._rerank_model,
                        self._rerank_model_path,
                        threads=self._rerank_threads,
                        max_length=self._rerank_max_length,
                        batch_size=self._rerank_batch_size,
                        mlx_model_path=self._rerank_mlx_model_path,
                        jina_model_path=self._rerank_jina_model_path,
                    )
                    if reranker is None:
                        self._reranker = None
                        self._reranker_state = "failed"
                        return
                    if getattr(self, "_closed", False):
                        # close() 已经执行：不要再把数百 MB 模型写回实例
                        # （旧实现会让 close 之后的实例「复活」持有权重）。
                        try:
                            del reranker
                        except UnboundLocalError:
                            pass
                        # 必须把状态机从 "loading" 里救出来：旧实现直接 return，
                        # 状态永远停在 loading → 之后所有 rerank() 都命中
                        # loading 分支永久返回 None，重排序静默失效且无日志。
                        self._reranker_state = "failed"
                        return
                    self._reranker = reranker
                    self._reranker_state = "done"
                    logger.info(
                        "  ✅ Reranker 加载完成: backend=%s model=%s",
                        self._rerank_backend, self._rerank_model,
                    )
                except Exception as e:
                    logger.warning("  ⚠ BGE Reranker 加载失败: %s", e)
                    self._reranker = None
                    self._reranker_state = "failed"

            # 先置 loading 再启动线程，并发的其他调用方看到 loading 即返回。
            self._reranker_state = "loading"
            t = threading.Thread(target=_load, daemon=True)
            t.start()

        # join 放在锁外：避免其他调用方为等这个 join 阻塞 2 分钟。
        try:
            t.join(timeout=120)  # 2 分钟超时，足够下载 500MB
            if t.is_alive():
                # 加载仍在后台进行（网络慢）。不丢弃结果——后台线程完成时会
                # 写入 self._reranker/_reranker_state，后续调用直接复用。
                logger.warning(
                    "  ⚠ BGE Reranker 加载超时 (2 分钟) — 仍在后台下载中。"
                    " 本次重排序回退到向量距离排序；加载完成后自动启用。"
                )
        except Exception as e:
            logger.warning("  ⚠ BGE Reranker 加载失败: %s", e)
            self._reranker = None
            self._reranker_state = "failed"

        return self._reranker if getattr(self, "_reranker_state", "idle") == "done" else None

    def _rerank_fallback(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fall back to vector-distance ranking when BGE reranker is unavailable.

        candidates 的 score 已由 search() 转换为相似度 [0,1]（越高越好）。
        回退模式下分数是余弦距离的近似（非 BGE sigmoid 语义分），因此不做
        0.5 阈值过滤——阈值仅对 reranker 的语义分数有意义。
        """
        logger.info("  ├─ 重排序: BGE Reranker 不可用, 使用向量距离排序 (回退模式)")
        for c in candidates:
            try:
                s = float(c.get("score", 0.0))
            except (TypeError, ValueError):
                c["score"] = 0.0
                continue
            if s > 1.0:
                c["score"] = max(0.0, 1.0 - (s * s) / 2.0)
            else:
                c["score"] = s
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        result = candidates[:top_k]
        logger.info("  ├─ 重排序完成(回退): %d → %d 条结果", len(candidates), len(result))
        return result

    @_store_op
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Re-rank candidates using BGE reranker for precise scoring.

        Falls back to distance-based similarity scoring when the reranker
        is unavailable. In both cases scores are normalized to [0, 1]
        (higher is better) and results are returned sorted descending.

        Results whose final score is below ``min_similarity`` are dropped
        (they are not semantically relevant enough to send to the LLM), so
        the returned list may be shorter than ``top_k``.
        """
        if not candidates:
            return []

        # 使用注入的自定义 embedding（如测试的 FakeEmbeddingFunction）时，
        # 向量空间与 BGE reranker 的语义空间不一致，rerank 打分无意义，
        # 直接回退到向量距离排序。
        if self._custom_ef is not None:
            return self._rerank_fallback(candidates, top_k)

        reranker = self._get_reranker()
        if reranker is None:
            return self._rerank_fallback(candidates, top_k)

        # Build (query, doc) pairs for the cross-encoder.
        logger.info("  ├─ BGE Reranker 精排: 对 %d 个候选进行交叉编码打分...", len(candidates))
        texts = [c["text"] for c in candidates]
        try:
            # fastembed 的 rerank() 返回模型原始 logit（未过 sigmoid）；
            # 手动 sigmoid 归一化到 [0,1]，保持与旧 FlagReranker
            # compute_score(normalize=True) 相同的分数语义——下游
            # min_similarity 阈值过滤依赖这个 [0,1] 区间。
            raw_scores = reranker.rerank(
                query, texts, batch_size=self._rerank_batch_size
            )
            scores = [_logit_to_score(s) for s in raw_scores]
        except Exception as e:
            # 模型/输入异常（超长序列截断失败等）时回退到向量距离排序，
            # 避免整个检索流程崩溃。必须同步置 failed，否则
            # _reranker_state 长期停在 "done" 而打分持续失败（状态机失真）。
            # 加锁写：与 _get_reranker 的状态读写并发属于数据竞争。
            logger.warning(
                "  ⚠ BGE Reranker 打分失败, 回退到向量距离排序: %s", e,
            )
            with self._reranker_lock:
                self._reranker = None
                self._reranker_state = "failed"
            return self._rerank_fallback(candidates, top_k)

        # Normalize to list if single result
        if not isinstance(scores, list):
            scores = [scores]

        # Attach rerank scores and sort (higher is better).
        # 用 zip 而不是按索引取：scores 与 candidates 长度不一致时（服务端
        # 返回了意外条数）enumerate 版本会 IndexError 崩掉整个检索。
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Return top_k, mapping rerank_score to score for downstream
        results = candidates[:top_k]
        for r in results:
            r["score"] = r.get("rerank_score", r.get("score", 0))

        # 输出重排序前后的分数变化
        if results:
            logger.info("  ├─ BGE 重排序完成: %d → %d 条结果", len(candidates), len(results))
            for i, r in enumerate(results[:3]):
                source = os.path.basename(r.get("source", ""))
                logger.info("  │   [%d] %s (重排序分: %.4f)", i + 1, source, r["score"])

        return self._filter_by_similarity(results, min_similarity)

    def _filter_by_similarity(
        self,
        results: List[Dict[str, Any]],
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        """Drop results whose final score is below ``min_similarity``.

        A rerank score (BGE sigmoid, or cosine-similarity fallback) below the
        threshold means the chunk is not semantically close enough to the
        query to be worth sending to the LLM.
        """
        if not results or min_similarity <= 0:
            return results
        kept = [r for r in results if r.get("score", 0.0) >= min_similarity]
        if len(kept) < len(results):
            dropped = len(results) - len(kept)
            logger.info(
                "  ├─ 相似度过滤: 丢弃 %d 条 < %.2f 的结果, 保留 %d 条",
                dropped, min_similarity, len(kept),
            )
        return kept

    @_store_op
    def search_with_rerank(
        self,
        query: str,
        top_k: int = 3,
        candidate_multiplier: int = 4,
        where: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval: vector similarity + BM25 keyword, then BGE rerank.

        Pipeline:
          1. Vector search: top_k * candidate_multiplier candidates.
          2. BM25 keyword search: top_k * candidate_multiplier candidates.
          3. Merge & dedupe by (source, chunk_index).
          4. BGE rerank the merged candidate set.
          5. Drop any result whose rerank score < min_similarity (0.5).
          6. Return up to top_k results.

        Args:
            query: Search query.
            top_k: Final number of results to return.
            candidate_multiplier: Candidates per retrieval channel = top_k × N.
            where: Optional SQL filter (applied to vector search only).
            min_similarity: Reject rerank results below this score.
        """
        n_candidates = top_k * candidate_multiplier

        # 1. Vector similarity search.
        vec_candidates = self.search(query, top_k=n_candidates, where=where)
        logger.info(
            "VectorStore.search_with_rerank: 向量检索 %d 条候选", len(vec_candidates),
        )

        # 2. BM25 keyword search (同样的 where 过滤，保证定向检索一致)。
        fts_candidates = self.search_fts(query, top_k=n_candidates, where=where)
        logger.info(
            "VectorStore.search_with_rerank: BM25 检索 %d 条候选", len(fts_candidates),
        )

        # 3. Merge & dedupe by (source, chunk_index), keeping vector candidate
        #    first (its score is a cosine similarity in [0,1]).
        merged: List[Dict[str, Any]] = []
        seen: set = set()
        for c in vec_candidates + fts_candidates:
            key = (c.get("source", ""), c.get("chunk_index", 0))
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)

        if not merged:
            logger.info("VectorStore.search_with_rerank: 混合检索无候选")
            return []

        logger.info(
            "VectorStore.search_with_rerank: 混合候选 %d 条 (向量 %d + BM25 %d, 去重后 %d)",
            len(merged), len(vec_candidates), len(fts_candidates), len(merged),
        )

        # 4. BGE rerank + 5. min_similarity filter.
        results = self.rerank(query, merged, top_k=top_k, min_similarity=min_similarity)
        logger.info(
            "VectorStore.search_with_rerank: %d 候选 → %d 条 (rerank + 相似度≥%.2f 过滤)",
            len(merged), len(results), min_similarity,
        )

        # 6. 当 rerank 严格过滤后无结果（例如查询词质量差——拼音、口语化、
        #    噪声词等导致 rerank 语义分普遍低于阈值）时，回退到向量距离排序，
        #    至少返回 top_k 条候选，避免 LLM 完全拿不到上下文。
        if not results and merged:
            logger.info(
                "VectorStore.search_with_rerank: rerank 过滤后无结果, 回退到向量距离排序",
            )
            results = self._rerank_fallback(merged, top_k)
        return results

    def _clear_locked(self) -> None:
        """Delete all documents from the store.

        调用方必须已排空在途操作（见 ``clear``）：drop_table 与在途的
        add/search 并发，会让后者打到一张已被删除的表上。
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return
        count = table.count_rows()
        if count > 0:
            # Drop and recreate empty table
            self._db.drop_table(_TABLE_NAME)
            self._table = None
            self._table_checked = False  # 重置标记，允许下次 add 重新创建表
            # 必须同时重置 FTS 三标记（与 close() 同款）：漏掉的话 _fts_ready
            # 仍为 True，add() 重建新表时会跳过建索引，search_fts 也认为索引
            # 就绪 → 在未索引列上跑 FTS 抛异常被吞掉 → BM25 通道从此永久
            # 静默返回空，无任何告警。
            self._fts_ready = False
            self._fts_checked = False
            self._fts_failed = False
            logger.info("VectorStore: cleared %d documents", count)
        # 同步清空文档元数据表。
        if _DOCUMENTS_TABLE_NAME in self._db.table_names():
            self._db.drop_table(_DOCUMENTS_TABLE_NAME)
        # documents 表**不存在**时也要清缓存/句柄：否则残留的旧缓存会让
        # get_document_digest 继续返回已删除文档的 digest。清缓存必须与
        # _load_documents 一样持 _documents_lock。
        with self._documents_lock:
            self._documents_table = None
            self._documents_cache = None
        logger.info("VectorStore: cleared documents metadata table")

    def clear(self) -> None:
        """清空向量库与文档元数据。

        真正实现在 ``_clear_locked``：``clear`` 需要在拿到在途操作排空权之后
        再动手——drop_table 与在途的 add/search 并发会让后者打到已删除的表上。
        注意这里**不能**加 ``@_store_op``：它会先把自己登记为在途操作，随后
        ``_drain_ops()`` 就会等自己结束（自锁）。
        """
        self._drain_ops()
        try:
            self._clear_locked()
        finally:
            self._resume_ops()

    @_store_op
    def count(self) -> int:
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return 0
        return table.count_rows()

    def close(self) -> None:
        """Release heavy resources (LanceDB connection, embedding model, reranker).

        After close(), the VectorStore cannot be used; create a new instance to
        access the same data (LanceDB persists to disk, so data survives).

        Important: FastEmbed's embedding and reranker ONNX Runtime sessions
        hold significant C++ heap memory (~130MB + ~570MB). Python's GC may not
        reclaim these promptly when the Python object is dropped — explicitly
        nulling the references here ensures the C++ destructors run as soon
        as possible, which is critical when SyncKnowledgeWorker creates a fresh
        RAGEngine per sync run.

        Waits for in-flight ``add``/``search`` operations to finish first:
        tearing down the LanceDB connection underneath them produces random
        AttributeErrors (``self._db`` is None) or silently wrong results.
        """
        # Try to explicitly release the ONNX InferenceSession held by
        # FastEmbed's OnnxEmbeddings. FastEmbed wraps the model in a
        # `model` attribute (ort.InferenceSession). Calling .release() on
        # the session frees the C++ memory immediately instead of waiting
        # for Python's cyclic GC to eventually notice the dead reference.
        # This is best-effort: if the attribute layout changes in a future
        # FastEmbed version, the AttributeError is swallowed and we fall
        # back to simply dropping the Python reference.
        # 先排空并**保持**排空态：若此时正有 add/search 在跑，它们会撞上被置
        # None 的 _db 或已被 drop 的表。这里不能像 clear() 那样 _resume_ops()——
        # close() 之后本实例永久不可用，_closing 保持 True 正好与
        # _begin_op() 的检查一致（close 后新操作一律被拒）。
        self._drain_ops()
        # 先标记关闭：后台 reranker 加载线程完成时看到 _closed 就不再把模型
        # 写回本实例（否则 close 之后的实例会「复活」持有 ~500MB 权重）。
        self._closed = True
        self._table = None
        self._db = None
        self._documents_cache = None
        if self._ef is not None:
            # Only release the ONNX session if this VectorStore owns a
            # private (non-shared) embedding function.  When using the
            # process-wide singleton (get_or_create_embedding_function),
            # the session may still be in use by graph_memory or another
            # VectorStore — releasing it would crash those callers.
            # **必须用 identity 判断**：`in` 走 `==`，会调用 EF 对象自定义的
            # __eq__（pydantic/自定义后端可能又重又会抛）。一旦比较抛异常，
            # 共享单例会被误判成「私有」→ 走到下面 session.release()，把
            # graph_memory / 其它 VectorStore 正在使用的模型砸掉。
            is_shared = any(v is self._ef for v in _EF_SINGLETONS.values())
            if not is_shared:
                try:
                    inner = getattr(self._ef, "model", None) or getattr(self._ef, "_model", None)
                    if inner is not None:
                        session = getattr(inner, "session", None) or getattr(inner, "_session", None)
                        if session is not None and hasattr(session, "release"):
                            try:
                                session.release()
                            except Exception:
                                pass
                except Exception:
                    pass
                # 一并调用 EF 自己的 close()：MLX 后端在此执行 mx.clear_cache()，
                # 单纯 self._ef = None 是不会触发的。私有（非共享）实例 close
                # 是安全的——共享单例必须跳过，否则会砸掉 graph_memory /
                # 其它 VectorStore 正在用的模型。
                try:
                    _ef_close = getattr(self._ef, "close", None)
                    if callable(_ef_close):
                        _ef_close()
                except Exception:
                    pass
            self._ef = None
        if hasattr(self, "_reranker"):
            # TextCrossEncoder 底层是 onnxruntime InferenceSession（纯 ONNX，
            # 无 PyTorch/transformers 权重），Python 对象置 None 后 C++ 侧
            # session 随析构释放，无需 CPU 迁移步骤。
            reranker = self._reranker
            try:
                close_fn = getattr(reranker, "close", None)
                if callable(close_fn):
                    close_fn()
                else:
                    # fastembed 旧版实例带 .model 子对象；统一丢弃引用即可。
                    _ = getattr(reranker, "model", None)
            except Exception:
                pass
            self._reranker = None
        self._reranker_state = "idle"
        self._initialized = False
        self._table_checked = False
        # 表句柄已丢弃，重连后打开的是「另一张」表（可能被 clear()/迁移重建
        # 过）。不重置 FTS 标记的话，_ensure_fts_index() 会直接返回 True，
        # 于是 BM25 检索静默失效（新表上没有索引）。
        self._fts_ready = False
        self._fts_checked = False
        # close() 后可以重新 open：失败标记必须一并复位，否则新建的实例会
        # 永久跳过全文索引。
        self._fts_failed = False
        logger.info("VectorStore: resources released")

    def reload(self) -> None:
        """Drop the cached table handle so the next operation re-opens it.

        LanceDB table objects hold a version snapshot at open time. When another
        VectorStore instance (e.g. SyncKnowledgeWorker's) writes new rows to the
        same table, this instance's cached table object won't see them —
        count_rows() and search() still return the old snapshot's data.

        Call this after a sync completes to ensure the agent's VectorStore sees
        the freshly written data on its next search.

        与 clear()/close() 一样先排空在途操作：``_table = None`` 与在途的
        search() 并发时，后者可能正持有旧句柄做查询，行为未定义。
        """
        self._drain_ops()
        try:
            self._reload_locked()
        finally:
            self._resume_ops()

    def _reload_locked(self) -> None:
        """``reload()`` 的实现，调用方必须已排空在途操作。"""
        self._table = None
        self._table_checked = False
        # documents 元数据表与其缓存同样持有打开时的快照：同步新增/更新了
        # 文档后，agent 侧的 rag_outline 必须丢弃旧句柄与旧缓存才能看到
        # 新 digest；FTS 标记同理，否则新写入的切片在 BM25 通道不可见。
        with self._documents_lock:
            self._documents_table = None
            self._documents_cache = None
        self._fts_ready = False
        self._fts_checked = False
        self._fts_failed = False
        logger.debug("VectorStore: table handles dropped for reload")

    @_store_op
    def list_sources(self) -> List[str]:
        """Return unique source filenames in the store.

        性能关键：**不要**用 ``table.to_arrow(columns=["source"])`` 做列投影。
        本地 LanceDB 版本不接受 ``columns`` 参数，会直接抛 TypeError，随后
        兜底的 ``table.to_arrow().to_pylist()`` 会把整表（57 万行 × 768 维
        向量 + 全文）读进 Python，实测单次 200+ 秒、约 2GB 内存。而
        ``status()`` 每次都要调用它，于是表现为「每句消息都卡几分钟」。

        改用查询构造器的 ``select([...])``——它把列投影下推到 Lance 存储层，
        只读 ``source`` 列（实测 57 万行约 0.6 秒）。
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return []
        try:
            count = table.count_rows()
        except Exception:
            count = 0
        if count == 0:
            return []
        try:
            import pyarrow.compute as pc
            # 投影下推：只读 source 列，避免把向量列一起读出来。
            arrow_tbl = table.search().select(["source"]).limit(count).to_arrow()
            unique_sources = pc.unique(arrow_tbl["source"]).to_pylist()
            return sorted([s for s in unique_sources if s])
        except Exception as e:
            logger.debug("list_sources 投影查询失败，回退全表读: %s", e)
        # 兜底：全表读（会连同向量一起读，很慢——仅当投影路径不可用时）。
        try:
            all_data = table.to_arrow().to_pylist()
            return sorted({r.get("source", "") for r in all_data if r and r.get("source")})
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 向量去重（UI「向量去重」按钮 / engine.dedupe_vectors 用）
    # ------------------------------------------------------------------

    @_store_op
    def scan_duplicate_texts(self) -> Dict[str, Any]:
        """全表按 ``text`` **精确**分组，找出重复向量组。

        返回 ``{"total": 行数, "groups": [...], "source_counts": {source: n}}``；
        每组 ``{"text": 样例, "keep": {...}, "drop": [{...}, ...]}``，条目含
        ``id/source/ingested_at``。``keep`` = 组内 ``ingested_at`` 最早的一份
        （缺失视为 0.0 = 最老，历史数据占先）；并列时取 source 字典序小者，
        保证确定性。

        text 完全一致才算重复——**不做归一化**，避免把「相似但不同」的文本误
        并组。整表一次性读入（投影 id/text/source/ingested_at，不含向量），
        大库（数十万行）内存与耗时都可接受；这是用户手动触发的维护操作，
        不是同步热路径。
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None:
            return {"total": 0, "groups": [], "source_counts": {}}
        try:
            count = table.count_rows()
        except Exception:
            count = 0
        if count == 0:
            return {"total": 0, "groups": [], "source_counts": {}}
        # 列投影必须走查询构造器 select([...])——**不能**用
        # to_arrow(columns=[...])：lancedb 0.40 的 LanceTable.to_arrow 不接受
        # columns kwarg（TypeError，#160 同一坑；list_sources 因此改走 select）。
        # limit(count) 取全表（与 list_sources 同模式）。
        try:
            arrow = (table.search()
                     .select(["id", "text", "source", _INGESTED_AT])
                     .limit(count).to_arrow())
        except Exception as e:
            # 旧表没有 ingested_at 列（迁移未跑/失败）：退化为不按时间保留，
            # 组内按 (source, id) 字典序取第一份——仍然确定，只是「最早」语义
            # 退化为「字典序最早」。select 缺列抛 ValueError，这里接住。
            logger.debug("scan_duplicate_texts: %s 列投影失败，退化分组: %s",
                         _INGESTED_AT, e)
            arrow = (table.search()
                     .select(["id", "text", "source"])
                     .limit(count).to_arrow())
        rows = arrow.to_pylist()
        has_ts = _INGESTED_AT in arrow.schema.names
        by_text: Dict[str, List[Dict[str, Any]]] = {}
        source_counts: Dict[str, int] = {}
        for r in rows:
            src = r.get("source") or ""
            source_counts[src] = source_counts.get(src, 0) + 1
            ts = (r.get(_INGESTED_AT) if has_ts else None)
            if ts is None:
                ts = 0.0
            by_text.setdefault(r.get("text") or "", []).append(
                {"id": r.get("id"), "source": src, "ingested_at": ts})
        groups = []
        for text, items in by_text.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: (x["ingested_at"], x["source"], x["id"] or ""))
            groups.append({"text": text, "keep": items[0], "drop": items[1:]})
        # 可清理量大的组排前面：进度/报告更直观
        groups.sort(key=lambda g: -len(g["drop"]))
        return {"total": len(rows), "groups": groups, "source_counts": source_counts}

    @_store_op
    def delete_by_ids(self, ids: List[str], batch: int = 400) -> int:
        """按 id 批量删除向量（去重用）。返回成功提交删除的条数。

        id 形如 ``{md5hex}_{chunk_index}``（hex + 下划线 + 数字），无引号
        注入面；仍做单引号转义防御。分批 IN 删除，避免单条 SQL 过长。
        """
        self._ensure_initialized()
        table = self._ensure_table()
        if table is None or not ids:
            return 0
        deleted = 0
        id_list = [str(x) for x in ids]
        for i in range(0, len(id_list), batch):
            chunk = id_list[i:i + batch]
            values = ", ".join("'" + x.replace("'", "''") + "'" for x in chunk)
            try:
                table.delete(f"id IN ({values})")
                deleted += len(chunk)
            except Exception as e:
                logger.error("VectorStore: delete_by_ids batch %d/%d failed: %s",
                             i // batch + 1,
                             (len(id_list) + batch - 1) // batch, e)
        return deleted

    # ------------------------------------------------------------------
    # documents metadata table (Meta-context + Targeted RAG)
    # ------------------------------------------------------------------

    def _ensure_documents_table(self):
        """Open or create the documents metadata table (thread-safe).

        用双重检查锁保护：ingest 的多 worker 并行调用 upsert_document 时，
        首次创建表会并发竞争——若不加锁，第二个 worker 会在 create_table 时报
        "Table already exists"。锁内再次检查后，创建或打开由唯一一个线程完成。

        注意：调用方（upsert_document / delete_document）通常已经持有
        ``_documents_lock``。锁现在是 RLock，所以这里再次 acquire 是安全的；
        但真正的首次建表路径过去是 **死锁**：upsert_document 持锁 →
        调用本函数 → 再次 acquire 同一个非可重入 Lock → 永久阻塞。第一次同步
        （documents 表尚不存在）会卡死整个 ingest，UI 一直停在「同步中」。
        """
        if self._documents_table is not None:
            return self._documents_table
        with self._documents_lock:
            return self._ensure_documents_table_locked()

    def _ensure_documents_table_locked(self):
        """``_ensure_documents_table`` 的实现，调用方必须已持有 _documents_lock。"""
        if self._documents_table is not None:
            return self._documents_table
        self._ensure_initialized()
        if _DOCUMENTS_TABLE_NAME in self._db.table_names():
            self._documents_table = self._db.open_table(_DOCUMENTS_TABLE_NAME)
        else:
            # 首次创建：空表（含占位空记录，后续 upsert 时清理）。
            # 注意：不建 markdown 列——全文已落在 rag/documents/*.md 缓存里，
            # 表内再存一份是纯浪费（且会被 _documents_cache 全量载入内存）。
            self._documents_table = self._db.create_table(
                _DOCUMENTS_TABLE_NAME,
                data=[{
                    "doc_id": "",
                    "doc_name": "",
                    "digest": "",
                    "chapters": "",
                    "md_path": "",
                }],
            )
        return self._documents_table

    @_store_op
    def upsert_document(
        self,
        doc_name: str,
        digest: str,
        markdown: str = "",
        chapters: str = "",
        md_path: str = "",
    ) -> None:
        """Store or update a document's digest (缩略版本) + chapter list.

        ``doc_id`` is a stable md5 of ``doc_name`` so re-ingesting the same
        document overwrites the previous entry (upsert semantics).

        The full markdown is **not** stored in this table: it already lives on
        disk in ``rag/documents/*.md`` (the ingest cache), and no consumer
        reads it from here — only ``digest`` is used (by ``rag_outline``).
        Keeping it would duplicate the whole corpus a third time on disk and
        load it entirely into memory via ``_documents_cache``.  We persist
        ``md_path`` instead so a consumer can read it on demand.

        The delete+add pair is serialized under ``_documents_lock`` because
        ingest workers call this concurrently (one per file). LanceDB table
        writes are not thread-safe, and interleaving delete/add from multiple
        threads could corrupt rows or raise "Table already exists"-style races.
        """
        self._ensure_initialized()
        with self._documents_lock:
            # 已在锁内 → 用不重复加锁的版本，避免嵌套 acquire。
            table = self._ensure_documents_table_locked()
            # doc_id 必须唯一：旧实现用 basename/doc_name 生成，不同目录下的
            # 同名文件（dir1/report.pdf 与 dir2/report.pdf）共用同一 doc_id，
            # 互相覆盖；删除其中一个时另一个的 digest 也被一并删掉。md_path
            # 是每篇文档唯一的缓存路径，优先用它做主键。
            doc_id = _hash_source(md_path) if md_path else _hash_source(doc_name)

            # Delete any previous entry (and the placeholder row), then add the new one.
            try:
                table.delete(f"doc_id = '{doc_id}'")
                # 清理旧版（doc_id 用 basename 生成）的同名残留行，避免升级后
                # documents 表出现新旧两份记录。旧记录本身已因 basename 碰撞
                # 互相覆盖，最多只剩一条，删掉无损失。
                table.delete(f"doc_id = '{_hash_source(os.path.basename(doc_name))}'")
                table.delete("doc_id = ''")
            except Exception:
                pass
            table.add([{
                "doc_id": doc_id,
                "doc_name": doc_name,
                "digest": digest,
                "chapters": chapters,
                "md_path": md_path,
            }])
            logger.info("  ├─ 文档元数据已写入: %s (digest %d 字符)", doc_name, len(digest))
            # 写入后失效缓存，下次读取时重新加载。
            self._documents_cache = None

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load (and cache) all non-placeholder document rows.

        Returns a list of dicts, filtered to drop the placeholder row
        (``doc_name`` empty).  Cached in ``_documents_cache``.

        Only the lightweight columns are read — the (now unused) ``markdown``
        column must never be pulled into the cache, otherwise the whole corpus
        is duplicated in memory for the process lifetime.
        """
        with self._documents_lock:
            if self._documents_cache is not None:
                return self._documents_cache
            table = self._ensure_documents_table_locked()
            if table is None:
                return []
            try:
                cols = ["doc_id", "doc_name", "digest", "chapters", "md_path"]
                try:
                    # 同 list_sources：to_arrow() 不接受 columns 参数，用查询
                    # 构造器的 select() 做投影下推，避免把整表（含完整 markdown）
                    # 读进内存。
                    n = table.count_rows()
                    rows = table.search().select(cols).limit(n).to_arrow().to_pylist()
                except Exception:
                    # 旧表没有 md_path 列 / 投影不支持 → 退化为全列读取，
                    # 但随后会 pop 掉 markdown，绝不长期缓存。
                    rows = table.to_arrow().to_pylist()
            except Exception as e:
                logger.warning("读取文档元数据失败: %s", e)
                return []
            rows = [r for r in rows if r.get("doc_name")]
            for r in rows:
                r.pop("markdown", None)  # 兼容旧数据：绝不缓存 markdown
            self._documents_cache = rows
            return rows

    @_store_op
    def get_document_digest(self, doc_name: str) -> Optional[Dict[str, Any]]:
        """Return {doc_name, digest, markdown, chapters} for a document by name.

        Matches by exact basename or by substring (so the model can reference
        a book by a partial title).  Returns None if not found.
        """
        self._ensure_initialized()
        rows = self._load_documents()
        if not rows:
            return None

        # 1. Exact match (case-insensitive) on doc_name or basename.
        target = doc_name.strip().lower()
        for r in rows:
            name = (r.get("doc_name") or "").lower()
            if name == target or os.path.basename(name) == target:
                return {
                    "doc_name": r.get("doc_name", ""),
                    "digest": r.get("digest", ""),
                    "chapters": r.get("chapters", ""),
                    "md_path": r.get("md_path", ""),
                }
        # 2. Substring match (partial title).
        for r in rows:
            name = (r.get("doc_name") or "").lower()
            if target and (target in name or name in target):
                return {
                    "doc_name": r.get("doc_name", ""),
                    "digest": r.get("digest", ""),
                    "chapters": r.get("chapters", ""),
                    "md_path": r.get("md_path", ""),
                }
        return None

    @_store_op
    def list_document_records(self) -> List[Dict[str, Any]]:
        """返回 documents 表全部记录（轻量列）。

        供批量预检使用：旧实现每处理一个缓存文件就调一次按名查询——而每次
        upsert 都会作废 _documents_cache，导致 O(M·N) 的重复全表加载。
        调用方应一次性取快照后在内存中匹配。
        """
        self._ensure_initialized()
        return list(self._load_documents())

    @_store_op
    def get_document_by_md_path(self, md_path: str) -> Optional[Dict[str, Any]]:
        """按 ``md_path``（写库时落下的权威键）定位文档元数据。

        缓存文件名经过 ``_safe_filename`` 改写（删除空格、截断到 32 字符），
        无法可靠地反推出入库时用的 ``doc_name``；``md_path`` 是原值，用它
        匹配才准。返回 None 表示还没有对应记录。
        """
        self._ensure_initialized()
        target = (md_path or "").strip()
        if not target:
            return None
        for r in self._load_documents():
            if (r.get("md_path") or "").strip() == target:
                return {
                    "doc_name": r.get("doc_name", ""),
                    "digest": r.get("digest", ""),
                    "chapters": r.get("chapters", ""),
                    "md_path": r.get("md_path", ""),
                }
        return None

    @_store_op
    def list_documents(self) -> List[str]:
        """Return the list of stored document names (for the outline tool)."""
        self._ensure_initialized()
        rows = self._load_documents()
        return sorted({r.get("doc_name", "") for r in rows if r.get("doc_name")})

    @_store_op
    def delete_document(self, doc_name: str) -> None:
        """Remove a document's metadata entry.

        同时按 doc_name 列精确删除（兼容 doc_id 主键从 basename 切换到
        md_path 之前的旧版记录——它们的 doc_id 是 basename 的哈希）。
        """
        self._ensure_initialized()
        with self._documents_lock:
            table = self._ensure_documents_table_locked()
            if table is None:
                return
            doc_id = _hash_source(doc_name)
            safe_name = (doc_name or "").replace("'", "''")
            try:
                table.delete(f"doc_id = '{doc_id}'")
                if safe_name:
                    table.delete(f"doc_name = '{safe_name}'")
            except Exception as e:
                logger.warning("删除文档元数据失败: %s", e)
            self._documents_cache = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_source(source: str) -> str:
    """Create a short stable hash for a source path."""
    import hashlib
    return hashlib.md5(source.encode()).hexdigest()[:12]


def _normalize(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length for cosine similarity via L2 distance."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _logit_to_score(logit: float) -> float:
    """Map a cross-encoder raw logit to a [0, 1] relevance score via sigmoid.

    fastembed 的 TextCrossEncoder.rerank() 返回模型原始 logit（单调，但无界）。
    与旧 FlagReranker.compute_score(normalize=True) 一致地过 sigmoid，下游
    min_similarity 阈值（0.5）才能维持原有语义。分段计算避免 exp 溢出。
    """
    try:
        logit = float(logit)
    except (TypeError, ValueError):
        return 0.0
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    e = math.exp(logit)
    return e / (1.0 + e)
