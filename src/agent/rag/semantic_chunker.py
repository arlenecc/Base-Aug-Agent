"""Semantic chunking via chonkie.

Splits text on semantic boundaries (sentence-level embeddings + similarity
threshold) instead of naive character/recursive splits.  Semantic chunks keep
topically-related sentences together, which materially improves retrieval
quality for e-books and long documents.

Embedding backends (in priority order):
  1. Explicitly injected ``embedding_function`` → wrapped via
     ``_ChonkieEmbeddingAdapter`` (tests / external callers only — the engine
     deliberately does NOT inject its nomic EF for ingestion).
  2. model2vec static embeddings (potion-base-32M) — the production path for
     chunking decisions, ~100x faster than BERT inference (#118).  Crucially it
     does NOT touch the nomic-embed-text EF/lock used to embed stored chunks,
     so semantic chunking never contends with the later storage-embedding stage.

The engine passes ``embedding_function=None`` for ingestion so the chunker uses
model2vec.  model2vec is auto-installed; if it is unavailable, semantic chunking
is skipped (returns ``None``) and callers fall back to the existing recursive
chunker — it intentionally no longer falls back to the nomic EF, which would
reintroduce contention with storage embedding.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, List, Optional

# _gap_between 已上移到 chunker.py 共用（recursive 路径的 _merge_overlap 也有
# 同样的「硬编码分隔符伪造非原文内容」问题），本模块通过既有依赖方向引用。
from .chunker import _gap_between, estimate_tokens  # noqa: F401  (estimate_tokens re-exported for tests)

logger = logging.getLogger(__name__)

# SemanticChunker splits sentences on ". "/"! "/"? "/newline.  Chinese text
# uses "。" without a trailing space, so add CJK-friendly delimiters to avoid
# treating an entire paragraph of Chinese prose as one "sentence".
_DELIM = [". ", "! ", "? ", "\n", "。", "！", "？", "；", "，"]

# chonkie is an optional dependency.  Resolve its BaseEmbeddings base class at
# import time if available; otherwise fall back to ``object`` so the module can
# still be imported (semantic_chunk_text() returns None in that case).
try:
    from chonkie.embeddings.base import BaseEmbeddings as _BaseEmbeddings
except ImportError:  # pragma: no cover - chonkie absent
    _BaseEmbeddings = object


def _is_chonkie_available() -> bool:
    try:
        import chonkie  # noqa: F401
        return True
    except ImportError:
        return False


def semantic_chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: Optional[int] = None,
    min_chunk_size: int = 100,
    overlap_percent: float = 0.10,
    embedding_function=None,
    fill_to_chunk_size: bool = True,
) -> Optional[List[str]]:
    """Split ``text`` into semantic chunks using chonkie's SemanticChunker.

    Args:
        text: The text to split.
        chunk_size: Target maximum chunk size in **tokens** (800).
        chunk_overlap: Overlap between consecutive chunks in **tokens**.
            显式提供时**优先**生效（含 0 = 不重叠）；未提供/为 0 时回退
            ``int(chunk_size * overlap_percent)``。
        min_chunk_size: Target minimum chunk size in **tokens** (100). Chunks
            smaller than this are greedily merged into their neighbours.
        overlap_percent: Fraction of ``chunk_size`` used as overlap (0.10 =>
            80 tokens), preserving continuity across chunk boundaries.  Takes
            precedence over ``chunk_overlap``.
        embedding_function: A callable ``embed_documents(list[str]) ->
            list[list[float]]``. When omitted, the shared nomic-embed-text
            singleton is used.

    Returns:
        List of chunk strings, or ``None`` if chonkie is unavailable or no
        embedding function could be resolved (callers should fall back).
    """
    if not text or not text.strip():
        return []

    if not _is_chonkie_available():
        logger.debug("semantic chunking skipped: chonkie not installed")
        return None

    try:
        from chonkie import SemanticChunker

        if embedding_function is not None:
            # 显式注入（测试/外部调用）：走适配器，行为与旧版一致。
            embeddings = _ChonkieEmbeddingAdapter(embedding_function)
        else:
            # 生产路径：强制 model2vec 静态嵌入（potion-base-32M，#118）。切片
            # 决策只需相邻相似度的相对大小，无需与入库检索向量同空间，静态嵌入
            # 足够；且它不碰入库用的 nomic EF 及其全局锁，避免与「分片后
            # embedding 入库」争抢同一串行资源。model2vec 不可用时直接跳过语义
            # 切片（返回 None），由上层递归切片兜底——不再回退 nomic。
            embeddings = _get_static_embeddings()
            if embeddings is None:
                logger.debug("semantic chunking skipped: model2vec unavailable")
                return None

        # When chonkie is present, _BaseEmbeddings is the real base class and
        # the isinstance check in SemanticChunker.__init__ will pass.  When it
        # fell back to ``object``, chonkie can't be imported anyway (handled
        # above by _is_chonkie_available), so this branch is defensive only.
        # 统一 token 计量（见 _CharTokenEmbeddings）：无论嵌入后端是谁，
        # chonkie 的 chunk_size 必须按 estimate_tokens（字符制）理解。
        chunker = SemanticChunker(
            embedding_model=_CharTokenEmbeddings(embeddings),
            threshold=0.7,          # lower => larger, more coherent groups
            chunk_size=chunk_size,  # hard ceiling in tokens
            min_sentences_per_chunk=2,
            min_characters_per_sentence=20,
            delim=_DELIM,
        )
        chunks = chunker.chunk(text)
        result = [c.text.strip() for c in chunks if c.text.strip()]
        # 传入原文 text，合并/重叠时用「源码里真实的间隔」拼接，保证产出的
        # chunk 一定是原文的连续子串（见 _gap_between）。
        merged = _merge_to_target_size(result, chunk_size, min_chunk_size, source=text,
                                       fill_to_chunk_size=fill_to_chunk_size)
        return _apply_overlap(
            merged, chunk_size, chunk_overlap, overlap_percent, source=text,
        )
    except Exception as e:
        # Never let a chunking failure abort ingestion — the caller falls back
        # to the recursive chunker.
        logger.warning("semantic chunking failed (%s), falling back to recursive: %s",
                       type(e).__name__, e)
        return None


def _merge_to_target_size(
    chunks: List[str],
    chunk_size: int,
    min_chunk_size: int,
    source: Optional[str] = None,
    fill_to_chunk_size: bool = True,
) -> List[str]:
    """Greedily merge adjacent chunks toward the [min, max] token target.

    SemanticChunker splits on *semantic* boundaries, so chunks can be smaller
    than the target when a document has many short, topically-distinct
    sentences.  We merge adjacent chunks (which are already topically
    contiguous, since semantic splitting only cuts at low-similarity points).

    合并策略（2026-10-27 修订）：

    * ``fill_to_chunk_size=True``（全局合并阶段）：只要并入下一片不超
      ``chunk_size`` 就持续合并——切片大小是用户在设置面板显式配置的，
      语义边界只决定「从哪里断」，不决定「切多碎」。旧实现只把 < 
      ``min_chunk_size`` 的碎片并到 100 字就停，样板重复文本（相似度曲线
      常数化，chonkie 逐边界产生局部极小值）会产出大量 ~150 字碎片，
      用户设 1200 也照样切短。
    * ``fill_to_chunk_size=False``（hybrid 的 ``_semantic_split`` 内部阶段）：
      保持旧行为，只并 ``< min_chunk_size`` 的碎片（那里 min_chunk_size=1，
      等于不合并）——语义切分的子片要先保持边界，全局阶段再统一填充。
    """
    from .chunker import estimate_tokens

    if not chunks:
        return chunks

    merged: List[str] = []
    current = chunks[0]

    for nxt in chunks[1:]:
        cur_tokens = estimate_tokens(current)
        if cur_tokens < min_chunk_size or (
                fill_to_chunk_size
                and cur_tokens + estimate_tokens(nxt) <= chunk_size):
            current = current + _gap_between(source, current, nxt) + nxt
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def _apply_overlap(
    chunks: List[str],
    chunk_size: int,
    chunk_overlap: int,
    overlap_percent: float,
    source: Optional[str] = None,
) -> List[str]:
    """Add token-level overlap between consecutive chunks.

    SemanticChunker produces non-overlapping chunks, so a sentence split at a
    chunk boundary loses its surrounding context.  We prepend to each chunk a
    tail slice of the *previous* chunk — chosen at a sentence boundary so we
    never split a sentence mid-way — so that retrieval across a boundary still
    sees the surrounding sentences.

    The overlap size is the explicit ``chunk_overlap`` when it is set;
    otherwise it falls back to ``int(chunk_size * overlap_percent)``.

    优先级（2026-10-27 修订）：显式 ``chunk_overlap`` **优先**于
    ``overlap_percent``（含 0 = 明确不要重叠）。旧实现 percent 优先——
    用户在设置面板把 overlap 设 0（明确不要重叠）时，仍被强加
    10% × chunk_size 的重叠，切片总量膨胀近 2×（实测 2900 字文本拼接出
    5684 字）。engine 每次都会把 config.rag_chunk_overlap 显式传下来，
    percent 只在 ``chunk_overlap is None`` 时兜底。
    """
    from .chunker import estimate_tokens

    if len(chunks) <= 1:
        return chunks

    overlap = (chunk_overlap if chunk_overlap is not None
               else int(chunk_size * overlap_percent))
    if overlap <= 0:
        return chunks

    result: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        # Overlap is always derived from the *original* previous chunk (not the
        # overlap-padded one), so overlap never compounds across boundaries.
        tail = _tail_by_tokens(chunks[i - 1], overlap)
        if tail:
            # 用源码里真实的间隔拼接，保证结果仍是原文的连续子串。
            result.append(tail + _gap_between(source, tail, chunks[i]) + chunks[i])
        else:
            result.append(chunks[i])
    return result


_TAIL_BOUNDARY_RE = re.compile(r"(?<=[。！？!?\.])[ \t]*\n?|\n")


def _tail_by_tokens(text: str, max_tokens: int) -> str:
    """Return the trailing portion of ``text`` up to ``max_tokens`` tokens,
    aligned to a sentence boundary (never splitting a sentence).

    切片必须取自原文 —— 旧实现先用 ``re.split`` 拆句（分隔符中的 ``\\s*``
    被丢弃）再用 ``"".join`` 拼回来，句间的换行/空格因此凭空消失，产出的
    重叠文本在原文里根本不存在（模型会看到源文件没有的内容）。这里改为直接
    在原文里找句末边界并切片，保证结果恒为原文子串。
    """
    from .chunker import estimate_tokens

    if not text or max_tokens <= 0:
        return ""

    # 所有可作为起点的句末边界（0 = 整段文本）。
    bounds = [0] + [m.end() for m in _TAIL_BOUNDARY_RE.finditer(text)]
    bounds = [b for b in bounds if b < len(text)]

    # 从最靠近末尾的边界往回找，取能满足 token 预算的最长后缀。
    start = len(text)
    for b in reversed(bounds):
        if estimate_tokens(text[b:]) > max_tokens:
            break
        start = b
    return text[start:].strip()


# ---------------------------------------------------------------------------
# static embeddings for chunking decisions (potion via model2vec)
# ---------------------------------------------------------------------------

# 语义切片的「快路径」嵌入后端：model2vec 静态嵌入（potion-base-32M，
# ~30MB，纯 numpy 查表 + 平均池化，无 BERT 前向）。它只服务「相邻窗口是否
# 语义断裂」的切片决策——决策只需要相邻相似度的相对大小，不需要与检索
# 向量同空间，静态嵌入完全够用。实测吞吐比共享 nomic 适配器（BERT 推理
# + 2N 次 embed）高约两个数量级（88k 字符章节 90-150s → 秒级，#118）。
#
# chonkie 自带的 AutoEmbeddings 会把 minishlab/* 路由到 Model2VecEmbeddings
# （chonkie.embeddings.registry），model2vec 未安装时其构造抛 ImportError。
_STATIC_EMBS = None
_STATIC_EMBS_RESOLVED = False
_STATIC_EMBS_FAILED_AT: Optional[float] = None
_STATIC_EMBS_RETRY_INTERVAL = 60.0  # 失败负缓存：60s 内不重复尝试
_STATIC_EMBS_LOCK = threading.Lock()

_POTION_MODEL = "minishlab/potion-base-32M"


def _get_static_embeddings():
    """Return the model2vec static embeddings for chunking, or None.

    失败负缓存与下方静态嵌入加载同款：deps 在同步前自动
    安装 model2vec，安装成功后的下一次尝试（60s 负缓存过期后）自动升级
    到快路径，无需重启。首次使用会下载 potion 权重（~30MB，HF 缓存）。
    """
    global _STATIC_EMBS, _STATIC_EMBS_RESOLVED, _STATIC_EMBS_FAILED_AT
    if _STATIC_EMBS_RESOLVED:
        return _STATIC_EMBS
    if (
        _STATIC_EMBS_FAILED_AT is not None
        and time.monotonic() - _STATIC_EMBS_FAILED_AT < _STATIC_EMBS_RETRY_INTERVAL
    ):
        return None
    with _STATIC_EMBS_LOCK:
        if _STATIC_EMBS_RESOLVED:
            return _STATIC_EMBS
        try:
            from chonkie.embeddings import AutoEmbeddings

            _STATIC_EMBS = AutoEmbeddings.get_embeddings(_POTION_MODEL)
            _STATIC_EMBS_RESOLVED = True
            _STATIC_EMBS_FAILED_AT = None
            logger.info("语义切片使用静态嵌入 %s（model2vec 快路径）", _POTION_MODEL)
        except Exception as e:
            logger.debug("model2vec 静态嵌入不可用（%s），回退 nomic 适配器", e)
            _STATIC_EMBS = None
            _STATIC_EMBS_FAILED_AT = time.monotonic()
    return _STATIC_EMBS


# ---------------------------------------------------------------------------
# chonkie embedding adapter
# ---------------------------------------------------------------------------

class _CharTokenEmbeddings(_BaseEmbeddings):
    """包装任意嵌入后端，仅把 chonkie 的 token 计量统一到 ``estimate_tokens``。

    **为什么必须**：chonkie 的 ``chunk_size`` 上限按「嵌入后端
    ``get_tokenizer()`` 返回的 tokenizer」数 token。显式注入的 EF 走
    ``_ChonkieEmbeddingAdapter``（get_tokenizer 已返回 estimate_tokens，
    字符制）没问题；但生产路径的 model2vec 静态嵌入自带 potion 词表
    tokenizer——中文在它词表里严重碎片化，实测 2900 字文本被它数成远超
    2900 的 token，用户在设置面板配置的 chunk_size=1200 被静默缩水成
    ~230 字的碎片（「某些环节把长分片切短」的实锤来源）。

    这里只重写 ``get_tokenizer``，embed/similarity 全部委托内部后端，
    切片决策的嵌入空间不受影响。
    """

    def __init__(self, inner):
        super().__init__()
        self._inner = inner

    # -- token 计量：统一字符制 ---------------------------------------------
    def get_tokenizer(self) -> Any:
        return estimate_tokens

    # -- 其余全部委托内部后端 -------------------------------------------------
    def embed(self, text: str):
        return self._inner.embed(text)

    def embed_batch(self, texts: List[str]):
        return self._inner.embed_batch(texts)

    def similarity(self, u, v) -> float:
        return self._inner.similarity(u, v)

    @property
    def dimension(self) -> int:
        return self._inner.dimension

class _ChonkieEmbeddingAdapter(_BaseEmbeddings):
    """Adapt a FastEmbed embedding function to chonkie's BaseEmbeddings interface.

    chonkie's SemanticChunker accepts either a model-name string or a
    ``BaseEmbeddings`` instance.  To keep the exact nomic-embed-text ONNX model
    shared with the vector store (instead of loading potion-base-32M or a
    second nomic copy), we subclass BaseEmbeddings and wrap the existing
    embedding function.  Required members:

      * ``embed(text)`` / ``embed_batch(texts)`` — sentence embeddings
      * ``similarity(u, v)`` — cosine similarity
      * ``get_tokenizer()`` — returns our ``estimate_tokens`` callable (chonkie
        wraps it via CallableAutoTokenizer for ``count_tokens``)
      * ``dimension`` — embedding vector dimension
    """

    def __init__(self, ef):
        super().__init__()
        self._ef = ef
        # Resolve dimension lazily (requires one embedding call) to avoid
        # forcing model load at construction time.
        self._dim = None

    # -- embedding ---------------------------------------------------------
    def _embed_many(self, texts: List[str]):
        """Embed a list of texts, tolerating either ``embed_documents`` or the
        callable ``__call__`` protocol (the test fake exposes ``__call__``)."""
        if hasattr(self._ef, "embed_documents"):
            return self._ef.embed_documents(texts)
        return self._ef(texts)

    def embed(self, text: str):
        return self._to_array(self._embed_many([text])[0])

    def embed_batch(self, texts: List[str]):
        return [self._to_array(v) for v in self._embed_many(list(texts))]

    @staticmethod
    def _to_array(vec):
        import numpy as np
        return np.asarray(vec, dtype="float32")

    # -- similarity --------------------------------------------------------
    def similarity(self, u, v) -> float:
        import numpy as np
        u = np.asarray(u, dtype="float32")
        v = np.asarray(v, dtype="float32")
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        if denom == 0.0:
            return 0.0
        return float(np.dot(u, v) / denom)

    # -- dimension ---------------------------------------------------------
    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("probe"))
        return self._dim

    # -- tokenizer ---------------------------------------------------------
    def get_tokenizer(self) -> Any:
        """Return a token counter callable.

        chonkie wraps callables via ``CallableAutoTokenizer`` whose
        ``count_tokens(text)`` calls it directly.  We use the same heuristic
        as the recursive chunker (``estimate_tokens``) so chunk sizes are
        measured consistently across both paths.
        """
        from .chunker import estimate_tokens
        return estimate_tokens
