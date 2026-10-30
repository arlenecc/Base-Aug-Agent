"""Direct local embedding and reranker runtimes.

Backends expose the small interfaces already consumed by VectorStore:
``embed_documents``, ``embed_query`` and ``rerank``. Heavy optional packages
are imported only when the corresponding backend is selected.
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..model_backends import resolve_model_path, apply_mlx_memory_policy
from . import embedding_prefixes as _eprefix

logger = logging.getLogger(__name__)

EMBEDDING_BACKENDS = ("fastembed", "mlx", "onnx", "llama_cpp")
# UI 下拉只提供前三项；"disabled" 仅由配置文件的 rag_rerank_enabled=false
# 间接产生（见 VectorStore.__init__），create_reranker_backend 仍容错处理。
RERANKER_BACKENDS = ("fastembed", "mlx", "onnx")

# MLX 内存策略：复用聊天后端同款 apply_mlx_memory_policy（2026-10-26 用户
# 决策版）：memory_limit = 60% × 系统当前**可用**内存（下限 1GB）、
# cache_limit = 一律 1GB。函数每次调用按当时状态重新评估（同步卸载 LLM 后
# 可用内存回升，限制随之刷新），RAG 各 MLX 模型加载时都会触发。
_MLX_MEM_CAP_DONE = False


def _cap_mlx_memory_limits() -> None:
    """限制 MLX 内存，避免同步期统一内存被 cache 吃满或解析阶段挤占系统。

    每次都重新应用（不做只跑一次的门闩）：策略按「当时可用内存」计算，
    而 MLX 模型加载时机各不相同（应用启动 / 同步开始 / OCR 惰性加载），
    只有重新评估才能让限制反映当前系统状态。apply_mlx_memory_policy 本身
    是两次系统调用 + 一次 psutil 读取，开销可忽略。
    """
    apply_mlx_memory_policy()

# bge-reranker-v2-m3 的位置嵌入上限是 8192（本地 MLX/ONNX 权重的 config 均为
# max_position_embeddings=8194，fastembed 侧 tokenizer 本身就截断在 8192），
# 旧版 bge-reranker-base 只有 512；embedding 侧 nomic-embed-text-v1.5 是旋转
# 位置编码，训练长度 2048、靠 RoPE 外推到 n_positions=8192。三类上限都按各自
# 模型 config 现算，不写死：请求值超过模型上限时夹回去，避免把超出位置嵌入的
# token 喂进网络（ONNX 侧不会报错，只会静默给出无意义结果）。
_RERANK_DEFAULT_MAX_LENGTH = 8192
_RERANK_UNKNOWN_MAX_LENGTH = 512
_EMBEDDING_DEFAULT_MAX_LENGTH = 8192
_EMBEDDING_UNKNOWN_MAX_LENGTH = 512
_RERANK_BATCH_TOKEN_BUDGET = 8192
# 嵌入批收缩：与 reranker 同理，但按后端分开标定（单位「条 × 字符²」，与
# VectorStore._shape_cap_size 同口径）。
#   mlx  : 走 flash attention，注意力不物化 [B,H,S,S]，内存 ∝ 条数 × S（O(S)），
#         故可放大——300M → B=4@8192，全 S 区间激活 ≤ ~2GB。
#   onnx : CPU 会物化注意力分数+概率两个 [B,H,S,S]，内存 ∝ 条数 × S²（O(S²)），
#         150M → B=2@8192；实际多为 int8（更低），即便误用 fp16 也仅 ~3.5GB，
#         用户确认可接受（fp32 会到 ~2 倍、约 14GB，不推荐）。
#   其它 : 沿用原 6M（fastembed 内部自管批，且默认 500 切片不触发）。
_EMBEDDING_SHAPE_BUDGET = {
    "mlx": 300_000_000,
    "onnx": 150_000_000,
}
_EMBEDDING_SHAPE_BUDGET_DEFAULT = 6_000_000


def _model_max_length(model_dir: Any, default: int = _RERANK_UNKNOWN_MAX_LENGTH) -> int:
    """按模型 config 算可截断上限。

    旋转式位置编码（NomicBERT）：``max_position_embeddings`` 只是「训练长度」
    （nomic=2048），真实序列上限是 ``n_positions``（nomic=8192，靠 RoPE 外推）
    —— 直接读它。绝对位置嵌入（XLM-R / bge 系列）：位置表是硬上限，留 2 个
    特殊 token 位取 ``max_position_embeddings - 2``。config 缺失则退回保守默认。
    """
    try:
        config_path = Path(model_dir) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return default
    n_positions = int(config.get("n_positions") or 0)
    if n_positions > 0:
        return max(8, n_positions)
    value = int(config.get("max_position_embeddings") or 0)
    if value > 0:
        return max(8, value - 2)
    return default


def _rerank_batch_for(batch_size: int, seq_len: int,
                      budget: int = _RERANK_BATCH_TOKEN_BUDGET) -> int:
    """Shrink the batch so ``batch * seq_len`` stays within the token budget."""
    size = max(1, int(batch_size))
    if seq_len and int(seq_len) > 0:
        size = min(size, max(1, budget // int(seq_len)))
    return size


def _embedding_batch_for(batch_size: int, seq_len: int, backend: str = "") -> int:
    """Embedding 版批收缩（与 reranker 同机制，但按后端取预算）。

    ``backend`` 为 "mlx" / "onnx" 时用各自标定预算，否则用默认 6M。
    返回「不超过请求 batch_size 且不超过形状预算允许的最大条数」。

    下限 **2 条**：内存硬约束可以把批压得很小，但不夹到 1——单条长切片吞吐
    塌到最优的 50-69%，模型加载 / 会话初始化的固定成本也被单条摊得极薄；
    实测 int8 ONNX @8192 B=2 内存可接受，故下探到 2 为止。请求本身只有 1
    条时仍返回 1，不会为了凑下限而放大批次。
    """
    size = max(1, int(batch_size))
    if seq_len and int(seq_len) > 0:
        budget = _EMBEDDING_SHAPE_BUDGET.get(backend, _EMBEDDING_SHAPE_BUDGET_DEFAULT)
        size = min(size, max(2, budget // (int(seq_len) * int(seq_len))))
    return size


def _normalize_rows(values):
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _find_onnx(path: str) -> tuple[str, str]:
    raw_path = Path(os.path.expandvars(os.path.expanduser(path or "")))
    expect_file = raw_path.suffix.lower() == ".onnx"
    resolved = Path(resolve_model_path(path, expect_file=expect_file))
    if resolved.is_file():
        return str(resolved), str(resolved.parent)
    preferred = (
        list(resolved.glob("model_quantized.onnx"))
        + list(resolved.glob("model_int8.onnx"))
        + list(resolved.glob("onnx/model_quantized.onnx"))
        + list(resolved.glob("onnx/model_int8.onnx"))
        + sorted(resolved.glob("*.onnx"))
        + sorted(resolved.glob("onnx/*.onnx"))
    )
    if not preferred:
        raise FileNotFoundError(f"目录中未找到 ONNX 文件: {resolved}")
    return str(preferred[0]), str(resolved)


class MLXEmbeddingFunction:
    """MLX embedding adapter with native NomicBERT and mlx-embeddings paths."""

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 8,
        max_length: int = _EMBEDDING_DEFAULT_MAX_LENGTH,
        document_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ):
        self._model_path = resolve_model_path(model_path)
        self._batch_size = max(1, int(batch_size))
        # 请求值先记下，模型加载后按 config 的位置/旋转上限夹紧（见 _model_max_length）。
        self._max_length_requested = max(8, int(max_length))
        self._max_length = self._max_length_requested
        # 任务前缀按模型族统一解析（embedding_prefixes：唯一事实源）；
        # 显式传入的前缀仍可覆盖（测试/特殊模型用）。
        _doc_p, _query_p = _eprefix.prefixes(self._model_path)
        self._document_prefix = document_prefix if document_prefix is not None else _doc_p
        self._query_prefix = query_prefix if query_prefix is not None else _query_p
        self._model = None
        self._processor = None
        self._generate = None
        self._model_type = ""
        self._is_bert = False
        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            _cap_mlx_memory_limits()
            config_path = Path(self._model_path) / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self._model_type = str(config.get("model_type", ""))
            # 截断长度不得超过模型上限（NomicBERT 旋转式取 n_positions=8192，
            # 绝对位置嵌入取 max_position_embeddings-2），否则越界序列会被静默忽略。
            ceiling = _model_max_length(self._model_path)
            self._max_length = min(self._max_length_requested, ceiling)
            if self._max_length < self._max_length_requested:
                logger.info(
                    "MLX Embedding 截断长度按模型上限夹紧: %d → %d",
                    self._max_length_requested, self._max_length,
                )
            if self._model_type == "nomic_bert":
                self._model, self._processor = self._load_nomic(config)
                self._generate = None
            else:
                try:
                    from mlx_embeddings import generate, load
                except ImportError as e:
                    raise ImportError(
                        "MLX Embedding 后端需要 mlx-embeddings: pip install mlx-embeddings"
                    ) from e
                try:
                    self._model, self._processor = load(self._model_path, lazy=False)
                except Exception:
                    # mlx-embeddings 不支持的 bert 类导出（如 bge 中文模型：权重键
                    # 命名与库预期不符会报 "parameters not in model"）→ 退回项目自带
                    # 的 mlx.nn Bert 实现（标准 HF bert 权重键，掩码均值池化）。
                    logger.info(
                        "mlx-embeddings 无法加载 %s，退回 mlx.nn Bert 实现", self._model_path
                    )
                    self._model, self._processor = self._load_bert(config)
                    self._generate = None
                    self._is_bert = True
                    # bge 检索约定（经模型族策略表统一给出）：文档不加前缀，
                    # 查询加检索指令前缀。bert 兜底路径多为 bge 中文导出，
                    # 即使目录名不含 "bge" 也按 bge 族取值。
                    self._document_prefix = _eprefix.document_prefix("bge")
                    self._query_prefix = _eprefix.query_prefix("bge")
                else:
                    self._generate = generate
            logger.info("MLX Embedding 加载完成: %s", self._model_path)

    def _load_nomic(self, config: Dict[str, Any]):
        """Build the NomicBERT graph matching the repository's MLX weights."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Nomic MLX Embedding 需要 mlx 与 transformers"
            ) from e

        hidden_size = int(config.get("hidden_size", config.get("n_embd", 768)))
        intermediate_size = int(config.get("intermediate_size", config.get("n_inner", 3072)))
        num_heads = int(config.get("num_attention_heads", config.get("n_head", 12)))
        num_layers = int(config.get("num_hidden_layers", config.get("n_layer", 12)))
        vocab_size = int(config.get("vocab_size", 30528))
        type_vocab_size = int(config.get("type_vocab_size", 2))
        eps = float(config.get("layer_norm_epsilon", config.get("layer_norm_eps", 1e-12)))
        rope_base = float(config.get("rotary_emb_base", 1000.0))
        head_dim = hidden_size // num_heads

        # 量化权重（MLX 约定：weight(uint32) + scales + biases 兄弟键 + config.quantization）
        # 必须用 QuantizedLinear / QuantizedEmbedding 承载——普通 nn.Linear 只认单个
        # fp16 的 weight，load_weights(strict=True) 会因为多出 62 组 scales/biases 而报
        # "Received 124 parameters not in model"。无 quantization 字段时保持原 fp16 图。
        _quant = config.get("quantization") or {}
        _bits = int(_quant.get("bits", 0)) or None
        _group = int(_quant.get("group_size", 64)) or 64
        _mode = str(_quant.get("mode", "affine"))
        quantized = bool(_bits)

        def _linear(in_dims, out_dims):
            if quantized:
                return nn.QuantizedLinear(in_dims, out_dims, bias=False,
                                          group_size=_group, bits=_bits, mode=_mode)
            return nn.Linear(in_dims, out_dims, bias=False)

        def _embedding(num, dims):
            if quantized:
                return nn.QuantizedEmbedding(num, dims, group_size=_group,
                                             bits=_bits, mode=_mode)
            return nn.Embedding(num, dims)

        def rotate_half(x):
            half = x.shape[-1] // 2
            return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)

        def apply_rope(x):
            seq_len = x.shape[1]
            inv_freq = 1.0 / (
                rope_base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
            )
            freqs = mx.arange(seq_len, dtype=mx.float32)[:, None] * inv_freq[None, :]
            duplicated = mx.concatenate((freqs, freqs), axis=-1)
            cos = mx.cos(duplicated).astype(x.dtype)[None, :, None, :]
            sin = mx.sin(duplicated).astype(x.dtype)[None, :, None, :]
            return x * cos + rotate_half(x) * sin

        class Embeddings(nn.Module):
            def __init__(self):
                super().__init__()
                self.word_embeddings = _embedding(vocab_size, hidden_size)
                self.token_type_embeddings = _embedding(type_vocab_size, hidden_size)
                self.norm = nn.LayerNorm(hidden_size, eps=eps)

            def __call__(self, input_ids, token_type_ids=None):
                if token_type_ids is None:
                    token_type_ids = mx.zeros_like(input_ids)
                return self.norm(
                    self.word_embeddings(input_ids)
                    + self.token_type_embeddings(token_type_ids)
                )

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.Wqkv = _linear(hidden_size, hidden_size * 3)
                self.out_proj = _linear(hidden_size, hidden_size)

            def __call__(self, x, attention_mask):
                batch, seq_len, _ = x.shape
                qkv = self.Wqkv(x).reshape(batch, seq_len, 3, num_heads, head_dim)
                q = apply_rope(qkv[:, :, 0]).transpose(0, 2, 1, 3)
                k = apply_rope(qkv[:, :, 1]).transpose(0, 2, 1, 3)
                v = qkv[:, :, 2].transpose(0, 2, 1, 3)
                mask = attention_mask[:, None, None, :].astype(mx.bool_)
                output = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=head_dim ** -0.5, mask=mask
                )
                output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden_size)
                return self.out_proj(output)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc11 = _linear(hidden_size, intermediate_size)
                self.fc12 = _linear(hidden_size, intermediate_size)
                self.fc2 = _linear(intermediate_size, hidden_size)

            def __call__(self, x):
                return self.fc2(self.fc11(x) * nn.silu(self.fc12(x)))

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()
                self.mlp = MLP()
                self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
                self.norm2 = nn.LayerNorm(hidden_size, eps=eps)

            def __call__(self, x, attention_mask):
                x = self.norm1(x + self.attn(x, attention_mask))
                return self.norm2(x + self.mlp(x))

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [Layer() for _ in range(num_layers)]

            def __call__(self, x, attention_mask):
                for layer in self.layers:
                    x = layer(x, attention_mask)
                return x

        class NomicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = Embeddings()
                self.encoder = Encoder()

            def __call__(self, input_ids, attention_mask, token_type_ids=None):
                hidden = self.embeddings(input_ids, token_type_ids)
                hidden = self.encoder(hidden, attention_mask)
                mask = attention_mask[..., None].astype(hidden.dtype)
                pooled = (hidden * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1)
                return pooled / mx.maximum(mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

        model = NomicModel()
        weights = mx.load(str(Path(self._model_path) / "model.safetensors"))
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, local_files_only=True, trust_remote_code=False
        )
        return model, tokenizer

    def _load_bert(self, config: Dict[str, Any]):
        """Build a standard BERT encoder with mlx.nn (for bert-style exports
        like bge that mlx-embeddings cannot load due to weight-key naming).

        Mirrors the repository's MLX weights: standard HF bert key names
        (``embeddings.word_embeddings`` / ``encoder.layers.N.attention.query_proj``
        …), masked mean pooling over real tokens, L2 normalize. ``pooler.*``
        keys are ignored (``strict=False``).
        """
        import mlx.core as mx
        import mlx.nn as nn
        from transformers import AutoTokenizer

        hidden_size = int(config.get("hidden_size", 768))
        intermediate_size = int(config.get("intermediate_size", 3072))
        num_heads = int(config.get("num_attention_heads", 12))
        num_layers = int(config.get("num_hidden_layers", 12))
        vocab_size = int(config.get("vocab_size", 30522))
        type_vocab_size = int(config.get("type_vocab_size", 2))
        eps = float(config.get("layer_norm_eps", config.get("layer_norm_epsilon", 1e-12)))
        max_pos = int(config.get("max_position_embeddings", 512))

        class _BertEmbeddings(nn.Module):
            def __init__(self):
                super().__init__()
                self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
                self.position_embeddings = nn.Embedding(max_pos, hidden_size)
                self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
                self.norm = nn.LayerNorm(hidden_size, eps=eps)

            def __call__(self, ids, tok, pos):
                return self.norm(
                    self.word_embeddings(ids)
                    + self.position_embeddings(pos)
                    + self.token_type_embeddings(tok)
                )

        class _BertAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = num_heads
                self.scale = 1.0 / (hidden_size // num_heads) ** 0.5
                self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

            def __call__(self, x, mask):
                batch, seq_len, _ = x.shape
                nh, hd = self.num_heads, x.shape[-1] // self.num_heads
                split = lambda a: a.reshape(batch, seq_len, nh, hd).transpose(0, 2, 1, 3)
                q = split(self.query_proj(x))
                k = split(self.key_proj(x))
                v = split(self.value_proj(x))
                scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask
                weights = mx.softmax(scores, axis=-1)
                ctx = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, seq_len, nh * hd)
                return self.out_proj(ctx)

        class _BertLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1 = nn.LayerNorm(hidden_size, eps=eps)
                self.attention = _BertAttention()
                self.ln2 = nn.LayerNorm(hidden_size, eps=eps)
                self.linear1 = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=False)

            def __call__(self, x, mask):
                a = self.attention(self.ln1(x), mask)
                x = x + a
                f = self.linear2(nn.gelu(self.linear1(self.ln2(x))))
                return x + f

        class _BertEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [_BertLayer() for _ in range(num_layers)]

        class _BertModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = _BertEmbeddings()
                self.encoder = _BertEncoder()

            def __call__(self, ids, tok, pos, mask):
                h = self.embeddings(ids, tok, pos)
                for layer in self.encoder.layers:
                    h = layer(h, mask)
                return h  # 全序列隐藏，交由 _embed 做掩码均值池化

        model = _BertModel()
        weights = mx.load(str(Path(self._model_path) / "model.safetensors"))
        model.load_weights(weights, strict=False)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, local_files_only=True, trust_remote_code=False
        )
        return model, tokenizer

    def _embed(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_model()
        import numpy as np

        size = max(1, int(batch_size or self._batch_size))
        with self._inference_lock:
            outputs = []
            for start in range(0, len(texts), size):
                batch = texts[start:start + size]
                if self._model_type == "nomic_bert":
                    import mlx.core as mx

                    encoded = self._processor(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self._max_length,
                        return_tensors="np",
                    )
                    ids = np.asarray(encoded["input_ids"])
                    # 长切片下按 token 预算二次切批：注意力 O(L²)，直接把
                    # batch×L² 的激活全展开会把显存/统一内存打爆。
                    inner = _embedding_batch_for(len(batch), ids.shape[1], "mlx")
                    for offset in range(0, len(batch), inner):
                        token_types = encoded.get("token_type_ids")
                        chunk_tt = None
                        if token_types is not None:
                            chunk_tt = mx.array(
                                np.asarray(token_types)[offset:offset + inner]
                            )
                        values = self._model(
                            mx.array(ids[offset:offset + inner]),
                            mx.array(
                                np.asarray(encoded["attention_mask"])[offset:offset + inner]
                            ),
                            chunk_tt,
                        )
                        mx.eval(values)
                        outputs.extend(np.asarray(values, dtype=np.float32).tolist())
                elif self._is_bert:
                    import mlx.core as mx

                    # bert 回退路径（bge）：标准 HF tokenizer + 掩码均值池化。
                    encoded = self._processor(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self._max_length,
                        return_tensors="np",
                    )
                    ids = np.asarray(encoded["input_ids"])
                    am = np.asarray(encoded["attention_mask"])
                    pos = mx.arange(ids.shape[1])[None, :]
                    mask = (1 - mx.array(am)[:, None, None, :]) * -1e9
                    inner = _embedding_batch_for(len(batch), ids.shape[1], "mlx")
                    for offset in range(0, len(batch), inner):
                        ids_b = mx.array(ids[offset:offset + inner])
                        amb = mx.array(am[offset:offset + inner])
                        m = mask[offset:offset + inner]
                        h = self._model(ids_b, mx.zeros_like(ids_b), pos, m)
                        pooled = (h * amb[:, :, None]).sum(1) / mx.maximum(
                            amb.sum(1, keepdims=True), 1
                        )
                        pooled = pooled / mx.maximum(
                            mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
                        )
                        mx.eval(pooled)
                        outputs.extend(np.asarray(pooled, dtype=np.float32).tolist())
                else:
                    import mlx.core as mx

                    # 非 nomic 路径：generate 内部完成 tokenize，按 max_length
                    # 预估序列长收缩子批（保守上界），保证 O(L²) 内存可控。
                    inner = _embedding_batch_for(len(batch), self._max_length, "mlx")
                    for offset in range(0, len(batch), inner):
                        values = self._generate(
                            self._model,
                            self._processor,
                            batch[offset:offset + inner],
                            max_length=self._max_length,
                            padding=True,
                            truncation=True,
                        )
                        if hasattr(values, "text_embeds"):
                            values = values.text_embeds
                        mx.eval(values)
                        outputs.extend(_normalize_rows(np.asarray(values)).tolist())
            return outputs

    def __call__(self, input: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        prefixed = [
            _eprefix.apply_prefix(self._document_prefix, text)
            for text in input
        ]
        return self._embed(prefixed, batch_size=batch_size)

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        return self(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> List[float]:
        value = _eprefix.apply_prefix(self._query_prefix, text)
        return self._embed([value])[0]

    def name(self) -> str:
        return f"mlx:{self._model_path}"

    def close(self) -> None:
        with self._load_lock:
            self._model = None
            self._processor = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass


class ONNXEmbeddingFunction:
    """Generic direct ONNX text embedding adapter with mean pooling."""

    def __init__(
        self,
        model_path: str,
        *,
        threads: int = 4,
        batch_size: int = 8,
        max_length: int = _EMBEDDING_DEFAULT_MAX_LENGTH,
        document_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ):
        self._model_file, self._model_dir = _find_onnx(model_path)
        self._threads = max(1, int(threads))
        self._batch_size = max(1, int(batch_size))
        # 模型加载后按 config 的位置/旋转上限夹紧（见 _model_max_length）。
        self._max_length_requested = max(8, int(max_length))
        self._max_length = self._max_length_requested
        # 任务前缀按模型族统一解析（embedding_prefixes：唯一事实源）；
        # 按原始 model_path 识别（目录名含 bge/nomic/e5/jina 等关键字）。
        _doc_p, _query_p = _eprefix.prefixes(model_path)
        self._document_prefix = document_prefix if document_prefix is not None else _doc_p
        self._query_prefix = query_prefix if query_prefix is not None else _query_p
        self._session = None
        self._tokenizer = None
        self._lock = threading.RLock()

    def _ensure_model(self):
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from transformers import AutoTokenizer

            options = ort.SessionOptions()
            options.intra_op_num_threads = self._threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self._model_file,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_dir, local_files_only=True, trust_remote_code=False
            )
            # ONNX 图是动态序列长，超长输入不会报错、只会给出无意义向量，所以
            # 必须按位置/旋转上限夹紧（NomicBERT 旋转式 → n_positions=8192，
            # 绝对位置嵌入 → max_position_embeddings-2）。
            ceiling = _model_max_length(self._model_dir)
            self._max_length = min(self._max_length_requested, ceiling)
            if self._max_length < self._max_length_requested:
                logger.info(
                    "ONNX Embedding 截断长度按模型上限夹紧: %d → %d",
                    self._max_length_requested, self._max_length,
                )
            logger.info("ONNX Embedding 加载完成(CPU): %s", self._model_file)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_model()
        import numpy as np

        names = {item.name for item in self._session.get_inputs()}
        all_values = []
        with self._lock:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start:start + self._batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="np",
                )
                ids = np.asarray(encoded["input_ids"])
                # 长切片下按 token 预算二次切批（注意力 O(L²)）。
                inner = _embedding_batch_for(len(batch), ids.shape[1], "onnx")
                for offset in range(0, len(batch), inner):
                    chunk = {
                        key: np.asarray(value, dtype=np.int64)[offset:offset + inner]
                        for key, value in encoded.items()
                        if key in names
                    }
                    outputs = self._session.run(None, chunk)
                    values = np.asarray(outputs[0])
                    if values.ndim == 3:
                        mask = np.asarray(
                            encoded["attention_mask"], dtype=np.float32
                        )[offset:offset + inner, ..., None]
                        values = (values * mask).sum(axis=1) / np.maximum(
                            mask.sum(axis=1), 1e-12
                        )
                    elif values.ndim != 2:
                        raise RuntimeError(f"不支持的 ONNX Embedding 输出形状: {values.shape}")
                    all_values.extend(_normalize_rows(values).tolist())
        return all_values

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed([
            _eprefix.apply_prefix(self._document_prefix, text)
            for text in input
        ])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        value = _eprefix.apply_prefix(self._query_prefix, text)
        return self._embed([value])[0]

    def name(self) -> str:
        return f"onnx:{self._model_file}"

    def close(self) -> None:
        with self._lock:
            self._session = None
            self._tokenizer = None


class GGUFEmbeddingFunction:
    """Direct llama-cpp-python GGUF embedding adapter."""

    def __init__(self, model_path: str, *, threads: int = 4, batch_size: int = 512):
        self._model_path = resolve_model_path(model_path, expect_file=True)
        self._threads = max(1, int(threads))
        self._batch_size = max(1, int(batch_size))
        # 任务前缀按模型族统一解析（此前 GGUF 路径完全没加前缀——nomic GGUF
        # 缺 document/query 前缀会明显拉低检索质量，现与其余后端对齐）。
        _doc_p, _query_p = _eprefix.prefixes(self._model_path)
        self._document_prefix = _doc_p
        self._query_prefix = _query_p
        self._model = None
        self._lock = threading.RLock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from llama_cpp import Llama

            self._model = Llama(
                model_path=self._model_path,
                embedding=True,
                n_threads=self._threads,
                n_batch=self._batch_size,
                n_gpu_layers=-1,
                verbose=False,
            )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_model()
        with self._lock:
            response = self._model.create_embedding(texts)
        ordered = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
        return _normalize_rows([item["embedding"] for item in ordered]).tolist()

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embed([
            _eprefix.apply_prefix(self._document_prefix, text)
            for text in input
        ])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        value = _eprefix.apply_prefix(self._query_prefix, text)
        return self._embed([value])[0]

    def name(self) -> str:
        return f"llama_cpp:{self._model_path}"

    def close(self) -> None:
        with self._lock:
            if self._model is not None:
                try:
                    self._model.close()
                except Exception:
                    pass
            self._model = None


def _build_xlm_roberta_reranker(config: Dict[str, Any]):
    """Build an XLM-RoBERTa cross-encoder matching the mlx-community layout.

    权重键为 HF 原样：``roberta.embeddings.*`` / ``roberta.encoder.layer.N.*``
    / ``classifier.dense.*`` / ``classifier.out_proj.*``。量化变体（如 mxfp8）
    把 ``weight`` 存成打包 uint32 + ``scales``，必须用 ``Quantized*`` 模块装载，
    否则 ``nn.Linear`` 会把它当普通浮点权重直接算成垃圾。
    """
    import mlx.core as mx
    import mlx.nn as nn

    quant = config.get("quantization") or config.get("quantization_config") or {}
    mode = str(quant.get("mode", "affine"))
    bits = int(quant.get("bits", 4))
    group = int(quant.get("group_size", 64))
    hidden = int(config.get("hidden_size", 1024))
    intermediate = int(config.get("intermediate_size", 4096))
    heads = int(config.get("num_attention_heads", 16))
    layers = int(config.get("num_hidden_layers", 24))
    vocab = int(config.get("vocab_size", 250002))
    max_pos = int(config.get("max_position_embeddings", 8194))
    type_vocab = int(config.get("type_vocab_size", 1))
    num_labels = int(config.get("num_labels", 1))
    eps = float(config.get("layer_norm_eps", config.get("layer_norm_epsilon", 1e-5)))
    pad_id = int(config.get("pad_token_id", 1))
    head_dim = hidden // heads

    def linear(in_dims: int, out_dims: int):
        if quant:
            return nn.QuantizedLinear(in_dims, out_dims, True, group, bits, mode)
        return nn.Linear(in_dims, out_dims)

    def embedding(count: int, dims: int):
        if quant:
            return nn.QuantizedEmbedding(count, dims, group, bits, mode)
        return nn.Embedding(count, dims)

    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = linear(hidden, hidden)
            self.key = linear(hidden, hidden)
            self.value = linear(hidden, hidden)

        def __call__(self, x, mask):
            batch, seq, _ = x.shape
            shape = (batch, seq, heads, head_dim)
            q = self.query(x).reshape(shape).transpose(0, 2, 1, 3)
            k = self.key(x).reshape(shape).transpose(0, 2, 1, 3)
            v = self.value(x).reshape(shape).transpose(0, 2, 1, 3)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=head_dim ** -0.5, mask=mask
            )
            return out.transpose(0, 2, 1, 3).reshape(batch, seq, hidden)

    class SelfOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = linear(hidden, hidden)
            self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

        def __call__(self, x, residual):
            return self.LayerNorm(self.dense(x) + residual)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = SelfAttention()
            self.output = SelfOutput()

        def __call__(self, x, mask):
            return self.output(self.self(x, mask), x)

    class Intermediate(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = linear(hidden, intermediate)

        def __call__(self, x):
            return nn.gelu(self.dense(x))

    class Output(nn.Module):
        """FFN 出口：维度是 intermediate → hidden，与注意力的 hidden → hidden 不同。"""

        def __init__(self):
            super().__init__()
            self.dense = linear(intermediate, hidden)
            self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

        def __call__(self, x, residual):
            return self.LayerNorm(self.dense(x) + residual)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.intermediate = Intermediate()
            self.output = Output()

        def __call__(self, x, mask):
            attn = self.attention(x, mask)
            return self.output(self.intermediate(attn), attn)

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = embedding(vocab, hidden)
            self.position_embeddings = embedding(max_pos, hidden)
            self.token_type_embeddings = embedding(type_vocab, hidden)
            self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

        def __call__(self, input_ids, attention_mask):
            # XLM-R 的位置编号从 pad_token_id 起算，跳过 padding 位。
            keep = mx.where(input_ids != pad_id, 1, 0)
            position_ids = mx.cumsum(keep, axis=1) * keep + pad_id
            token_type_ids = mx.zeros(input_ids.shape, dtype=mx.int32)
            return self.LayerNorm(
                self.word_embeddings(input_ids)
                + self.token_type_embeddings(token_type_ids)
                + self.position_embeddings(position_ids)
            )

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = [Block() for _ in range(layers)]

        def __call__(self, x, mask):
            for block in self.layer:
                x = block(x, mask)
            return x

    class Roberta(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = Embeddings()
            self.encoder = Encoder()

        def __call__(self, input_ids, mask):
            return self.encoder(self.embeddings(input_ids, mask), mask)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = linear(hidden, hidden)
            self.out_proj = linear(hidden, num_labels)

        def __call__(self, features):
            x = nn.tanh(self.dense(features[:, 0]))
            return self.out_proj(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.roberta = Roberta()
            self.classifier = Classifier()

        def __call__(self, input_ids):
            mask = mx.where(input_ids != pad_id, 1, 0)
            hidden = self.roberta(input_ids, mask[:, None, None, :].astype(mx.bool_))
            return self.classifier(hidden)

    return Model()


class MLXReranker:
    """Direct MLX cross-encoder reranker (XLM-RoBERTa sequence classification)."""

    def __init__(self, model_path: str, *, batch_size: int = 16,
                 max_length: int = _RERANK_DEFAULT_MAX_LENGTH):
        self._model_path = resolve_model_path(model_path)
        self._batch_size = max(1, int(batch_size))
        # 请求值先记下，模型加载后按 config.json 的位置嵌入上限夹紧
        # （bge-reranker-v2-m3 → 8192，bge-reranker-base → 512）。
        self._max_length_requested = max(8, int(max_length))
        self._max_length = self._max_length_requested
        self._model = None
        self._tokenizer = None
        self._lock = threading.RLock()
        self._inference_lock = threading.RLock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            _cap_mlx_memory_limits()
            import json as _json

            import mlx.core as mx
            from mlx.utils import tree_flatten

            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "MLX Reranker 需要 transformers 加载本地 tokenizer.json"
                ) from e
            config_path = Path(self._model_path) / "config.json"
            config = _json.loads(config_path.read_text(encoding="utf-8"))
            model = _build_xlm_roberta_reranker(config)
            # 截断长度不得超过模型的位置嵌入数，否则 query+doc 拼接过长时
            # token 越界（MX 索引直接报错，不会静默出错）。
            ceiling = _model_max_length(self._model_path)
            self._max_length = min(self._max_length_requested, ceiling)
            if self._max_length < self._max_length_requested:
                logger.info(
                    "MLX Reranker 截断长度按模型上限夹紧: %d → %d（max_position_embeddings）",
                    self._max_length_requested, self._max_length,
                )
            weights = mx.load(str(Path(self._model_path) / "model.safetensors"))
            # 丢弃 MLX 侧不存在的 HF 残留键（如 roberta.embeddings.position_ids）。
            expected = {key for key, _ in tree_flatten(model.parameters())}
            usable = {k: v for k, v in weights.items() if k in expected}
            missing = sorted(expected - set(usable))
            if missing:
                raise RuntimeError(
                    f"MLX Reranker 权重缺少 {len(missing)} 个键，例如 {missing[:3]}；"
                    f"模型类型 {config.get('model_type', '?')} 可能与本实现不匹配"
                )
            model.load_weights(list(usable.items()), strict=True)
            mx.eval(model.parameters())
            model.eval()
            self._model = model
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, local_files_only=True, trust_remote_code=False
            )
            quantized = config.get("quantization") or config.get("quantization_config") or {}
            logger.info(
                "MLX Reranker 加载完成: %s (%s)",
                self._model_path,
                f"{quantized.get('mode', 'fp')}{quantized.get('bits', '')}" if quantized else "fp",
            )

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 16, **kwargs):
        docs = list(documents)
        if not docs:
            return []
        self._ensure_model()
        import mlx.core as mx
        import numpy as np

        size = max(1, int(batch_size or self._batch_size))
        scores: List[float] = []
        with self._inference_lock:
            for start in range(0, len(docs), size):
                batch = docs[start:start + size]
                encoded = self._tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="np",
                )
                ids = np.asarray(encoded["input_ids"])
                # 长切片下按 token 预算二次切批：注意力 O(L²)，直接把
                # batch×L² 的激活全展开会把显存/统一内存打爆。
                inner = _rerank_batch_for(len(batch), ids.shape[1])
                for offset in range(0, len(batch), inner):
                    logits = self._model(mx.array(ids[offset:offset + inner]))
                    mx.eval(logits)
                    scores.extend(
                        float(value) for value in np.asarray(logits).reshape(-1)
                    )
        return scores

    def name(self) -> str:
        return f"mlx-reranker:{self._model_path}"

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass


class ONNXReranker:
    """Direct CPU-only ONNX cross-encoder using a local Transformers tokenizer."""

    def __init__(self, model_path: str, *, threads: int = 4,
                 max_length: int = _RERANK_DEFAULT_MAX_LENGTH):
        self._model_file, self._model_dir = _find_onnx(model_path)
        self._threads = max(1, int(threads))
        # 模型加载后按 config.json 的位置嵌入上限夹紧（见 MLXReranker）。
        self._max_length_requested = max(8, int(max_length))
        self._max_length = self._max_length_requested
        self._session = None
        self._tokenizer = None
        self._lock = threading.RLock()
        self._ensure_model()

    def _ensure_model(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            self._model_file,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"Reranker 必须使用 CPUExecutionProvider，实际为 {self._session.get_providers()}"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_dir, local_files_only=True, trust_remote_code=False
        )
        # ONNX 图是动态序列长，超长输入不会报错、只会给出无意义分数，所以必须
        # 按位置嵌入上限夹紧：bge-reranker-v2-m3 → 8192，base → 512。
        ceiling = _model_max_length(self._model_dir)
        self._max_length = min(self._max_length_requested, ceiling)
        if self._max_length < self._max_length_requested:
            logger.info(
                "ONNX Reranker 截断长度按模型上限夹紧: %d → %d（max_position_embeddings）",
                self._max_length_requested, self._max_length,
            )
        logger.info("ONNX Reranker 加载完成(CPU): %s", self._model_file)

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 16, **kwargs):
        import numpy as np

        docs = list(documents)
        input_names = {item.name for item in self._session.get_inputs()}
        scores: List[float] = []
        size = max(1, int(batch_size))
        with self._lock:
            for start in range(0, len(docs), size):
                batch = docs[start:start + size]
                encoded = self._tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="np",
                )
                ids = np.asarray(encoded["input_ids"])
                # 长切片下按 token 预算二次切批（注意力 O(L²)，CPU 上尤甚）。
                inner = _rerank_batch_for(len(batch), ids.shape[1])
                for offset in range(0, len(batch), inner):
                    chunk = {
                        key: np.asarray(value[offset:offset + inner], dtype=np.int64)
                        for key, value in encoded.items()
                        if key in input_names
                    }
                    output = np.asarray(self._session.run(None, chunk)[0]).reshape(-1)
                    scores.extend(float(value) for value in output)
        return scores

    def close(self) -> None:
        with self._lock:
            self._session = None
            self._tokenizer = None


def _rerank_llama_cls():
    """构造 ``llama_cpp.Llama`` 子类，强制 tokenize 默认 special=True。

    基类 ``embed()`` 内部用 ``self.tokenize(text)`` 分词，基类默认
    ``special=False``——文本里的 "</s>" 会被切成字面字符而不是真实 SEP
    token，rerank 分数就废了。**必须子类化**（而不是组合 + __getattr__
    委托）：embed 是基类方法，里面的 ``self.tokenize`` 只会解析到实例
    真实类型的方法链，委托包装器拦不到它。
    """
    import llama_cpp

    class _RerankLlama(llama_cpp.Llama):
        def tokenize(self, text, add_bos=True, special=True):
            return super().tokenize(text, add_bos=add_bos, special=special)

    return _RerankLlama


class GGUFReranker:
    """GGUF 交叉编码器 reranker（llama.cpp RANK 池化，经 llama-cpp-python）。

    输入格式遵循 llama.cpp server rerank 端点约定：``query + sep + doc``。
    分隔符优先读 GGUF 元数据 ``tokenizer.ggml.separation_token(_id)``
    （bge-reranker 系 GGUF 为 </s>），缺失时回退 </s>（XLMRoberta 系）。
    ``embed()`` 在 RANK 池化下每条序列输出分类头分数——**原始 logit**，
    与 ONNX/MLX 路径语义一致，由 vector_store 的 _logit_to_score 做
    sigmoid 归一化到 [0,1]。
    """

    def __init__(self, model_path: str, *, threads: int = 4,
                 max_length: int = _RERANK_DEFAULT_MAX_LENGTH,
                 batch_size: int = 16):
        try:
            import llama_cpp
        except ImportError as e:
            raise RuntimeError(
                "GGUF Reranker 需要 llama-cpp-python：pip install llama-cpp-python"
            ) from e
        if not hasattr(llama_cpp, "LLAMA_POOLING_TYPE_RANK"):
            raise RuntimeError(
                "llama-cpp-python 版本过旧（无 LLAMA_POOLING_TYPE_RANK），"
                "请升级后重试"
            )
        self._max_length = max(8, int(max_length))
        self._batch_size = max(1, int(batch_size))
        self._lock = threading.RLock()
        # rerank 输入 = query + sep + doc（单片 ≤ 千余字符），上下文开到
        # min(max_length, 4096) 足够；n_batch 必须同步给足，否则 embed()
        # 的 truncate 会把输入截到 512 token（n_batch 默认值）。
        n_ctx = min(self._max_length, 4096)
        self._llm = _rerank_llama_cls()(
            model_path=model_path,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_threads=max(1, int(threads)),
            verbose=False,
        )
        self._sep = self._sep_from_metadata()
        logger.info(
            "GGUF Reranker 加载完成: %s (sep=%r, n_ctx=%d)", model_path, self._sep, n_ctx
        )

    def _sep_from_metadata(self) -> str:
        """从 GGUF 元数据解析分隔符；bge 系 GGUF 带 </s>（XLMRoberta SEP）。"""
        meta = getattr(self._llm, "metadata", None) or {}
        text = meta.get("tokenizer.ggml.separation_token")
        if text:
            return str(text)
        sep_id = meta.get("tokenizer.ggml.separation_token_id")
        if sep_id is not None and str(sep_id).strip().lstrip("-").isdigit():
            try:
                piece = self._llm.detokenize([int(sep_id)])
                if piece:
                    return piece.decode("utf-8", errors="replace")
            except Exception:
                pass
        return "</s>"

    def rerank(self, query: str, documents: Iterable[str],
               batch_size: int = 16, **kwargs) -> List[float]:
        docs = list(documents)
        size = max(1, int(batch_size or self._batch_size))
        scores: List[float] = []
        with self._lock:
            for start in range(0, len(docs), size):
                batch = [
                    f"{query}{self._sep}{doc}" for doc in docs[start:start + size]
                ]
                embeddings = self._llm.embed(batch)
                for emb in embeddings:
                    # RANK 池化：每条序列输出 [score]（单元素向量）
                    scores.append(float(emb[0] if isinstance(emb, list) else emb))
        return scores

    def close(self) -> None:
        with self._lock:
            llm = getattr(self, "_llm", None)
            if llm is not None:
                try:
                    llm.close()
                except Exception:
                    pass
            self._llm = None


class FastEmbedReranker:
    def __init__(self, model_name: str, *, threads: int = 4):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError:
            from fastembed import TextCrossEncoder

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastembed")
        os.makedirs(cache_dir, exist_ok=True)
        self._model = TextCrossEncoder(
            model_name,
            cache_dir=cache_dir,
            providers=["CPUExecutionProvider"],
            threads=max(1, int(threads)),
        )

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 16, **kwargs):
        return list(self._model.rerank(query, documents, batch_size=batch_size, **kwargs))


def mlx_runtime_blocker() -> str:
    """Explain why MLX cannot run here, or "" when it is usable.

    MLX 只在 Apple Silicon macOS 上可用。默认后端是 MLX，非兼容机器若直接抛错
    会让 RAG 整体不可用，所以调用方据此回退到 fastembed CPU。
    """
    if sys.platform != "darwin":
        return f"MLX 仅支持 macOS，当前为 {sys.platform}"
    if platform.machine() != "arm64":
        return f"MLX 需要 Apple Silicon，当前为 {platform.machine()}"
    try:
        import mlx.core  # noqa: F401
    except ImportError as e:
        return f"未安装 MLX（{e}）"
    return ""


_JINA_TASK_QUERY = "retrieval.query"
_JINA_TASK_PASSAGE = "retrieval.passage"
# jina-embeddings-v3 ONNX 单文件的 task → task_id（模型输入，见 example/example_onnx.py）。
_JINA_ONNX_TASK_ID = {
    "retrieval.query": 0,
    "retrieval.passage": 1,
    "separation": 2,
    "classification": 3,
    "text-matching": 4,
}
# task_prompt_map 缺失时的兜底（与模型 config.json / example 一致）。
_JINA_DEFAULT_TASK_PROMPTS = {
    "retrieval.query": "Represent the query for retrieving evidence documents: ",
    "retrieval.passage": "Represent the document for retrieval: ",
    "separation": "Represent the sentence for retrieving duplicate sentences: ",
    "classification": "Represent the document for classifying: ",
    "text-matching": "Represent the sentence for retrieving the most similar sentence: ",
}


def _jina_compose(texts: List[str], task: str, task_prompt: str,
                  label_prefix: str) -> List[str]:
    """jina-embeddings-v3 输入组装：任务提示词 + 标签前缀 + 原文。

    任务提示词（task_prompt_map，模型训练时的约定）在最前，标签前缀
    （"passage: "/"query: "，embedding_prefixes 按模型族给出）紧贴原文。
    """
    return [task_prompt + _eprefix.apply_prefix(label_prefix, t) for t in texts]


class JinaMLXEmbeddingFunction:
    """jina-embeddings-v3 本地 MLX 直载后端。

    mlx-lm / mlx-embeddings 均不支持该单文件统一导出（model_type=jina-v3-mlx），
    模型结构与加载走参考实现 rag/jina_v3_mlx.py（vendor 自模型目录 example）。
    任务类型经 encode(task=...) 在运行时选择 LoRA 适配器：文档分片用
    retrieval.passage、检索查询用 retrieval.query。输出 1024 维、已 L2 归一化；
    Matryoshka 截 256 维 / 重归一化 / int8 量化由 VectorStore 统一处理。
    """

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 8,
        max_length: int = 512,
        document_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ):
        self._model_path = resolve_model_path(model_path)
        self._batch_size = max(1, int(batch_size))
        self._max_length = max(64, int(max_length))
        _doc_p, _query_p = _eprefix.prefixes(self._model_path)
        self._document_prefix = (
            document_prefix if document_prefix is not None else _doc_p)
        self._query_prefix = (
            query_prefix if query_prefix is not None else _query_p)
        self._model = None
        self._tokenizer = None
        self._task_prompts: Dict[str, str] = {}
        self._load_lock = threading.RLock()
        self._embed_lock = threading.RLock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            _cap_mlx_memory_limits()
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, local_files_only=True, trust_remote_code=False)
            from .jina_v3_mlx import load_model

            self._model, cfg = load_model(self._model_path)
            self._task_prompts = dict(cfg.get("task_prompt_map")
                                      or _JINA_DEFAULT_TASK_PROMPTS)
            logger.info("Jina MLX Embedding 加载完成: %s (unified=%s)",
                        self._model_path, cfg.get("unified"))

    def _embed(self, texts: List[str], task: str) -> List[np.ndarray]:
        if not texts:
            return []
        self._ensure_model()
        import mlx.core as mx
        import numpy as np

        prompt = self._task_prompts.get(task, "")
        label = (self._document_prefix if task == _JINA_TASK_PASSAGE
                 else self._query_prefix)
        out: List[np.ndarray] = []
        with self._embed_lock:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start:start + self._batch_size]
                prefixed = _jina_compose(batch, task, prompt, label)
                enc = self._tokenizer(
                    prefixed, padding=True, truncation=True,
                    max_length=self._max_length, return_tensors="np",
                )
                emb = self._model.encode(
                    mx.array(enc["input_ids"]),
                    mx.array(enc["attention_mask"]),
                    task=task,
                )
                out.extend(np.asarray(emb, dtype=np.float32))
        return out

    def __call__(self, input: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        return self._embed(input, _JINA_TASK_PASSAGE)

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        return self(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], _JINA_TASK_QUERY)[0]

    def name(self) -> str:
        return f"jina_mlx:{self._model_path}"

    def close(self) -> None:
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass


class JinaONNXEmbeddingFunction:
    """jina-embeddings-v3 单文件 ONNX 后端（task_id 输入选择任务 LoRA）。

    单 model.onnx 服务全部 5 个任务：输入 input_ids / attention_mask（int64）
    外加标量 task_id（retrieval.query=0、retrieval.passage=1，见模型目录
    example/example_onnx.py）。输出 (B, 1024) 已 mean-pool + L2 归一化。
    """

    def __init__(
        self,
        model_path: str,
        *,
        threads: int = 4,
        batch_size: int = 8,
        max_length: int = 512,
        document_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ):
        self._model_file, self._model_dir = _find_onnx(model_path)
        self._threads = max(1, int(threads))
        self._batch_size = max(1, int(batch_size))
        self._max_length = max(64, int(max_length))
        _doc_p, _query_p = _eprefix.prefixes(f"{self._model_dir} {model_path}")
        self._document_prefix = (
            document_prefix if document_prefix is not None else _doc_p)
        self._query_prefix = (
            query_prefix if query_prefix is not None else _query_p)
        self._session = None
        self._tokenizer = None
        self._task_prompts: Dict[str, str] = {}
        self._lock = threading.RLock()

    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import json
            import onnxruntime as ort
            from transformers import AutoTokenizer

            options = ort.SessionOptions()
            options.intra_op_num_threads = self._threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self._model_file, sess_options=options,
                providers=["CPUExecutionProvider"])
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_dir, local_files_only=True, trust_remote_code=False)
            try:
                with open(os.path.join(self._model_dir, "config.json"),
                          encoding="utf-8") as f:
                    self._task_prompts = dict(
                        json.load(f).get("task_prompt_map")
                        or _JINA_DEFAULT_TASK_PROMPTS)
            except Exception:
                self._task_prompts = dict(_JINA_DEFAULT_TASK_PROMPTS)
            logger.info("Jina ONNX Embedding 加载完成(CPU): %s", self._model_file)

    def _embed(self, texts: List[str], task: str) -> List[np.ndarray]:
        if not texts:
            return []
        self._ensure_model()
        import numpy as np

        prompt = self._task_prompts.get(task, "")
        label = (self._document_prefix if task == _JINA_TASK_PASSAGE
                 else self._query_prefix)
        task_id = _JINA_ONNX_TASK_ID.get(task)
        if task_id is None:
            raise ValueError(f"jina-embeddings-v3 不支持任务类型: {task}")
        out: List[np.ndarray] = []
        with self._lock:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start:start + self._batch_size]
                prefixed = _jina_compose(batch, task, prompt, label)
                enc = self._tokenizer(
                    prefixed, padding=True, truncation=True,
                    max_length=self._max_length, return_tensors="np")
                values = self._session.run(None, {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                    "task_id": np.array(task_id, dtype=np.int64),
                })[0]
                out.extend(np.asarray(values, dtype=np.float32))
        return out

    def __call__(self, input: List[str]) -> List[np.ndarray]:
        return self._embed(input, _JINA_TASK_PASSAGE)

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        return self(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], _JINA_TASK_QUERY)[0]

    def name(self) -> str:
        return f"jina_onnx:{self._model_file}"

    def close(self) -> None:
        with self._lock:
            self._session = None
            self._tokenizer = None


class JinaV5NanoEmbeddingFunction:
    """jina-embeddings-v5-text-nano 本地 MLX 直载后端（EuroBERT 骨干）。

    mlx-lm / mlx-embeddings / fastembed 均不支持该模型，模型结构与加载走
    参考实现 ``rag/jina_v5_nano_mlx.py``（vendor 自模型目录 model.py/utils.py，
    见 models/jina-embeddings-v5-text-nano-mlx-int4）。

    任务类型用**内部任务类型** ``"query"`` / ``"doc"``：检索查询用 query、
    文档分片用 doc。该模型目录的 int4 量化版已**预合并 retrieval LoRA**
    （packed 量化权重无法原位切适配器），query/doc 仅靠文本前缀区分：
    query 加 ``"Query: "``、doc 加 ``"Document: "``（模型 encode 的
    task_type 参数只决定前缀）。

    输出全 768 维、已 L2 归一化 fp32；Matryoshka 截 128 维 / 重归一化 /
    int8 量化由 VectorStore 统一管线处理（入库/查询对称，见
    vector_store._prepare_query）。
    """

    # 内部任务类型 → 模型侧 task_type（决定 "Query: " / "Document: " 前缀）。
    _TASK_QUERY = "query"
    _TASK_DOC = "doc"
    _TASK_MAP = {
        "query": "retrieval.query",
        "doc": "retrieval.passage",
    }

    def __init__(self, model_path: str, *, batch_size: int = 8,
                 max_length: int = 1024):
        self._model_path = resolve_model_path(model_path)
        self._batch_size = max(1, int(batch_size))
        self._max_length = max(64, int(max_length))
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.RLock()
        self._embed_lock = threading.RLock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            _cap_mlx_memory_limits()
            from .jina_v5_nano_mlx import load_model

            self._model, self._tokenizer, cfg = load_model(self._model_path)
            logger.info("Jina v5-nano MLX Embedding 加载完成: %s (bits=%s)",
                        self._model_path,
                        (cfg.get("quantization") or {}).get("bits"))

    def _embed(self, texts: List[str], task: str) -> List[np.ndarray]:
        if not texts:
            return []
        if task not in self._TASK_MAP:
            raise ValueError(
                f"未知任务类型: {task!r}（可用: query / doc）")
        self._ensure_model()
        import mlx.core as mx
        import numpy as np

        model_task = self._TASK_MAP[task]
        out: List[np.ndarray] = []
        with self._embed_lock:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start:start + self._batch_size]
                emb = self._model.encode(
                    batch, self._tokenizer,
                    max_length=self._max_length,
                    task_type=model_task,
                )
                out.extend(np.asarray(emb.astype(mx.float32),
                                      dtype=np.float32))
        return out

    def __call__(self, input: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        return self._embed(input, self._TASK_DOC)

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        return self(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], self._TASK_QUERY)[0]

    def name(self) -> str:
        return f"jina_v5_nano:{self._model_path}"

    def close(self) -> None:
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass


class JinaV5NanoONNXEmbeddingFunction:
    """jina-embeddings-v5-text-nano 本地 ONNX 后端（EuroBERT 骨干，CPU-only）。

    与 MLX 后端（JinaV5NanoEmbeddingFunction）同模型、不同运行时：用
    onnxruntime 的 CPUExecutionProvider 直接推理，不依赖 MLX / torch，适合
    跨平台无 GPU 环境。模型目录参考 ``models/jina-embeddings-v3-onnx-int4``
    的结构（单 model_quantized.onnx + tokenizer.json + config.json）。

    与 v3 ONNX（JinaONNXEmbeddingFunction，单文件服务全部任务、靠 task_id
    输入切 LoRA）不同：v5 量化版已**预合并 retrieval LoRA**，ONNX 输入只有
    ``input_ids`` / ``attention_mask``（无 task_id），query/doc 仅靠文本前缀
    区分——查询加 ``"Query: "``、文档加 ``"Document: "``（前缀从模型目录
    config.json 读取，见 JinaV5NanoEmbeddingFunction）。输出全 768 维、图内
    已 L2 归一化；Matryoshka 截 128 维 / 重归一化 / int8 量化由 VectorStore
    统一管线处理（入库/查询对称）。
    """

    # 内部任务类型 → 前缀键（config.json 的 prefixes 映射）。
    _TASK_QUERY = "query"
    _TASK_DOC = "doc"

    def __init__(self, model_path: str, *, threads: int = 4,
                 batch_size: int = 8, max_length: int = 1024):
        self._model_file, self._model_dir = _find_onnx(model_path)
        self._threads = max(1, int(threads))
        self._batch_size = max(1, int(batch_size))
        self._max_length = max(64, int(max_length))
        self._session = None
        self._tokenizer = None
        self._query_prefix = "Query: "
        self._doc_prefix = "Document: "
        self._lock = threading.RLock()

    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import json

            import onnxruntime as ort
            from tokenizers import Tokenizer

            options = ort.SessionOptions()
            options.intra_op_num_threads = self._threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self._model_file, sess_options=options,
                providers=["CPUExecutionProvider"])
            self._tokenizer = Tokenizer.from_file(
                os.path.join(self._model_dir, "tokenizer.json"))
            try:
                with open(os.path.join(self._model_dir, "config.json"),
                          encoding="utf-8") as f:
                    cfg = json.load(f)
                _prefixes = cfg.get("prefixes") or {}
                self._query_prefix = _prefixes.get("query", "Query: ")
                self._doc_prefix = _prefixes.get("doc", "Document: ")
            except Exception:
                pass
            logger.info("Jina v5-nano ONNX Embedding 加载完成(CPU): %s",
                        self._model_file)

    def _embed(self, texts: List[str], task: str) -> List[np.ndarray]:
        if not texts:
            return []
        if task == self._TASK_QUERY:
            prefix = self._query_prefix
        elif task == self._TASK_DOC:
            prefix = self._doc_prefix
        else:
            raise ValueError(
                f"未知任务类型: {task!r}（可用: query / doc）")
        self._ensure_model()
        import numpy as np

        prefixed = [prefix + t for t in texts]
        self._tokenizer.enable_truncation(self._max_length)
        encs = self._tokenizer.encode_batch(prefixed)
        max_len = max(len(e.ids) for e in encs)
        ids = np.array(
            [e.ids + [0] * (max_len - len(e.ids)) for e in encs],
            dtype=np.int64)
        attn = np.array(
            [e.attention_mask + [0] * (max_len - len(e.ids)) for e in encs],
            dtype=np.int64)
        with self._lock:
            out = self._session.run(None, {
                "input_ids": ids,
                "attention_mask": attn,
            })[0]
        return [np.asarray(v, dtype=np.float32) for v in out]

    def __call__(self, input: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        return self._embed(input, self._TASK_DOC)

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        return self(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], self._TASK_QUERY)[0]

    def name(self) -> str:
        return f"jina_v5_nano_onnx:{self._model_file}"

    def close(self) -> None:
        with self._lock:
            self._session = None
            self._tokenizer = None


def _fastembed_embedding(model_name: str, threads: int):
    from .vector_store import FastEmbedEmbeddingFunction

    return FastEmbedEmbeddingFunction(model_name=model_name, threads=threads)


def create_embedding_backend(
    backend: str,
    model_name: str,
    model_path: str,
    *,
    threads: int = 4,
    batch_size: int = 8,
    max_length: int = _EMBEDDING_DEFAULT_MAX_LENGTH,
):
    backend = (backend or "fastembed").lower()
    # jina-embeddings-v3：mlx-lm/mlx-embeddings/fastembed 均不支持该导出，
    # 必须走专用后端；且不允许静默回退 fastembed（会错成别的模型）。
    ident = f"{model_name or ''} {model_path or ''}".lower()
    if "jina-embeddings-v5" in ident:
        # v5-text-nano（EuroBERT 骨干）：支持 mlx / onnx 两种后端，但都走专用
        # 后端（接口与 nomic/v3 不同），不允许静默回退（回退会错成别的模型、
        # 向量空间完全不同）。
        if backend == "mlx":
            blocker = mlx_runtime_blocker()
            if blocker:
                raise RuntimeError(
                    f"jina-embeddings-v5 需要 MLX 运行时（当前不可用: {blocker}）。"
                    "可改用 onnx 后端 + models/jina-embeddings-v5-text-nano-onnx-int8。")
            return JinaV5NanoEmbeddingFunction(
                model_path, batch_size=batch_size, max_length=max_length)
        if backend == "onnx":
            return JinaV5NanoONNXEmbeddingFunction(
                model_path, threads=threads, batch_size=batch_size,
                max_length=max_length)
        raise ValueError(
            "jina-embeddings-v5 仅支持 mlx / onnx 后端"
            f"（当前后端 {backend!r} 无此模型）")
    if "jina-embeddings-v3" in ident:
        if backend == "mlx":
            blocker = mlx_runtime_blocker()
            if blocker:
                raise RuntimeError(
                    f"jina-embeddings-v3 需要 MLX 运行时（当前不可用: {blocker}）。"
                    "可改用 ONNX 后端 + models/jina-embeddings-v3-onnx-int4。")
            return JinaMLXEmbeddingFunction(
                model_path, batch_size=batch_size, max_length=max_length)
        if backend == "onnx":
            return JinaONNXEmbeddingFunction(
                model_path, threads=threads, batch_size=batch_size,
                max_length=max_length)
        raise ValueError(
            "jina-embeddings-v3 仅支持 mlx / onnx 后端"
            f"（当前后端 {backend!r} 无此模型）")
    if backend == "mlx":
        blocker = mlx_runtime_blocker()
        if blocker:
            logger.warning("MLX Embedding 不可用（%s），回退 fastembed CPU: %s", blocker, model_name)
            return _fastembed_embedding(model_name, threads)
        try:
            return MLXEmbeddingFunction(
                model_path,
                batch_size=batch_size,
                max_length=max_length,
            )
        except Exception as e:
            # 路径缺失 / 模型类型不被 mlx-embeddings 支持时不要让建库整体失败。
            logger.warning("MLX Embedding 初始化失败（%s: %s），回退 fastembed CPU: %s",
                           type(e).__name__, e, model_name)
            return _fastembed_embedding(model_name, threads)
    if backend == "onnx":
        return ONNXEmbeddingFunction(
            model_path,
            threads=threads,
            batch_size=batch_size,
            max_length=max_length,
        )
    if backend == "llama_cpp":
        return GGUFEmbeddingFunction(
            model_path,
            threads=threads,
            batch_size=batch_size,
        )
    raise ValueError(f"未知本地 Embedding 后端: {backend}")


def create_reranker_backend(
    backend: str,
    model_name: str,
    model_path: str,
    *,
    threads: int = 4,
    max_length: int = _RERANK_DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
    mlx_model_path: str = "",
    jina_model_path: str = "",
):
    backend = (backend or "fastembed").lower()
    if backend == "disabled":
        return None
    if backend == "onnx":
        return ONNXReranker(model_path, threads=threads, max_length=max_length)
    if backend == "gguf":
        # GGUF 交叉编码器（llama.cpp RANK 池化）：模型文件是单 .gguf，与
        # ONNX 共用 rag_rerank_model_path 字段（UI 路径框同一个）。
        return GGUFReranker(model_path, threads=threads, max_length=max_length,
                            batch_size=batch_size)
    if backend == "jina":
        # jina-reranker-v2-base-multilingual 的 INT8 ONNX 导出，强制 CPU，与
        # bge ONNX 共用 ONNXReranker（按目录自动找 model_quantized.onnx + tokenizer）。
        return ONNXReranker(jina_model_path or model_path, threads=threads, max_length=max_length)
    if backend == "mlx":
        blocker = mlx_runtime_blocker()
        if not blocker:
            try:
                return MLXReranker(
                    mlx_model_path or model_path,
                    batch_size=batch_size,
                    max_length=max_length,
                )
            except Exception as e:
                blocker = f"{type(e).__name__}: {e}"
        logger.warning("MLX Reranker 不可用（%s），回退 fastembed CPU: %s", blocker, model_name)
        return FastEmbedReranker(model_name, threads=threads)
    if backend == "fastembed":
        return FastEmbedReranker(model_name, threads=threads)
    raise ValueError(f"未知 Reranker 后端: {backend}")
