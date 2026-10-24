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

logger = logging.getLogger(__name__)

EMBEDDING_BACKENDS = ("fastembed", "mlx", "onnx", "llama_cpp")
# UI 下拉只提供前三项；"disabled" 仅由配置文件的 rag_rerank_enabled=false
# 间接产生（见 VectorStore.__init__），create_reranker_backend 仍容错处理。
RERANKER_BACKENDS = ("fastembed", "mlx", "onnx")

# MLX 内存策略：复用聊天后端同款 apply_mlx_memory_policy（#fix: 同步期
# phys_footprint 冲到 18-19GB）。MLX 默认 memory_limit = 95% RAM（16G 机器
# ~15G），buffer cache 会一路涨到系统上限；但单纯把 cache 压到 1GB 又会
# 在 prefill 长上下文时让 `[seq,seq]` 注意力矩阵每步反复释放重分配 → 生成
# 抖动卡顿。统一策略：总内存封顶 ~65% RAM（防 OS 换页卡死、给系统留余量）、buffer cache
# 封顶 ~6GB（既能封顶又足够跨层复用）。聊天后端与 RAG 共用，保证一致。
_MLX_MEM_CAP_DONE = False


def _cap_mlx_memory_limits() -> None:
    """限制 MLX 内存，避免同步期统一内存被 cache 吃满或 prefill 抖动。

    直接复用聊天后端的统一策略 apply_mlx_memory_policy（见 model_backends.py），
    保证两端内存上限一致、互不压低。
    """
    global _MLX_MEM_CAP_DONE
    if _MLX_MEM_CAP_DONE:
        return
    apply_mlx_memory_policy()
    _MLX_MEM_CAP_DONE = True

_DEFAULT_DOC_PREFIX = "search_document: "
_DEFAULT_QUERY_PREFIX = "search_query: "

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
        document_prefix: str = _DEFAULT_DOC_PREFIX,
        query_prefix: str = _DEFAULT_QUERY_PREFIX,
    ):
        self._model_path = resolve_model_path(model_path)
        self._batch_size = max(1, int(batch_size))
        # 请求值先记下，模型加载后按 config 的位置/旋转上限夹紧（见 _model_max_length）。
        self._max_length_requested = max(8, int(max_length))
        self._max_length = self._max_length_requested
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._model = None
        self._processor = None
        self._generate = None
        self._model_type = ""
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
                except Exception as e:
                    raise RuntimeError(
                        f"mlx-embeddings 不支持或无法加载模型类型 '{self._model_type}': {e}"
                    ) from e
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
            text if text.startswith(self._document_prefix) else self._document_prefix + text
            for text in input
        ]
        return self._embed(prefixed, batch_size=batch_size)

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        return self(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> List[float]:
        value = text if text.startswith(self._query_prefix) else self._query_prefix + text
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
        document_prefix: str = _DEFAULT_DOC_PREFIX,
        query_prefix: str = _DEFAULT_QUERY_PREFIX,
    ):
        self._model_file, self._model_dir = _find_onnx(model_path)
        self._threads = max(1, int(threads))
        self._batch_size = max(1, int(batch_size))
        # 模型加载后按 config 的位置/旋转上限夹紧（见 _model_max_length）。
        self._max_length_requested = max(8, int(max_length))
        self._max_length = self._max_length_requested
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
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
            text if text.startswith(self._document_prefix) else self._document_prefix + text
            for text in input
        ])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        value = text if text.startswith(self._query_prefix) else self._query_prefix + text
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
        return self._embed(input)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

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
):
    backend = (backend or "fastembed").lower()
    if backend == "disabled":
        return None
    if backend == "onnx":
        return ONNXReranker(model_path, threads=threads, max_length=max_length)
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
