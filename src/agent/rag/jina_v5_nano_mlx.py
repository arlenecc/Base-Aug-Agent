"""jina-embeddings-v5-text-nano 本地 MLX 直载参考实现。

vendor 自模型目录 ``models/jina-embeddings-v5-text-nano-mlx-int4`` 的
``model.py`` / ``utils.py``（EuroBERT-210m 骨干的纯 MLX 移植，零
PyTorch/transformers 依赖）。与 ``jina_v3_mlx.py`` 同一套 vendor 约定：
模型目录里的实现是权威版本，本文件随其同步。

要点：
- 双向编码器（无因果掩码）、last-token 池化、L2 归一化；768 维。
- 该目录的 int4 量化版已**预合并 retrieval LoRA**（packed 量化权重无法
  原位切适配器），是 retrieval 单任务模型；query/passage 仅靠文本前缀区分：
  查询加 ``"Query: "``、文档加 ``"Document: "``（model.encode 的
  task_type 参数只决定前缀，不做 LoRA 切换）。
- Matryoshka：``encode(..., truncate_dim=N)`` 截取前 N 维后自动重新归一化。
  项目里**不在模型侧截维**——后端输出全 768 维，截维/重归一化/int8 量化
  由 VectorStore 统一管线处理（入库/查询对称，见 vector_store._prepare_query）。

用法::

    model, tok, config = load_model(model_dir)
    emb = encode(model, tok, ["Query: 如何做菜?"])   # (1, 768) fp32 已归一化
"""

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    head_dim: int
    tie_word_embeddings: bool
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    attention_bias: bool = False


def initialize_rope(head_dim: int, base: float = 250000.0):
    """Create an nn.RoPE instance for EuroBERT.

    Uses traditional=False (rotate_half convention) to match PyTorch's
    implementation. nn.RoPE internally uses mx.fast.rope for optimized
    computation.
    """
    return nn.RoPE(head_dim, traditional=False, base=base)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        # EuroBERT does NOT have q_norm/k_norm
        self.rotary_emb = initialize_rope(head_dim, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rotary_emb(queries)
        keys = self.rotary_emb(keys)

        # GQA（Grouped Query Attention）按需展开
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size,
                       bias=args.attention_bias)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class EuroBertModel(nn.Module):
    """EuroBERT encoder model (bidirectional, no causal mask)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args)
                       for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None):
        h = self.embed_tokens(inputs)

        for layer in self.layers:
            h = layer(h, mask)

        return self.norm(h)


class JinaEmbeddingModel(nn.Module):
    """Jina v5-text-nano embedding model with last-token pooling."""

    def __init__(self, config: dict):
        super().__init__()
        args = ModelArgs(**config)
        self.model = EuroBertModel(args)
        self.config = config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):
        """Forward pass with last-token pooling and L2 normalization."""
        batch_size, seq_len = input_ids.shape

        # 双向编码器——只用 padding 掩码，无因果掩码。attention_mask 为
        # [batch, seq_len]（1=真实 token，0=padding），转加性掩码后广播到
        # [batch, 1, 1, seq_len]；dtype 对齐模型权重（fp16 推理必需）。
        if attention_mask is not None:
            padding_mask = mx.where(attention_mask == 0, -1e9, 0.0)
            mask = mx.expand_dims(mx.expand_dims(padding_mask, 1), 1)
            h_dtype = self.model.embed_tokens.weight.dtype
            if mask.dtype != h_dtype:
                mask = mask.astype(h_dtype)
        else:
            mask = None

        hidden_states = self.model(input_ids, mask)  # [B, L, H]

        # last-token 池化：取每条序列最后一个非 padding token
        if attention_mask is not None:
            sequence_lengths = mx.sum(attention_mask, axis=1) - 1  # [B]
            batch_indices = mx.arange(hidden_states.shape[0])
            embeddings = hidden_states[batch_indices, sequence_lengths]
        else:
            embeddings = hidden_states[:, -1, :]

        norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings

    def encode(
        self,
        texts: list,
        tokenizer,
        max_length: int = 8192,
        truncate_dim: Optional[int] = None,
        task_type: str = "retrieval.query",
    ):
        """Encode texts to embeddings.

        task_type 仅决定文本前缀（量化版 LoRA 已合并，无运行时切换）：
        retrieval.query → "Query: "，retrieval.passage 等文档类 → "Document: "。
        返回 [batch, dim]，已 L2 归一化；truncate_dim 截维后自动重归一化。
        """
        prefix_map = {
            "retrieval.query": "Query: ",
            "retrieval.passage": "Document: ",
            "classification": "Document: ",
            "text-matching": "Document: ",
            "clustering": "Document: ",
        }
        prefix = prefix_map.get(task_type, "")

        if prefix:
            texts = [prefix + text for text in texts]

        encodings = tokenizer.encode_batch(texts)

        max_len = min(max_length, max(len(enc.ids) for enc in encodings))
        input_ids = []
        attention_mask = []

        for encoding in encodings:
            ids = encoding.ids[:max_len]
            mask = encoding.attention_mask[:max_len]

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [0] * pad_len
                mask = mask + [0] * pad_len

            input_ids.append(ids)
            attention_mask.append(mask)

        input_ids = mx.array(input_ids)
        attention_mask = mx.array(attention_mask)

        embeddings = self(input_ids, attention_mask)

        if truncate_dim is not None:
            embeddings = embeddings[:, :truncate_dim]
            norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings


def load_model(model_dir: str) -> Tuple[JinaEmbeddingModel, Any, Dict[str, Any]]:
    """加载模型目录（含量化 config），返回 ``(model, tokenizer, config)``。

    tokenizer 用 ``tokenizers.Tokenizer.from_file``（模型目录只有
    tokenizer.json，无 transformers 依赖）。
    """
    from tokenizers import Tokenizer

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    _keys = {f.name for f in dataclasses.fields(ModelArgs)}
    model = JinaEmbeddingModel(
        {k: v for k, v in config.items() if k in _keys})
    q = config.get("quantization")
    if q:  # 先按 config 构建量化结构，再加载 packed 权重
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"])
    weights = mx.load(os.path.join(model_dir, "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    del weights
    return model, Tokenizer.from_file(
        os.path.join(model_dir, "tokenizer.json")), config


def encode(model, tokenizer, texts, *, max_length: int = 8192,
           task_type: str = "retrieval.query"):
    """便捷封装：encode → fp32 numpy 可用数组（bfloat16 权重 → fp32）。"""
    emb = model.encode(texts, tokenizer, max_length=max_length,
                       task_type=task_type)
    return emb.astype(mx.float32)
