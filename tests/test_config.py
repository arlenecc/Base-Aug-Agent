"""Tests for AgentConfig: persistence, defaults, and migration of stale values."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from agent.config import AgentConfig


def test_load_returns_defaults_when_missing(tmp_path):
    p = tmp_path / "missing.json"
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"
    assert cfg.rag_embedding_model_path == "models/nomic-embed-text-v1.5-mlx-int8"
    assert cfg.rag_chunk_size == 800
    assert cfg.rag_matryoshka_dim == 512
    assert cfg.base_url == "https://api.openai.com/v1"


def test_load_preserves_known_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-abc",
        "model": "deepseek-chat",
        "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5-Q",
        "rag_chunk_size": 800,
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.base_url == "https://api.deepseek.com/v1"
    assert cfg.api_key == "sk-abc"
    assert cfg.model == "deepseek-chat"
    assert cfg.rag_chunk_size == 800


def test_load_ignores_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "obsolete_field": "should_be_ignored",
        "another_unknown": 123,
    }))
    # Should not raise; unknown keys silently dropped.
    cfg = AgentConfig.load(str(p))
    assert cfg.base_url == "http://x"
    assert not hasattr(cfg, "obsolete_field")


def test_save_round_trip(tmp_path):
    p = tmp_path / "c.json"
    cfg = AgentConfig(base_url="http://api.example.com/v1", api_key="k", model="m")
    cfg.rag_chunk_size = 800
    cfg.save(str(p))
    loaded = AgentConfig.load(str(p))
    assert loaded.base_url == "http://api.example.com/v1"
    assert loaded.api_key == "k"
    assert loaded.rag_chunk_size == 800


# ---------------------------------------------------------------------------
# Migration: stale embedding model names from pre-LanceDB era
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stale_name", [
    "all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-large-zh-v1.5",
    "",
])
def test_migrate_deprecated_embedding_models(stale_name, tmp_path):
    """Stale embedding model names saved before the LanceDB migration must be
    silently rewritten to the current default so old config files don't break
    RAG sync."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": stale_name,
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_migrate_preserves_valid_embedding_model(tmp_path):
    """A currently-supported embedding model name must not be touched."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5-Q",
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_migrate_nonquantized_to_quantized(tmp_path):
    """Non-quantized nomic-embed-text-v1.5 (~548MB) must be migrated to the
    quantized -Q variant (~137MB) on load: same 768-dim, 4x smaller."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5",
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_migrate_persisted_after_save(tmp_path):
    """After migration, saving the config should persist the corrected value
    so the migration only runs once."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": "all-MiniLM-L6-v2",
    }))
    cfg = AgentConfig.load(str(p))
    cfg.save(str(p))  # persist the migrated value
    # Reload — should now read the corrected value from disk directly.
    cfg2 = AgentConfig.load(str(p))
    assert cfg2.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_migrate_legacy_rerank_model(tmp_path):
    """FlagEmbedding 时代的旧默认 bge-reranker-base（fp32 1.1GB）迁移到
    fastembed 的 INT8 版；其他显式配置的模型名保持不动。"""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "rag_rerank_model": "BAAI/bge-reranker-base",
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_rerank_model == "BAAI/bge-reranker-v2-m3-int8"

    # 显式配置的其他模型名（如 fp32 官方版）不是旧默认值，不迁移
    p2 = tmp_path / "c2.json"
    p2.write_text(json.dumps({
        "base_url": "http://x",
        "rag_rerank_model": "BAAI/bge-reranker-v2-m3",
    }))
    cfg2 = AgentConfig.load(str(p2))
    assert cfg2.rag_rerank_model == "BAAI/bge-reranker-v2-m3"


def test_migrate_bumps_small_max_tokens(tmp_path):
    """Older configs saved max_tokens=4096 (the old LLMClient default). Migration
    must bump any sub-8192 value to the current default (32768) so users
    upgrading don't keep hitting the tiny old budget."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_tokens": 4096, "max_context_tokens": 4096}))
    cfg = AgentConfig.load(str(p))
    assert cfg.max_tokens == 32768, f"expected 32768, got {cfg.max_tokens}"
    assert cfg.max_context_tokens == 32768, f"expected 32768, got {cfg.max_context_tokens}"


def test_migrate_preserves_large_max_tokens(tmp_path):
    """If the user explicitly set a large max_tokens, migration must not touch it."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_tokens": 16384, "max_context_tokens": 16384}))
    cfg = AgentConfig.load(str(p))
    assert cfg.max_tokens == 16384
    assert cfg.max_context_tokens == 16384


def test_default_max_tokens_is_32768():
    """The default max_tokens must be 32768, not the old 4096."""
    cfg = AgentConfig()
    assert cfg.max_tokens == 32768
    assert cfg.max_context_tokens == 32768


def test_aux_defaults_are_16384_ctx_and_05_temp():
    """辅助模型默认加载参数：上下文 16384、temp 0.5，其它参数不变。"""
    cfg = AgentConfig()
    assert cfg.aux_max_context_tokens == 16384
    assert cfg.aux_temperature == 0.5
    assert cfg.aux_max_tokens == 2048
    assert cfg.aux_top_p == 0.95
    assert cfg.aux_min_p == 0.05
    assert cfg.aux_top_k == 20
    assert cfg.aux_repetition_penalty == 1.0


def test_migrate_aux_legacy_temperature(tmp_path):
    """旧配置里的 aux_temperature=0.2（旧默认）升级到 0.5；
    用户显式调过的值必须保留。"""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"aux_temperature": 0.2}))
    assert AgentConfig.load(str(p)).aux_temperature == 0.5

    p2 = tmp_path / "c2.json"
    p2.write_text(json.dumps({"aux_temperature": 0.7}))
    assert AgentConfig.load(str(p2)).aux_temperature == 0.7


def test_migrate_aux_context_floor(tmp_path):
    """aux_max_context_tokens 是后加字段：旧配置默认 16384；
    被设成比生成长度还小的非法值时兜底回 16384。"""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({}))
    assert AgentConfig.load(str(p)).aux_max_context_tokens == 16384

    p2 = tmp_path / "c2.json"
    p2.write_text(json.dumps({"aux_max_context_tokens": 512}))
    assert AgentConfig.load(str(p2)).aux_max_context_tokens == 16384


def test_ensure_workspace_creates_dir(tmp_path):
    cfg = AgentConfig(workspace=str(tmp_path / "ws"))
    cfg.ensure_workspace()
    assert os.path.isdir(str(tmp_path / "ws"))


def test_coerce_string_bools(tmp_path):
    """`"rag_auto_ingest": "false"` was `bool("false")` == True — the user's
    explicit opt-out silently became opt-in on every load."""
    for raw, expected in [("false", False), ("no", False), ("0", False),
                          ("true", True), ("1", True)]:
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"rag_auto_ingest": raw}))
        assert AgentConfig.load(str(p)).rag_auto_ingest is expected, raw


def test_coerce_null_string_fields_fall_back_to_default(tmp_path):
    """`"api_key": null` used to become the literal string "None" — then
    `base_url + "/models"` and friends used it as a real value, and
    `workspace = "None"` created a directory literally named None."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"api_key": None, "workspace": None}))
    cfg = AgentConfig.load(str(p))
    assert cfg.api_key == AgentConfig().api_key
    assert cfg.workspace == AgentConfig().workspace
    assert cfg.workspace != "None"


# ---------------------------------------------------------------------------
# Robustness: corrupted config files and user-chosen small budgets
# ---------------------------------------------------------------------------

def test_load_recovers_from_corrupted_config(tmp_path):
    """A config.json truncated by a crash (or hand-edited badly) must not make
    the app unreadable: back it up and start from defaults."""
    p = tmp_path / "c.json"
    p.write_text('{"base_url": "http://x", "api_key":')
    cfg = AgentConfig.load(str(p))
    assert cfg.base_url == AgentConfig().base_url
    # The bad file is kept for inspection instead of being silently lost.
    assert os.path.exists(str(p) + ".corrupt")


def test_load_rejects_non_object_root(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(["not", "an", "object"]))
    cfg = AgentConfig.load(str(p))
    assert cfg.base_url == AgentConfig().base_url


def test_load_coerces_string_typed_numbers(tmp_path):
    """Hand-edited configs often quote numbers. They must be coerced back to
    int/float so later arithmetic and comparisons don't blow up."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_tokens": "8192", "temperature": "0.3"}))
    cfg = AgentConfig.load(str(p))
    assert cfg.max_tokens == 8192
    assert isinstance(cfg.max_tokens, int)
    assert cfg.temperature == 0.3
    assert isinstance(cfg.temperature, float)


def test_save_is_atomic(tmp_path):
    """save() must write via temp + rename so a crash can never leave a
    half-written config.json behind."""
    p = tmp_path / "c.json"
    cfg = AgentConfig(base_url="http://x")
    cfg.save(str(p))
    assert os.path.exists(str(p))
    assert not os.path.exists(str(p) + ".tmp")


def test_migrate_keeps_explicitly_small_budget(tmp_path):
    """A deliberately small context budget (e.g. a local 4K model) must survive
    a restart — migration should only rewrite the legacy 4096 default."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"max_tokens": 2048, "max_context_tokens": 2048}))
    cfg = AgentConfig.load(str(p))
    assert cfg.max_tokens == 2048
    assert cfg.max_context_tokens == 2048


def test_local_model_defaults_are_parallel_not_forced():
    cfg = AgentConfig()
    # #140：默认本地模式 + MLX-LM 主模型，启动即加载；辅助模型同样本地 MLX。
    assert cfg.llm_mode == "local"
    assert cfg.llm_backend == "mlx_lm"
    assert cfg.llm_model_path.endswith("qwen3.5-4b-agentic-coder-v4-4bit-mlx")
    assert cfg.aux_llm_backend == "mlx_lm"
    assert cfg.aux_llm_model_path.endswith("Qwen3.5-2B-MLX-4bit")
    # 主模型默认本地化，RAG 两个模型默认也走本地 MLX（2026-10-27 切回 nomic）。
    assert cfg.rag_embedding_backend == "mlx"
    assert cfg.rag_embedding_model_path.endswith("nomic-embed-text-v1.5-mlx-int8")
    assert cfg.rag_rerank_backend == "jina"
    assert cfg.rag_rerank_mlx_model_path.endswith("bge-reranker-v2-m3-mlx-mxfp8")
    # 回退用的 fastembed 目录名必须保留，否则 MLX 不可用时无路可退。
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5-Q"
    assert cfg.rag_rerank_model == "BAAI/bge-reranker-v2-m3-int8"
    assert cfg.rag_rerank_mlx_model_path.endswith("bge-reranker-v2-m3-mlx-mxfp8")


def test_llm_mode_and_backend_stay_coherent():
    """模式与后端必须自洽：接口模式 ⇒ openai；本地模式 ⇒ 非 openai（#140）。"""
    # 接口模式下 llm_backend 被强制成 openai。
    cfg = AgentConfig(llm_mode="openai", llm_backend="mlx_lm")
    cfg._migrate()
    assert cfg.llm_mode == "openai"
    assert cfg.llm_backend == "openai"
    # 本地模式却配了 openai：按本地模式修正，回退 mlx_lm（否则本地路径字段
    # 被忽略，配置窗口显示的模式与实际行为相反）。
    cfg = AgentConfig(llm_mode="local", llm_backend="openai")
    cfg._migrate()
    assert cfg.llm_mode == "local"
    assert cfg.llm_backend == "mlx_lm"
    # 未知模式退回本地。
    cfg = AgentConfig(llm_mode="bogus")
    cfg._migrate()
    assert cfg.llm_mode == "local"


def test_legacy_config_without_llm_mode_infers_from_backend(tmp_path):
    """旧配置没有 llm_mode：不能把一直用 OpenAI 接口的用户静默切到本地（#140）。"""
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({
        "llm_backend": "openai",
        "base_url": "https://api.example.com/v1",
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.llm_mode == "openai"
    assert cfg.llm_backend == "openai"

    # 旧配置本就是本地后端 → 推为本地模式。
    p2 = tmp_path / "legacy-local.json"
    p2.write_text(json.dumps({"llm_backend": "mlx_lm"}))
    cfg2 = AgentConfig.load(str(p2))
    assert cfg2.llm_mode == "local"
    assert cfg2.llm_backend == "mlx_lm"


def test_local_model_config_round_trip(tmp_path):
    p = tmp_path / "c.json"
    cfg = AgentConfig(
        llm_backend="mlx_lm",
        rag_embedding_backend="mlx",
        rag_rerank_backend="mlx",
        rag_rerank_mlx_model_path="models/bge-reranker-v2-m3-mlx-mxfp8",
        onnx_threads=6,
        rag_embedding_batch_size=3,
        rag_rerank_batch_size=7,
    )
    cfg.save(str(p))
    loaded = AgentConfig.load(str(p))
    assert loaded.llm_backend == "mlx_lm"
    assert loaded.rag_embedding_backend == "mlx"
    assert loaded.rag_rerank_backend == "mlx"
    assert loaded.rag_rerank_mlx_model_path.endswith("bge-reranker-v2-m3-mlx-mxfp8")
    assert loaded.onnx_threads == 6
    assert loaded.rag_embedding_batch_size == 3
    assert loaded.rag_rerank_batch_size == 7


def test_embedding_batch_size_default_and_clamp(tmp_path):
    """默认 16（MLX 拐点），越界值夹回 [2, 64]。"""
    assert AgentConfig().rag_embedding_batch_size == 16
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"rag_embedding_batch_size": 1000}))
    assert AgentConfig.load(str(p)).rag_embedding_batch_size == 64
    p.write_text(json.dumps({"rag_embedding_batch_size": 0}))
    assert AgentConfig.load(str(p)).rag_embedding_batch_size == 2


def test_invalid_local_backend_values_fall_back(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "llm_backend": "bad",
        "rag_embedding_backend": "bad",
        "rag_rerank_backend": "bad",
        "onnx_threads": 0,
    }))
    cfg = AgentConfig.load(str(p))
    # 非法后端 + 无 llm_mode：按本地模式推断，后端回退到新的默认值 mlx_lm
    # （旧行为是回退 openai，但默认模式已经是本地，回退到 openai 反而自相矛盾）。
    assert cfg.llm_mode == "local"
    assert cfg.llm_backend == "mlx_lm"
    assert cfg.rag_embedding_backend == "fastembed"
    assert cfg.rag_rerank_backend == "fastembed"
    assert cfg.onnx_threads == 1
    # 辅助后端非法 → mlx_lm（只支持本地直载，没有 openai 这一档）。
    assert cfg.aux_llm_backend == "mlx_lm"
