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
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5"
    assert cfg.base_url == "https://api.openai.com/v1"


def test_load_preserves_known_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-abc",
        "model": "deepseek-chat",
        "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5",
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
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5"


def test_migrate_preserves_valid_embedding_model(tmp_path):
    """A currently-supported embedding model name must not be touched."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "base_url": "http://x",
        "api_key": "",
        "model": "m",
        "rag_embedding_model": "nomic-ai/nomic-embed-text-v1.5",
    }))
    cfg = AgentConfig.load(str(p))
    assert cfg.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5"


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
    assert cfg2.rag_embedding_model == "nomic-ai/nomic-embed-text-v1.5"


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


def test_ensure_workspace_creates_dir(tmp_path):
    cfg = AgentConfig(workspace=str(tmp_path / "ws"))
    cfg.ensure_workspace()
    assert os.path.isdir(str(tmp_path / "ws"))
