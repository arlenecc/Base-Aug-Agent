"""Configuration for the agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Runtime configuration. Persisted to ~/.base-agent/config.json by the UI."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 32768
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 20
    repetition_penalty: float = 1.0
    workspace: str = field(default_factory=lambda: os.path.expanduser("~/base-agent-workspace"))
    # Default to <workspace>/knowledge_base so RAG tools are available out of
    # the box.  Users can override via the UI or config.json.
    knowledge_base: str = ""
    request_timeout: float = 120.0
    max_iterations: int = 15
    max_history: int = 50  # max messages retained in conversation history
    # Context window budget (tokens). When the estimated prompt size reaches
    # 90% of this value, the agent proactively summarizes older messages and
    # persists the summary to memory.md. Also used to recover from
    # context_length_exceeded errors returned by the LLM API.
    # Default matches max_tokens (32768) so the shrink threshold tracks the
    # model's actual context window.
    max_context_tokens: int = 32768
    # Shrink ratio: trigger proactive shrink when estimated context size
    # exceeds this fraction of max_context_tokens.
    context_shrink_ratio: float = 0.9

    # RAG settings
    rag_chunk_size: int = 500       # tokens per chunk
    rag_chunk_overlap: int = 50     # 10% of chunk_size
    rag_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    rag_rerank_model: str = "BAAI/bge-reranker-base"  # bge-reranker for precise scoring
    rag_rerank_enabled: bool = True
    rag_auto_ingest: bool = True

    # Browser endpoint for webexec_js (Playwright-like bridge). Optional.
    browser_endpoint: str = ""

    # MCP servers – list of {name, command, args?, env?} dicts.
    # Example:
    #   [{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}]
    mcp_servers: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        import json

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            known = {k for k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
            cfg = cls(**{k: v for k, v in data.items() if k in known})
            cfg._migrate()
            return cfg
        return cls()

    def _migrate(self) -> None:
        """Normalize obsolete config values from older versions.

        Runs after loading from disk so that stale values saved before a
        migration don't break the current code. Silently rewrites the
        in-memory config; the next ``save()`` persists the fix.
        """
        # Pre-LanceDB era used sentence-transformers model names or
        # ChromaDB defaults that FastEmbed doesn't recognize. Map any
        # unsupported embedding model to the current default.
        _DEPRECATED_EMBEDDING_MODELS = {
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-zh-v1.5",
            "BAAI/bge-large-zh-v1.5",
            "",
        }
        if self.rag_embedding_model in _DEPRECATED_EMBEDDING_MODELS:
            self.rag_embedding_model = "nomic-ai/nomic-embed-text-v1.5"
        # Older configs saved max_tokens=4096 (the old LLMClient default).
        # Bump any sub-8192 value to the current default (32768) so users
        # upgrading don't keep hitting the tiny old budget.
        if self.max_tokens < 8192:
            self.max_tokens = 32768
        # max_context_tokens was introduced at 32000; older configs that
        # saved a stale small value should also be bumped. Track max_tokens
        # so the shrink threshold follows the model's actual window.
        if self.max_context_tokens < 8192:
            self.max_context_tokens = self.max_tokens

    def save(self, path: str) -> None:
        import json

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

    def ensure_workspace(self) -> str:
        os.makedirs(self.workspace, exist_ok=True)
        return self.workspace
