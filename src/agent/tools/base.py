"""Tool base class, registry, and exports."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    # When requires_input is set, the agent should call callbacks.ask_user and
    # feed the answer back (used by ask_user tool internally).
    requires_input: bool = False

    def to_message(self) -> str:
        if self.success:
            return self.output
        return f"ERROR: {self.error}" if self.error else "ERROR: unknown"


class Tool:
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    # destructive marks tools that *can* affect the system or files outside the
    # workspace. Workspace-locked file ops set this False (they cannot escape).
    destructive: bool = False

    def run(self, **kwargs) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        """Return True if this invocation needs user confirmation before running.

        Default policy: confirm iff `self.destructive` is set. Override for
        finer-grained control (e.g. confirm only when args target paths outside
        the workspace). Operations confined to the workspace never need
        confirmation, since they cannot harm the system or other files.
        """
        return self.destructive

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, config: AgentConfig, callbacks: "Any"):
        self.config = config
        self.callbacks = callbacks
        self.config.ensure_workspace()
        self._tools: Dict[str, Tool] = {}
        self._mcp_clients: list = []  # MCPClient instances, for cleanup
        self._rag_engine = None  # lazy-initialized RAGEngine
        self._register_defaults()

    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        from .file_ops import FileReadTool, FileWriteTool, FileModifyTool
        from .code_run import CodeRunTool
        from .shell_run import ShellRunTool
        from .web import WebScanTool, WebExecJsTool
        from .interact import AskUserTool
        from .memory import WorkMemoryTool, MemoryGraphTool, MemorySearchTool
        from .skill_search import SkillSearchTool, SkillLoadTool

        tools_to_register = [
            FileReadTool(), FileWriteTool(), FileModifyTool(),
            CodeRunTool(),
            ShellRunTool(),
            WebScanTool(),
            AskUserTool(),
            WorkMemoryTool(),
            MemoryGraphTool(),
            MemorySearchTool(),
            SkillSearchTool(),
            SkillLoadTool(),
        ]

        # webexec_js: only register when browser_endpoint is configured.
        # Saves ~146 tokens of schema when no browser bridge is available
        # (the common case).  Without an endpoint the tool just returns
        # guidance text — no functional value.
        if getattr(self.config, "browser_endpoint", ""):
            tools_to_register.append(WebExecJsTool())

        for t in tools_to_register:
            t.bind(self.config, self)
            self._tools[t.name] = t

        # Register RAG tools
        self._register_rag_tools()

        # Register MCP tools from configured servers
        self._register_mcp_tools()

    def _register_mcp_tools(self) -> None:
        """Connect to configured MCP servers and register their tools."""
        mcp_servers = getattr(self.config, "mcp_servers", None) or []
        if not mcp_servers:
            return

        from .mcp_client import MCPClient, MCPError
        from .mcp_tool import MCPTool

        for server_cfg in mcp_servers:
            if not isinstance(server_cfg, dict):
                continue
            name = server_cfg.get("name", "")
            command = server_cfg.get("command", "")
            if not name or not command:
                logger.warning("MCP server config missing name or command: %s", server_cfg)
                continue

            try:
                client = MCPClient(
                    name=name,
                    command=command,
                    args=server_cfg.get("args", []),
                    env=server_cfg.get("env"),
                )
                client.start()
                self._mcp_clients.append(client)

                for tool_def in client.list_tools():
                    t = MCPTool(client, tool_def)
                    if t.name and t.name not in self._tools:
                        t.bind(self.config, self)
                        self._tools[t.name] = t
                        logger.info("MCP tool registered: %s (server: %s)", t.name, name)
            except MCPError as e:
                logger.warning("MCP server '%s' init failed: %s", name, e)
            except Exception as e:
                logger.warning("MCP server '%s' unexpected error: %s", name, e)

    def _register_rag_tools(self) -> None:
        """Initialize RAG engine and register RAG tools."""
        kb = getattr(self.config, "knowledge_base", "") or ""
        # If knowledge_base is not explicitly set, default to
        # <workspace>/knowledge_base so that RAG tools are available out of
        # the box when users place documents there (or have previously
        # synced knowledge).  Without this, the agent never registers RAG
        # tools and cannot discover the local knowledge base.
        if not kb:
            kb = os.path.join(self.config.workspace, "knowledge_base")
            logger.info(
                "RAG: knowledge_base not configured, defaulting to %s", kb
            )

        # Create the directory if it doesn't exist so that RAGEngine
        # initialization doesn't fail on a missing path.
        os.makedirs(kb, exist_ok=True)

        try:
            from ..rag.engine import RAGEngine
            from .rag_tool import RagSearchTool, RagStatusTool, RagIngestTool
        except ImportError as e:
            logger.warning("RAG module not available: %s", e)
            return

        logger.info(
            "RAG: initializing engine (workspace=%s kb=%s embedding=%s)",
            self.config.workspace, kb,
            getattr(self.config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q"),
        )
        engine = RAGEngine(
            workspace=self.config.workspace,
            knowledge_base=kb,
            embedding_model=getattr(self.config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5-Q"),
        )
        self._rag_engine = engine

        registered = []
        for t in [RagSearchTool(engine), RagStatusTool(engine), RagIngestTool(engine)]:
            if t.name not in self._tools:
                t.bind(self.config, self)
                self._tools[t.name] = t
                registered.append(t.name)
        logger.info("RAG: tools registered: %s", ", ".join(registered))

        # NOTE: auto-ingest is intentionally NOT performed here.
        # _register_rag_tools() runs in ToolRegistry.__init__, which is called
        # from the UI main thread (via _rebuild_agent). Synchronous ingest would
        # block the UI for minutes on first run (model download + parsing +
        # embedding of all files). The UI provides SyncKnowledgeWorker (async,
        # on a background thread) for this purpose. For non-UI usage, call
        # engine.ingest() explicitly after construction.

    def get_rag_engine(self):
        """Return the RAG engine instance, or None if not configured."""
        return self._rag_engine

    def reload_rag(self) -> None:
        """Refresh the RAG engine's table handle so new sync data is visible.

        Call this after SyncKnowledgeWorker finishes — the agent's RAGEngine
        holds a stale LanceDB table snapshot that won't reflect newly written
        rows until the table is re-opened.
        """
        if self._rag_engine is not None:
            try:
                self._rag_engine.reload()
            except Exception as e:
                logger.warning("ToolRegistry.reload_rag error: %s", e)

    def shutdown(self) -> None:
        """Stop all MCP server subprocesses and release RAG engine resources."""
        for client in self._mcp_clients:
            try:
                client.stop()
            except Exception:
                pass
        self._mcp_clients.clear()
        # Close shared httpx clients in web tools to release connection pools.
        for tool in self._tools.values():
            http_client = getattr(tool, "_http", None)
            if http_client is not None:
                try:
                    http_client.close()
                except Exception:
                    pass
        # Release the RAG engine's VectorStore (LanceDB connection + FastEmbed
        # ONNX model + BGE reranker). Without this, each _rebuild_agent() call
        # leaks ~600MB of C++ heap memory.
        if self._rag_engine is not None:
            try:
                logger.info("🔌 正在释放 RAG 引擎资源 (向量库连接 + 嵌入模型 + Reranker)...")
                self._rag_engine.close()
                logger.info("✅ RAG 引擎资源已释放")
            except Exception as e:
                logger.warning("⚠ RAG 引擎资源释放异常: %s", e)
            self._rag_engine = None
        # Release the singleton LongTermMemory's GraphMemoryStore (LanceDB +
        # FastEmbed ONNX model ~137MB).  Without this, each _rebuild_agent()
        # call leaks the model loaded for the knowledge graph.
        from .memory import _LONG_MEMORY_SINGLETON_ATTR
        lm = getattr(self, _LONG_MEMORY_SINGLETON_ATTR, None)
        if lm is not None:
            try:
                lm.close()
            except Exception as e:
                logger.warning("⚠ 长期记忆资源释放异常: %s", e)
            setattr(self, _LONG_MEMORY_SINGLETON_ATTR, None)
        # Release the SkillIndex singleton (if any)
        skill_idx = getattr(self, "_skill_index", None)
        if skill_idx is not None:
            setattr(self, "_skill_index", None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    # ------------------------------------------------------------------
    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        try:
            return tool.run(**args)
        except TypeError as e:
            return ToolResult(success=False, error=f"Invalid arguments for {name}: {e}")
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}")


def parse_args(raw: str) -> Dict[str, Any]:
    """Parse a tool-call arguments JSON string.

    Returns ``{}`` for empty input (a model may legitimately call a no-arg
    tool with ``""``). For non-empty but malformed JSON, raises ``ValueError``
    with a clear message so the agent can report it back to the model —
    silently returning ``{}`` here would turn a JSON typo into a confusing
    "missing required argument" error one layer up, which the model cannot
    easily diagnose.
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool arguments ({e.msg} at pos {e.pos}): {raw[:200]!r}") from e
