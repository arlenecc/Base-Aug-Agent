"""Tool base class, registry, and exports."""
from __future__ import annotations

import json
import logging
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
        from .memory import WorkMemoryTool, MemoryExtractTool

        for t in [
            FileReadTool(), FileWriteTool(), FileModifyTool(),
            CodeRunTool(),
            ShellRunTool(),
            WebScanTool(), WebExecJsTool(),
            AskUserTool(),
            WorkMemoryTool(),
            MemoryExtractTool(),
        ]:
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
        if not kb:
            return

        try:
            from ..rag.engine import RAGEngine
            from .rag_tool import RagSearchTool, RagStatusTool, RagIngestTool
        except ImportError as e:
            logger.warning("RAG module not available: %s", e)
            return

        engine = RAGEngine(
            workspace=self.config.workspace,
            knowledge_base=kb,
            embedding_model=getattr(self.config, "rag_embedding_model", "nomic-ai/nomic-embed-text-v1.5"),
        )
        self._rag_engine = engine

        for t in [RagSearchTool(engine), RagStatusTool(engine), RagIngestTool(engine)]:
            if t.name not in self._tools:
                t.bind(self.config, self)
                self._tools[t.name] = t

        # Auto-ingest if configured
        if getattr(self.config, "rag_auto_ingest", True):
            try:
                stats = engine.ingest(force=False)
                logger.info(
                    "RAG auto-ingest: %d files processed, %d chunks",
                    stats.get("files_extracted", 0) + stats.get("files_skipped", 0),
                    stats.get("chunks", 0),
                )
            except Exception as e:
                logger.warning("RAG auto-ingest error: %s", e)

    def get_rag_engine(self):
        """Return the RAG engine instance, or None if not configured."""
        return self._rag_engine

    def shutdown(self) -> None:
        """Stop all MCP server subprocesses."""
        for client in self._mcp_clients:
            try:
                client.stop()
            except Exception:
                pass
        self._mcp_clients.clear()

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
