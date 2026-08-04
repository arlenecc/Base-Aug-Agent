"""MCP tool adapter – wraps an MCP server tool as a base-agent Tool."""
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import Tool, ToolResult, ToolRegistry

if TYPE_CHECKING:
    from .mcp_client import MCPClient


class MCPTool(Tool):
    """A Tool that delegates execution to an MCP server."""

    def __init__(self, mcp_client: "MCPClient", tool_def: Dict[str, Any]):
        self._client = mcp_client
        self._tool_def = tool_def

        self.name = tool_def.get("name", "")
        self.description = tool_def.get("description", "")
        # MCP uses JSON Schema for inputSchema; convert to OpenAI function-call
        # parameters format (they are compatible).
        input_schema = tool_def.get("inputSchema", {})
        self.parameters = {
            "type": input_schema.get("type", "object"),
            "properties": input_schema.get("properties", {}),
            "required": input_schema.get("required", []),
        }
        # MCP tools are external — always mark as destructive so they require
        # confirmation unless overridden in the tool definition.
        self.destructive = True

    def bind(self, config: Any, registry: ToolRegistry) -> None:
        pass

    def run(self, **kwargs) -> ToolResult:
        try:
            output = self._client.call_tool(self.name, kwargs)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def mcp_schema_to_tool_schema(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an MCP tool definition to OpenAI function-calling schema."""
    input_schema = tool_def.get("inputSchema", {})
    return {
        "type": "function",
        "function": {
            "name": tool_def.get("name", ""),
            "description": tool_def.get("description", ""),
            "parameters": {
                "type": input_schema.get("type", "object"),
                "properties": input_schema.get("properties", {}),
                "required": input_schema.get("required", []),
            },
        },
    }