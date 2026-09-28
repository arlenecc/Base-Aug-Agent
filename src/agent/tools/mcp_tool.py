"""MCP tool adapter – wraps an MCP server tool as a base-agent Tool."""
from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

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
        # run(**kwargs) 吞掉所有参数，base.execute() 的签名预检对它无效：
        # 缺 required 参数不会被拦下，而是原样发给远端。server 若不自己校验，
        # 就会静默产生错误结果（模型拿到的是「成功」）。这里按 schema 先本地
        # 校验，错误信息与 base.execute() 的口径一致，便于模型修正调用。
        missing = [k for k in (self.parameters.get("required") or []) if k not in kwargs]
        if missing:
            return ToolResult(
                success=False,
                error=f"Invalid arguments for {self.name}: "
                      f"missing required argument(s): {', '.join(missing)}",
            )
        try:
            output = self._client.call_tool(self.name, kwargs)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))