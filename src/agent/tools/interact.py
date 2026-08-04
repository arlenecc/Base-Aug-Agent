"""ask_user tool: pause and ask the human a question via callbacks."""
from __future__ import annotations

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "Ask the user a clarifying question or request confirmation/info. "
        "Use when information is missing or a non-destructive decision is needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Question to ask the user."},
        },
        "required": ["prompt"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, prompt: str) -> ToolResult:
        answer = self.registry.callbacks.ask_user(prompt)
        if answer is None:
            return ToolResult(False, error="User dismissed the question.")
        return ToolResult(True, output=str(answer))
