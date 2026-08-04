"""work_memory and memory_extract tools: backed by the memory module."""
from __future__ import annotations

import json
import os
from typing import Any

from ..config import AgentConfig
from ..memory import LongTermMemory, WorkMemory
from .base import Tool, ToolRegistry, ToolResult


def _work_memory(config: AgentConfig) -> WorkMemory:
    return WorkMemory(path=os.path.join(config.workspace, ".agent", "work_memory.json"))


def _long_memory(config: AgentConfig) -> LongTermMemory:
    return LongTermMemory(path=os.path.join(config.workspace, ".agent", "long_memory.json"))


class WorkMemoryTool(Tool):
    name = "work_memory"
    description = (
        "A persistent scratchpad across the task. op: 'set' (key,value), 'get' (key), "
        "'list', 'clear'. Use it to remember intermediate results and plans."
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["set", "get", "list", "clear"]},
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["op"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, op: str, key: str = "", value: str = "") -> ToolResult:
        wm = _work_memory(self.config)
        if op == "set":
            if not key:
                return ToolResult(False, error="op 'set' requires 'key'")
            wm.set(key, value)
            return ToolResult(True, output=f"set {key}")
        if op == "get":
            if not key:
                return ToolResult(False, error="op 'get' requires 'key'")
            v = wm.get(key)
            return ToolResult(True, output=v if v is not None else "")
        if op == "list":
            return ToolResult(True, output=json.dumps(wm.list(), ensure_ascii=False))
        if op == "clear":
            wm.clear()
            return ToolResult(True, output="cleared")
        return ToolResult(False, error=f"unknown op: {op}")


class MemoryExtractTool(Tool):
    name = "memory_extract"
    description = (
        "Persist facts to long-term memory. Either {'facts':[...]} to add, or "
        "{'op':'search','query':'...'} to recall, or {'op':'all'} to list everything."
    )
    parameters = {
        "type": "object",
        "properties": {
            "facts": {"type": "array", "items": {"type": "string"}},
            "op": {"type": "string", "enum": ["search", "all", "clear"]},
            "query": {"type": "string"},
        },
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, facts: list = None, op: str = "", query: str = "") -> ToolResult:
        lm = _long_memory(self.config)
        if facts:
            lm.add_many([str(f) for f in facts])
            return ToolResult(True, output=f"stored {len(facts)} fact(s)")
        if op == "search":
            return ToolResult(True, output=json.dumps(lm.search(query), ensure_ascii=False))
        if op == "all":
            return ToolResult(True, output=json.dumps(lm.all(), ensure_ascii=False))
        if op == "clear":
            lm.clear()
            return ToolResult(True, output="cleared")
        return ToolResult(False, error="Provide 'facts' to add, or 'op' in {search,all,clear}.")
