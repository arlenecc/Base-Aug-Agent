"""Skill search tool: lets the agent discover and load skill prompts.

The agent can call skill_search to find skills relevant to the current
task, then skill_load to read the full prompt and inject it into the
conversation.

Skills live in .agent/skills/, each in its own subdirectory with
a skill.json metadata file and a prompt.md entry file.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ..config import AgentConfig
from ..skill_index import SkillIndex
from .base import Tool, ToolRegistry, ToolResult


def _get_skill_index(config: AgentConfig, registry: ToolRegistry) -> SkillIndex:
    """Get or create the singleton SkillIndex, cached on the registry."""
    idx = getattr(registry, "_skill_index", None)
    if idx is not None:
        return idx
    skills_dir = os.path.join(config.workspace, ".agent", "skills")
    idx = SkillIndex(skills_dir)
    setattr(registry, "_skill_index", idx)
    return idx


class SkillSearchTool(Tool):
    """Search for skills relevant to the current task.

    The agent should call this when it encounters a task that might
    benefit from a pre-defined skill (e.g. "extract PDF tables", "scrape
    a website").  Returns matching skills with name, description, and
    path.
    """
    name = "skill_search"
    description = (
        "Search for available skills relevant to a task. "
        "Returns matching skills with name, description, and path. "
        "Use before attempting complex tasks — a skill may provide "
        "specialized instructions. op: 'search' (query), 'list' (all)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["search", "list"], "default": "search"},
            "query": {"type": "string", "description": "Task description to find matching skills"},
            "top_k": {"type": "integer", "default": 5},
        },
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, op: str = "search", query: str = "", top_k: int = 5) -> ToolResult:
        idx = _get_skill_index(self.config, self.registry)
        if op == "list":
            results = idx.list_all()
            # Compact output: name + path + description (truncated)
            compact = [
                {
                    "name": s["name"],
                    "path": s["path"],
                    "description": s.get("description", "")[:100],
                }
                for s in results
            ]
            return ToolResult(True, output=json.dumps(compact, ensure_ascii=False))
        # Default: search
        if not query.strip():
            return ToolResult(False, error="op 'search' requires 'query'")
        results = idx.search(query, top_k=top_k)
        return ToolResult(True, output=json.dumps(results, ensure_ascii=False))


class SkillLoadTool(Tool):
    """Load a skill's full prompt content by its path.

    After skill_search finds a relevant skill, use skill_load to read
    its prompt/instructions file and inject them into the conversation.
    """
    name = "skill_load"
    description = (
        "Load a skill's full prompt content. Provide 'path' (from skill_search result). "
        "Returns the skill's entry file content (e.g. prompt.md)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Skill directory name (from skill_search result 'path' field)"},
            "entry": {"type": "string", "description": "Entry file name (default: prompt.md)"},
        },
        "required": ["path"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, path: str, entry: str = "prompt.md") -> ToolResult:
        idx = _get_skill_index(self.config, self.registry)
        content = idx.read_skill_content(path, entry)
        if content is None:
            return ToolResult(False, error=f"skill not found or entry file missing: {path}/{entry}")
        return ToolResult(True, output=content)
