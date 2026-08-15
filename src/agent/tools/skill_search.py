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


class SkillSemanticSearchTool(Tool):
    """Hybrid (vector + BM25) skill search over SKILL.md files.

    Scans ``<workspace>/skills`` recursively for ``SKILL.md`` files, indexes
    their descriptions + examples (vector) and combined text (BM25), then
    retrieves candidate skills for a task description.  Optionally hands the
    candidates to the LLM for a final pick.

    This is the semantic counterpart to ``skill_search`` (keyword-based).
    """
    name = "skill_semantic_search"
    description = (
        "根据任务描述语义检索最合适的 skill（向量 + BM25 混合检索 SKILL.md）。"
        "当需要为当前任务寻找可复用的 skill/工作流时使用。"
        "返回候选 skill 的 name + tags + description；配合 skill_load 读取完整指令。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "任务描述，用于检索匹配的 skill"},
            "top_k": {"type": "integer", "description": "返回候选数，默认 5"},
        },
        "required": ["query"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, query: str, top_k: int = 5) -> ToolResult:
        if not query.strip():
            return ToolResult(False, error="requires 'query'")
        retriever = self.registry.get_skill_retriever()
        # 只做混合检索 + 阈值过滤；最终的「选哪个」由 agent（大模型）根据
        # 返回的 name+tags+description 自行判断，避免工具层再调一次 LLM。
        result = retriever.retrieve(query, llm=None)

        candidates = result.get("candidates", [])
        if not candidates:
            return ToolResult(True, output="找不到匹配的技能。")

        # Compact output: dir + name + tags + description (truncated).
        items = []
        for c in candidates[:top_k]:
            item = {
                "dir": c["dir"],
                "name": c["name"],
                "tags": c.get("tags", ""),
                "description": (c.get("description", "") or "")[:200],
                "score": round(c["score"], 4),
            }
            items.append(item)
        return ToolResult(True, output=json.dumps(items, ensure_ascii=False, indent=2))


class SkillLoadTool(Tool):
    """Load a skill's full prompt content by its path.

    After skill_search finds a relevant skill, use skill_load to read
    its prompt/instructions file and inject them into the conversation.
    """
    name = "skill_load"
    description = (
        "读取 skill 的完整指令内容。提供 'path'（来自 skill_search / "
        "skill_semantic_search 结果的 'dir' 字段），entry 默认 SKILL.md。"
        "也可读取 skill 目录下的其它支持文件（如 references/xxx.md）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Skill 目录（检索结果的 'dir' 字段）"},
            "entry": {"type": "string", "description": "文件名，默认 SKILL.md（旧体系默认 prompt.md）"},
        },
        "required": ["path"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, path: str, entry: str = "SKILL.md") -> ToolResult:
        # 1) Try the semantic retriever first (SKILL.md directory).
        retriever = getattr(self.registry, "_skill_retriever", None)
        if retriever is not None:
            abs_dir = retriever.read_skill_dir(path)
            if abs_dir:
                target = os.path.join(abs_dir, entry)
                if os.path.isfile(target):
                    try:
                        with open(target, "r", encoding="utf-8") as f:
                            return ToolResult(True, output=f.read())
                    except (OSError, UnicodeDecodeError) as e:
                        return ToolResult(False, error=f"read failed: {e}")

        # 2) Fall back to the legacy SkillIndex (skill.json + prompt.md).
        idx = _get_skill_index(self.config, self.registry)
        content = idx.read_skill_content(path, entry)
        if content is None:
            return ToolResult(False, error=f"skill not found or entry file missing: {path}/{entry}")
        return ToolResult(True, output=content)
