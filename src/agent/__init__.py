"""base-agent: a streamlined local agent with a PyQt6 UI."""
from __future__ import annotations

from .agent import Agent, AgentCallbacks, SYSTEM_PROMPT
from .config import AgentConfig
from .llm_client import LLMClient, StreamEvent, LLMError
from .memory import LongTermMemory, WorkMemory
from .skills import Skill, SkillManager
from .tools import Tool, ToolRegistry, ToolResult

__all__ = [
    "Agent", "AgentCallbacks", "SYSTEM_PROMPT",
    "AgentConfig",
    "LLMClient", "StreamEvent", "LLMError",
    "LongTermMemory", "WorkMemory",
    "Skill", "SkillManager",
    "Tool", "ToolRegistry", "ToolResult",
]
