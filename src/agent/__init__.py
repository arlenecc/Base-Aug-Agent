"""base-agent: a streamlined local agent with a PyQt6 UI."""
from __future__ import annotations

import os

# Ensure HuggingFace models use local cache without attempting network access.
# This prevents multi-minute hangs when HuggingFace is unreachable (e.g., behind GFW).
# Models must be pre-downloaded or cached for this to work.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
