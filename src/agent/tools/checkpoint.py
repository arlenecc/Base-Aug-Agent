"""两个「沉淀态」原子工具（对齐 GenericAgent 的同名原子工具）。

- ``update_working_checkpoint``  —— 短期：本任务后续必须记住的非显然事实。
- ``start_long_term_update``     —— 长期：任务完成/检查点后触发经验蒸馏。

二者都不做重活：只写 Session 的工作记忆并返回极简回执。真正的 L4 蒸馏由内核
在 ``CONSOLIDATE`` / 终止边通过 Agent 的 hook 完成（异步、不阻塞回复）。
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from ..config import AgentConfig
from ..kernel.compressor import compact_json
from .base import Tool, ToolRegistry, ToolResult
from .memory import _work_memory

# 跨 run 持久化的 key（Session 是进程内的，工作记忆还要能跨会话存活）
_CHECKPOINT_KEY = "__checkpoint__"
_MAX_KEY_INFO = 4_000


def _session(registry: ToolRegistry) -> Optional[Any]:
    """内核把当前 Session 挂在 registry 上，工具通过它读写工作记忆。"""
    return getattr(registry, "session", None)


class UpdateWorkingCheckpointTool(Tool):
    """Short-term working notepad — auto-injected into every subsequent turn.

    对应 GenericAgent 的 ``update_working_checkpoint``：只记「非显然、后续必须
    知道」的状态（已验证的路径、关键 ID、阶段性结论、已排除的方案），不记流水账。
    """

    name = "update_working_checkpoint"
    description = (
        "Short-term working notepad, auto-injected into every later turn. "
        "Call early/mid task ONLY when there is non-obvious state worth keeping: "
        "verified paths/IDs/ports, decisions, approaches already ruled out. "
        "Do NOT store transcripts or facts re-derivable in one tool call. "
        "Empty key_info clears it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key_info": {
                "type": "string",
                "description": f"Compact notes (<={_MAX_KEY_INFO} chars). Empty string clears the notepad.",
            },
            "persist": {
                "type": "boolean",
                "description": "Also persist across sessions (default false).",
            },
        },
        "required": ["key_info"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, key_info: str, persist: bool = False) -> ToolResult:
        text = (key_info or "").strip()[:_MAX_KEY_INFO]
        sess = _session(self.registry)
        if sess is None:
            # 旧实现这里静默返回成功，模型以为 checkpoint 已写入，实际什么都没存
            # （无内核 session 时工具收不到任何回注）——失败必须可见。
            return ToolResult(
                False,
                error="working checkpoint unavailable (no kernel session bound)",
            )
        sess.set_key_info(text)
        if persist:
            try:
                _work_memory(self.config).set(_CHECKPOINT_KEY, text)
            except Exception as e:
                return ToolResult(False, error=f"persist failed: {e}")
        if not text:
            return ToolResult(True, output="working checkpoint cleared")
        hint = ""
        if sess.current_turn <= 2:
            hint = " (tip: skip re-calling unless new non-obvious facts appear)"
        return ToolResult(True, output=f"working checkpoint updated ({len(text)} chars){hint}")


class StartLongTermUpdateTool(Tool):
    """Trigger L4 distillation of this task's experience into long-term memory.

    对应 GenericAgent 的 ``start_long_term_update``。它不直接改记忆文件 ——
    只把「本会话摘要 + 待提纯材料」交给模型，并给内核打上 distill 标记；
    内核在终止边调用 Agent 的 hook 真正写入知识图谱 / 技能库。
    """

    name = "start_long_term_update"
    description = (
        "After finishing a task or reaching a checkpoint, distill durable experience "
        "into long-term memory. Only for verified facts, user preferences and lessons "
        "that are expensive to rediscover. Then (optionally) call memory_graph to add "
        "the distilled entities, or skill_save to freeze a reusable procedure. "
        "Skip if nothing durable was learned."
    )
    parameters = {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "What to distill, e.g. 'python env setup' (optional).",
            },
        },
        "required": [],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, focus: str = "") -> ToolResult:
        sess = _session(self.registry)
        transcript = ""
        if sess is not None:
            transcript = "\n".join(sess.all_digest_lines()[-25:])
            ki = sess.get_key_info()
            if ki:
                transcript = f"[checkpoint]\n{ki}\n\n{transcript}"
        if not transcript:
            transcript = "(当前会话尚无有效记录)"

        payload = {
            "status": "DISTILL_REQUESTED",
            "focus": focus or "",
            "transcript": transcript[-3000:],
            "next": [
                "memory_graph(op=create_entity/add_observations) 写入验证过的事实与偏好",
                "skill_save(...) 若该流程可复用，固化为 SKILL.md",
                "rag_ingest() 若有外部语料需要灌入本地知识库",
            ],
            "rules": [
                "只写经工具验证成功的信息；禁止写入推测、临时状态、通用常识",
                "合并重复、压缩冗述，不堆叠流水账",
                "无提纯价值则跳过，不要为了调用而调用",
            ],
        }
        return ToolResult(True, output=compact_json(payload, 2600))


# ---------------------------------------------------------------------------
# 供 Agent 的 L4 hook 复用：把 checkpoint 从磁盘恢复到 Session
# ---------------------------------------------------------------------------

def restore_checkpoint(config: AgentConfig, session: Any) -> str:
    """把上次持久化的 checkpoint 恢复到 Session（跨会话连续性）。"""
    if session is None:
        return ""
    try:
        text = str(_work_memory(config).get(_CHECKPOINT_KEY) or "")
    except Exception:
        return ""
    if text and not session.get_key_info():
        session.set_key_info(text)
    return text
