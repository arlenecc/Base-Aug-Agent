"""base-agent 内核：三态自动机 + 五级信息压缩。

内核只控制流；历史交给 ``Session``，能力交给 ``dispatcher``（code_run 为核心），
经验交给 ``KernelHooks``（memory / skills / rag）。

    from agent.kernel import KernelLoop, Session, ContextBudget

详见 docs/KERNEL_REFACTOR.md。
"""
from __future__ import annotations

from .budget import ContextBudget, estimate_tokens
from .compressor import (
    DISTILL_PROMPT, DISTILL_SYSTEM, fit_tokens, l0_capture, l1_trim,
    l1_trim_result, l2_summary, l3_fold, l4_extract_json, l4_normalize,
    strip_summary_tags,
)
from .loop import KernelHooks, KernelLoop, KernelResult
from .outcome import (
    ExitReason, KernelState, LLMResponse, StepOutcome, ToolCall, TurnRecord,
    next_state,
)
from .session import Session, normalize_plan, render_plan_dict

__all__ = [
    "ContextBudget", "estimate_tokens",
    "l0_capture", "l1_trim", "l1_trim_result", "l2_summary", "l3_fold",
    "l4_extract_json", "l4_normalize", "fit_tokens", "strip_summary_tags",
    "DISTILL_PROMPT", "DISTILL_SYSTEM",
    "KernelHooks", "KernelLoop", "KernelResult",
    "ExitReason", "KernelState", "LLMResponse", "StepOutcome", "ToolCall",
    "TurnRecord", "next_state",
    "Session", "normalize_plan", "render_plan_dict",
]
