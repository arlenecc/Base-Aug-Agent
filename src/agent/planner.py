"""Task plan: a structured goal + steps the agent maintains across a task.

解决「任务半路跑丢」的问题：模型开始一个非平凡任务时先用 ``task_plan``
工具创建计划（目标 + 步骤），之后每一步根据上一步结果更新进度。计划
持久化到 ``<workspace>/.agent/task_plan.json`` 并注入系统提示，因此：

* 每轮迭代模型都能看到「目标是什么、进行到哪一步」，不会漂移；
* 计划在系统提示里，不参与上下文收缩（_shrink_context 只压 _history），
  长任务被摘要后目标依然完整；
* 换任务时模型重建计划，完成任务后清除，不留陈旧状态。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import List, Optional

PLAN_FILENAME = "task_plan.json"

STATUSES = ("pending", "in_progress", "done", "blocked")

# 渲染/存储上限：计划注入系统提示，而系统提示不在上下文收缩范围内，
# 必须给每个字段设上限，否则长步骤文本会随任务推进单调膨胀。
_MAX_STEPS = 20
_MAX_GOAL_CHARS = 300
_MAX_STEP_CHARS = 200
_MAX_NOTE_CHARS = 200

_write_lock = threading.Lock()


@dataclass
class PlanStep:
    text: str
    status: str = "pending"
    note: str = ""


@dataclass
class TaskPlan:
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    updated_at: str = ""


def _plan_path(workspace: str) -> str:
    return os.path.join(workspace, ".agent", PLAN_FILENAME)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text[:limit] + "…" if len(text) > limit else text


def _normalize(plan: TaskPlan) -> TaskPlan:
    """Enforce storage/render caps and a valid status vocabulary."""
    plan.goal = _clip(plan.goal, _MAX_GOAL_CHARS)
    steps = [s for s in plan.steps if isinstance(s, PlanStep) and s.text.strip()]
    normalized: List[PlanStep] = []
    for s in steps[:_MAX_STEPS]:
        status = s.status if s.status in STATUSES else "pending"
        normalized.append(PlanStep(
            text=_clip(s.text, _MAX_STEP_CHARS),
            status=status,
            note=_clip(s.note, _MAX_NOTE_CHARS),
        ))
    plan.steps = normalized
    plan.updated_at = datetime.now().isoformat(timespec="seconds")
    return plan


def load_plan(workspace: str) -> Optional[TaskPlan]:
    """Load the persisted plan, or None when absent/corrupt."""
    path = _plan_path(workspace)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        goal = data.get("goal", "")
        raw_steps = data.get("steps", [])
        if not isinstance(goal, str) or not goal.strip():
            return None
        if not isinstance(raw_steps, list):
            raw_steps = []
        steps = []
        for s in raw_steps:
            if not isinstance(s, dict):
                continue
            steps.append(PlanStep(
                text=str(s.get("text", "")),
                status=str(s.get("status", "pending")),
                note=str(s.get("note", "")),
            ))
        return _normalize(TaskPlan(goal=goal, steps=steps,
                                   updated_at=str(data.get("updated_at", ""))))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None


def save_plan(workspace: str, plan: TaskPlan) -> None:
    """Persist the plan atomically (temp + os.replace)."""
    _normalize(plan)
    path = _plan_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = json.dumps(asdict(plan), ensure_ascii=False, indent=2)
    with _write_lock:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)


def clear_plan(workspace: str) -> None:
    """Remove the plan file (task finished / conversation reset)."""
    with _write_lock:
        try:
            os.unlink(_plan_path(workspace))
        except FileNotFoundError:
            pass
        except OSError:
            pass


def render_plan(plan: TaskPlan) -> str:
    """Render the plan for system-prompt injection (compact, capped)."""
    lines = ["# Current task plan", f"Goal: {plan.goal}"]
    for i, s in enumerate(plan.steps, 1):
        line = f"{i}. [{s.status}] {s.text}"
        if s.note:
            line += f" — {s.note}"
        lines.append(line)
    if plan.steps and all(s.status == "done" for s in plan.steps):
        # 全部步骤完成：给模型一个明确的收尾信号，避免陈旧计划一直挂在
        # 每轮系统提示里（模型此前没有主动 clear 的提示，任务做完计划还在）。
        lines.append(
            "All steps are done. If the task is complete, call `task_plan` "
            "with op='clear' to remove the plan."
        )
    else:
        lines.append(
            "Keep this plan updated via the `task_plan` tool and work through "
            "the steps in order."
        )
    return "\n".join(lines)
