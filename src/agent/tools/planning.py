"""``task_plan`` tool: create / update / inspect the agent's task plan.

The plan gives the model a persistent anchor for multi-step tasks: goal at
the top, concrete steps with status, injected into the system prompt every
iteration (see agent._system_prompt). State lives in planner.py.
"""
from __future__ import annotations

from typing import List, Optional

from ..config import AgentConfig
from ..planner import (
    PlanStep,
    STATUSES,
    TaskPlan,
    clear_plan,
    load_plan,
    render_plan,
    save_plan,
)
from .base import Tool, ToolRegistry, ToolResult


class TaskPlanTool(Tool):
    name = "task_plan"
    description = (
        "Maintain the current task plan (goal + ordered steps with status). "
        "op: create(goal, steps) for a new task; update(index, status, note) "
        "to mark progress; add(step) for newly discovered steps; get to view; "
        "clear when the task is done."
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["create", "update", "add", "get", "clear"]},
            "goal": {"type": "string", "description": "create: the overall task goal"},
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "create: concrete steps, in execution order",
            },
            "index": {"type": "integer", "description": "update: 1-based step index"},
            "status": {
                "type": "string",
                "enum": list(STATUSES),
                "description": "update: new status for the step",
            },
            "note": {"type": "string", "description": "update: key result/finding for the step"},
            "step": {"type": "string", "description": "add: a newly discovered step"},
        },
        "required": ["op"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, op: str, goal: str = "", steps: Optional[List[str]] = None,
            index: int = 0, status: str = "", note: str = "", step: str = "") -> ToolResult:
        workspace = self.config.workspace

        if op == "create":
            if not goal.strip():
                return ToolResult(False, error="op 'create' requires 'goal'")
            texts = [s for s in (steps or []) if isinstance(s, str) and s.strip()]
            if not texts:
                return ToolResult(False, error="op 'create' requires a non-empty 'steps' list")
            plan = TaskPlan(goal=goal, steps=[PlanStep(text=t) for t in texts])
            save_plan(workspace, plan)
            return ToolResult(True, output="Plan created:\n" + render_plan(plan))

        if op == "update":
            plan = load_plan(workspace)
            if plan is None:
                return ToolResult(False, error="No active plan. Use op 'create' first.")
            if index < 1 or index > len(plan.steps):
                return ToolResult(
                    False,
                    error=f"index out of range: {index} (plan has {len(plan.steps)} steps)",
                )
            st = plan.steps[index - 1]
            if status:
                if status not in STATUSES:
                    return ToolResult(
                        False, error=f"invalid status '{status}'; one of {', '.join(STATUSES)}"
                    )
                st.status = status
            if note:
                st.note = note
            save_plan(workspace, plan)
            return ToolResult(True, output="Plan updated:\n" + render_plan(plan))

        if op == "add":
            if not step.strip():
                return ToolResult(False, error="op 'add' requires 'step'")
            plan = load_plan(workspace)
            if plan is None:
                return ToolResult(False, error="No active plan. Use op 'create' first.")
            plan.steps.append(PlanStep(text=step))
            save_plan(workspace, plan)
            return ToolResult(True, output="Step added:\n" + render_plan(plan))

        if op == "get":
            plan = load_plan(workspace)
            if plan is None:
                return ToolResult(True, output="(no active plan)")
            return ToolResult(True, output=render_plan(plan))

        if op == "clear":
            clear_plan(workspace)
            return ToolResult(True, output="Plan cleared.")

        return ToolResult(False, error=f"unknown op: {op}")
