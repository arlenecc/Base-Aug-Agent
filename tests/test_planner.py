"""Tests for the task planner (planner.py + task_plan tool + agent integration)."""
from __future__ import annotations

import json

import pytest

from agent.planner import (
    PLAN_FILENAME,
    PlanStep,
    TaskPlan,
    clear_plan,
    load_plan,
    render_plan,
    save_plan,
)
from agent.tools.planning import TaskPlanTool


# ---------------------------------------------------------------------------
# planner.py primitives
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    plan = TaskPlan(goal="部署服务", steps=[
        PlanStep(text="构建镜像", status="done", note="v1.2"),
        PlanStep(text="推送镜像", status="in_progress"),
        PlanStep(text="重启容器"),
    ])
    save_plan(str(tmp_path), plan)

    loaded = load_plan(str(tmp_path))
    assert loaded is not None
    assert loaded.goal == "部署服务"
    assert [s.text for s in loaded.steps] == ["构建镜像", "推送镜像", "重启容器"]
    assert loaded.steps[0].status == "done"
    assert loaded.steps[0].note == "v1.2"


def test_load_missing_plan_returns_none(tmp_path):
    assert load_plan(str(tmp_path)) is None


def test_load_corrupt_plan_returns_none(tmp_path):
    path = tmp_path / ".agent"
    path.mkdir()
    (path / PLAN_FILENAME).write_text("{not json", encoding="utf-8")
    assert load_plan(str(tmp_path)) is None


def test_load_invalid_goal_returns_none(tmp_path):
    path = tmp_path / ".agent"
    path.mkdir()
    (path / PLAN_FILENAME).write_text(json.dumps({"goal": "  ", "steps": []}), encoding="utf-8")
    assert load_plan(str(tmp_path)) is None


def test_save_normalizes_caps_and_status(tmp_path):
    plan = TaskPlan(goal="g" * 500, steps=[
        PlanStep(text="s" * 500, status="bogus", note="n" * 500),
        PlanStep(text=""),  # 空步骤应被丢弃
    ] + [PlanStep(text=f"step {i}") for i in range(30)])
    save_plan(str(tmp_path), plan)

    loaded = load_plan(str(tmp_path))
    assert loaded is not None
    assert len(loaded.goal) <= 301  # 截断 + 省略号
    assert len(loaded.steps) <= 20
    assert all(s.status in ("pending", "in_progress", "done", "blocked") for s in loaded.steps)
    assert all(s.text.strip() for s in loaded.steps)


def test_clear_plan(tmp_path):
    save_plan(str(tmp_path), TaskPlan(goal="g", steps=[PlanStep(text="s")]))
    assert load_plan(str(tmp_path)) is not None
    clear_plan(str(tmp_path))
    assert load_plan(str(tmp_path)) is None
    # 清除不存在的计划不应报错
    clear_plan(str(tmp_path))


def test_render_plan_contains_goal_and_statuses():
    plan = TaskPlan(goal="修复 bug", steps=[
        PlanStep(text="定位问题", status="done"),
        PlanStep(text="写修复", status="in_progress", note="已改 2 处"),
    ])
    text = render_plan(plan)
    assert "修复 bug" in text
    assert "[done]" in text and "定位问题" in text
    assert "[in_progress]" in text and "已改 2 处" in text


# ---------------------------------------------------------------------------
# task_plan tool
# ---------------------------------------------------------------------------

@pytest.fixture()
def plan_tool(config):
    tool = TaskPlanTool()
    tool.bind(config, None)
    return tool


def test_tool_create_update_add_get_clear(plan_tool, config):
    r = plan_tool.run(op="create", goal="整理数据", steps=["下载数据", "清洗", "入库"])
    assert r.success and "整理数据" in r.output

    r = plan_tool.run(op="update", index=1, status="done", note="已下载 100 条")
    assert r.success and "[done]" in r.output and "已下载 100 条" in r.output

    r = plan_tool.run(op="add", step="校验结果")
    assert r.success and "校验结果" in r.output

    r = plan_tool.run(op="get")
    assert r.success and "入库" in r.output

    # 持久化到磁盘且 Agent 能读到
    loaded = load_plan(config.workspace)
    assert loaded is not None and len(loaded.steps) == 4

    r = plan_tool.run(op="clear")
    assert r.success
    assert load_plan(config.workspace) is None


def test_tool_requires_create_first(plan_tool):
    r = plan_tool.run(op="update", index=1, status="done")
    assert not r.success and "create" in r.error
    r = plan_tool.run(op="add", step="x")
    assert not r.success and "create" in r.error


def test_tool_create_validation(plan_tool):
    r = plan_tool.run(op="create", goal="", steps=["a"])
    assert not r.success
    r = plan_tool.run(op="create", goal="g", steps=["", "  "])
    assert not r.success
    r = plan_tool.run(op="create", goal="g")
    assert not r.success


def test_tool_update_validation(plan_tool):
    plan_tool.run(op="create", goal="g", steps=["only step"])
    r = plan_tool.run(op="update", index=0, status="done")
    assert not r.success and "out of range" in r.error
    r = plan_tool.run(op="update", index=5, status="done")
    assert not r.success and "out of range" in r.error
    r = plan_tool.run(op="update", index=1, status="bogus")
    assert not r.success and "invalid status" in r.error
    r = plan_tool.run(op="update", index=99)
    assert not r.success
    r = plan_tool.run(op="bogus")
    assert not r.success


def test_tool_recreate_replaces_old_plan(plan_tool):
    plan_tool.run(op="create", goal="旧任务", steps=["a"])
    plan_tool.run(op="create", goal="新任务", steps=["b"])
    loaded = load_plan(plan_tool.config.workspace)
    assert loaded.goal == "新任务"
    assert [s.text for s in loaded.steps] == ["b"]


# ---------------------------------------------------------------------------
# Agent integration: plan injected into system prompt, cleared on reset
# ---------------------------------------------------------------------------

def test_plan_injected_into_system_prompt(config):
    from agent.agent import Agent
    save_plan(config.workspace, TaskPlan(goal="写一份周报", steps=[PlanStep(text="收集数据")]))
    agent = Agent(llm=object(), config=config, callbacks=object())
    prompt = agent._system_prompt()
    assert "写一份周报" in prompt
    assert "收集数据" in prompt
    assert "# Current task plan" in prompt


def test_reset_clears_plan(config):
    from agent.agent import Agent
    save_plan(config.workspace, TaskPlan(goal="g", steps=[PlanStep(text="s")]))
    agent = Agent(llm=object(), config=config, callbacks=object())
    agent.reset()
    assert load_plan(config.workspace) is None
    assert "# Current task plan" not in agent._system_prompt()


def test_plan_survives_context_shrink(config):
    """计划在系统提示里，上下文收缩只压 _history，目标不应被摘要掉。"""
    from agent.agent import Agent

    class SummarizeLLM:
        def chat_stream(self, messages, tools=None, temperature=0.7):
            return iter([_content_evt("要点摘要")])

    def _content_evt(text):
        from agent.llm_client import StreamEvent
        return StreamEvent(type="content", content=text, tool_calls=[], usage={})

    save_plan(config.workspace, TaskPlan(goal="跨收缩目标X", steps=[PlanStep(text="step1")]))
    agent = Agent(llm=SummarizeLLM(), config=config, callbacks=object())
    # 填充足够长的历史使收缩可触发
    for i in range(12):
        agent._history.append({"role": "user", "content": f"filler {i} " + "z" * 300})
        agent._history.append({"role": "assistant", "content": "ok " + "z" * 300})

    shrunk = agent._shrink_context(reason="test")
    assert shrunk
    # 收缩后系统提示仍包含计划
    assert "跨收缩目标X" in agent._system_prompt()
