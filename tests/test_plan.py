"""计划与反思能力：规划门、反思门、update_plan 工具、anchor 回注。

对应「先规划 → 每步反思 → 防跑丢/防半路停」的闭环。
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.agent import Agent                      # noqa: E402
from agent.config import AgentConfig               # noqa: E402
from agent.kernel import normalize_plan            # noqa: E402
from agent.kernel.loop import _PLAN_GATE_TURNS     # noqa: E402
from agent.kernel.session import Session           # noqa: E402
from agent.kernel.budget import ContextBudget      # noqa: E402
from agent.llm_client import StreamEvent           # noqa: E402


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [], usage=usage or {})


def _tc(name, args, cid="c1"):
    return [{"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}]


class MockLLM:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools})
        if not self._scripts:
            raise AssertionError("MockLLM ran out of scripted responses")
        return iter(self._scripts.pop(0))


class Noop:
    def is_cancelled(self): return False
    def on_content(self, t): pass
    def on_reasoning(self, t): pass
    def on_tool_start(self, n, a): pass
    def on_tool_end(self, n, r): pass
    def on_log(self, l): pass
    def on_usage(self, u): pass
    def on_token_speed(self, t, s): pass
    def on_finished(self): pass
    def on_plan_updated(self, plan): pass
    def confirm(self, m): return True
    def ask_user(self, p): return "ok"


@pytest.fixture()
def cfg(tmp_path):
    c = AgentConfig()
    c.workspace = str(tmp_path)
    c.knowledge_base = str(tmp_path / "kb")
    return c


@pytest.fixture()
def session():
    return Session(budget=ContextBudget(30_000), raw_turns=2)


# ---------------------------------------------------------------------------
# normalize_plan：脏输入必须被归一，不能让内核抛异常
# ---------------------------------------------------------------------------

def test_normalize_plan_accepts_wellformed():
    p = normalize_plan({"goal": "部署", "steps": [
        {"title": "读配置", "status": "done"},
        {"title": "改端口"},
        "写文档",                      # 字符串形式
    ]})
    assert p is not None
    assert p["goal"] == "部署"
    assert [s["status"] for s in p["steps"]] == ["done", "pending", "pending"]
    assert p["steps"][2]["title"] == "写文档"
    # id 自动编号
    assert [s["id"] for s in p["steps"]] == ["1", "2", "3"]


def test_normalize_plan_rejects_garbage():
    assert normalize_plan(None) is None
    assert normalize_plan({"goal": "x"}) is None                 # 无 steps
    assert normalize_plan({"goal": "x", "steps": []}) is None    # 空 steps
    assert normalize_plan({"goal": "x", "steps": "nope"}) is None
    # 全部无标题 → 空 steps → 拒绝
    assert normalize_plan({"goal": "x", "steps": [{"note": "no title"}]}) is None
    # 非法 status 归一为 pending
    p = normalize_plan({"goal": "g", "steps": [{"title": "a", "status": "WEIRD"}]})
    assert p["steps"][0]["status"] == "pending"


def test_normalize_plan_truncates_long_fields():
    p = normalize_plan({"goal": "G" * 500,
                        "steps": [{"title": "T" * 500, "note": "N" * 500}]})
    assert len(p["goal"]) == 300
    assert len(p["steps"][0]["title"]) == 120
    assert len(p["steps"][0]["note"]) == 200


# ---------------------------------------------------------------------------
# Session：计划渲染与 open 步骤
# ---------------------------------------------------------------------------

def test_session_plan_roundtrip_and_render(session):
    assert session.get_plan() is None
    assert session.plan_open_steps() == []
    assert session.render_plan() == ""

    assert session.set_plan({"goal": "修 bug", "steps": [
        {"title": "复现", "status": "done"},
        {"title": "定位", "status": "doing", "note": "在 auth.py"},
        {"title": "验证"},
    ]}) is True
    text = session.render_plan()
    assert "Goal: 修 bug" in text
    assert "- [✓] 1. 复现" in text
    assert "- [▶] 2. 定位  # 在 auth.py" in text
    assert "- [ ] 3. 验证" in text
    assert "Now: 2. 定位" in text
    assert "Remaining: 2 step(s)" in text
    assert [s["id"] for s in session.plan_open_steps()] == ["2", "3"]

    session.clear_plan()
    assert session.get_plan() is None


def test_anchor_prompt_injects_plan(session):
    session.set_plan({"goal": "上线", "steps": [{"title": "打包", "status": "doing"}]})
    anchor = session.anchor_prompt(extra="用户请求")
    assert "### [PLAN]" in anchor
    assert "Goal: 上线" in anchor
    assert "打包" in anchor


def test_reset_clears_plan(session):
    session.set_plan({"goal": "g", "steps": [{"title": "t"}]})
    session.reset()
    assert session.get_plan() is None


# ---------------------------------------------------------------------------
# 内核：规划门（未规划先提醒）与反思门（未完成不得收尾）
# ---------------------------------------------------------------------------

def test_plan_gate_reminds_before_acting(cfg):
    """前几轮没有计划时，锚点里必须出现规划提示。"""
    mock = MockLLM([
        [_evt("content", "好的"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("帮我部署一下")

    first = mock.calls[0]["messages"]
    anchor = next((str(m.get("content") or "") for m in first
                   if m.get("role") == "user"), "")
    assert "update_plan" in anchor, "首轮应提醒先规划"


def test_plan_gate_stops_after_a_few_turns(cfg):
    """一直不规划也不能无限提醒（一句话任务不该被强制规划）。

    注意：提示一旦注入就成为该轮 user 消息的一部分，会在 raw 窗口里多留
    几轮——这是历史模型的正常行为。所以要用足够多的轮次让窗口整个滚过，
    才能验证「不再有新的注入」。
    """
    n = 8
    mock = MockLLM(
        [[_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}), usage={"total_tokens": 1})]
         for _ in range(n)]
        + [[_evt("content", "读完"), _evt("done", usage={"total_tokens": 1})]]
    )
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("A")
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 20
    agent.kernel.run("读文件")

    hinted = [
        i for i, c in enumerate(mock.calls, 1)
        if any("开始执行前" in str(m.get("content") or "") for m in c["messages"])
    ]
    # 前 3 轮注入 + raw 窗口残留几轮，远小于总轮数
    assert len(hinted) <= _PLAN_GATE_TURNS + 3, f"门控提醒没有上限: {hinted}"
    assert hinted[-1] < n, "最后一轮仍被提醒，说明门控没有停止"


def test_reflection_gate_blocks_premature_finish(cfg):
    """有计划且步骤未完成时，模型想收尾会被拦回来继续。"""
    mock = MockLLM([
        # 第 1 轮：建立计划
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "改端口",
            "steps": [{"title": "读配置"}, {"title": "改端口"}, {"title": "重启验证"}],
        }), usage={"total_tokens": 1})],
        # 第 2 轮：什么工具都不调就想收尾 → 应被反思门拦下
        [_evt("content", "已经读完了"), _evt("done", usage={"total_tokens": 1})],
        # 第 3 轮：被拦后继续执行
        [_evt("done", tool_calls=_tc("file_read", {"path": "a.txt"}), usage={"total_tokens": 1})],
        # 第 4 轮：标记完成
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "patch", "step_id": "2", "status": "done", "note": "已改",
        }), usage={"total_tokens": 1})],
        # 第 5 轮：再次想收尾（还有 1 步未完成）→ 再拦一次
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 1})],
        # 第 6 轮：真正收尾
        [_evt("content", "全部完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    with open(os.path.join(cfg.workspace, "a.txt"), "w") as f:
        f.write("port = 8080")
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 15
    out = agent.kernel.run("把端口改成 9090")

    assert agent.session.get_plan() is not None
    # 计划被建立、步骤被 patch，说明流程真的走过去了
    assert agent.session.plan_open_steps() != []
    assert "全部完成" in out.text
    # 至少 5 次 LLM 调用：被反思门拦了两次
    assert len(mock.calls) >= 5


def test_reflection_gate_gives_up_after_max_and_warns_user(cfg):
    """提醒次数用尽后尊重模型，但回复必须附上未完成清单。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "g", "steps": [{"title": "a"}, {"title": "b"}],
        }), usage={"total_tokens": 1})],
        # 连续 3 次只想文字收尾
        [_evt("content", "我做完了"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "真的做完了"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "就这样吧"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 15
    out = agent.kernel.run("做任务")

    assert "未完成步骤" in out.text
    assert "a" in out.text and "b" in out.text


def test_finish_allowed_when_all_steps_terminal(cfg):
    """计划全部终态化后，模型可以正常收尾，不被拦。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "create", "goal": "g", "steps": [{"title": "a"}],
        }), usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_tc("update_plan", {
            "op": "patch", "step_id": "1", "status": "done", "note": "ok",
        }), usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("做任务")

    assert out.text == "完成"
    assert "未完成步骤" not in out.text
    assert agent.session.plan_open_steps() == []


def test_no_plan_means_no_gate(cfg):
    """没有计划的普通任务不受反思门影响（回归：不能把简单任务卡住）。"""
    mock = MockLLM([[_evt("content", "42"), _evt("done", usage={"total_tokens": 1})]])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    out = agent.kernel.run("6 乘 7 等于几")
    assert out.text == "42"
    assert len(mock.calls) == 1


# ---------------------------------------------------------------------------
# update_plan 工具
# ---------------------------------------------------------------------------

def _registry(cfg, session, callbacks=None):
    from agent.tools import ToolRegistry
    reg = ToolRegistry(config=cfg, callbacks=callbacks or Noop())
    reg.session = session
    return reg


def test_update_plan_tool_lifecycle(cfg, session):
    reg = _registry(cfg, session)

    # create
    res = reg.execute("update_plan", {
        "op": "create", "goal": "修 bug",
        "steps": [{"title": "复现"}, {"title": "修复"}, {"title": "验证"}],
    })
    assert res.success, res.error
    plan = session.get_plan()
    assert plan["goal"] == "修 bug"
    # 第一步自动标 doing
    assert plan["steps"][0]["status"] == "doing"
    assert "Now: 1. 复现" in res.output

    # patch：标完成后自动推进下一步
    res = reg.execute("update_plan", {"op": "patch", "step_id": "1",
                                      "status": "done", "note": "稳定复现"})
    assert res.success, res.error
    plan = session.get_plan()
    assert plan["steps"][0]["status"] == "done"
    assert plan["steps"][0]["note"] == "稳定复现"
    assert plan["steps"][1]["status"] == "doing", "完成后应自动推进下一步"

    # append
    res = reg.execute("update_plan", {"op": "append", "steps": [{"title": "补回归测试"}]})
    assert res.success, res.error
    assert len(session.get_plan()["steps"]) == 4

    # patch 不存在的 id
    res = reg.execute("update_plan", {"op": "patch", "step_id": "99", "status": "done"})
    assert not res.success

    # clear
    res = reg.execute("update_plan", {"op": "clear"})
    assert res.success and session.get_plan() is None

    reg.shutdown()


def test_update_plan_create_accepts_steps_without_goal(cfg, session):
    """goal 选填：漏写时仍应建成计划（steps 才是路径的主体）。"""
    reg = _registry(cfg, session)
    ok = reg.execute("update_plan", {"op": "create", "steps": [{"title": "a"}]})
    assert ok.success, ok.error
    assert session.get_plan()["steps"][0]["title"] == "a"
    reg.shutdown()


def test_update_plan_rejects_bad_input(cfg, session):
    reg = _registry(cfg, session)
    # 有 goal 没 steps → 空计划 → 拒绝
    assert not reg.execute("update_plan", {"op": "create", "goal": "g"}).success
    # 没有任何计划时 patch 无从下手
    assert not reg.execute("update_plan", {"op": "patch", "status": "done"}).success
    # 未知 op
    assert not reg.execute("update_plan", {"op": "bogus"}).success
    # 非法 status
    assert not reg.execute("update_plan", {"op": "patch",
                                          "status": "WEIRD"}).success
    reg.shutdown()


def test_update_plan_notifies_callback(cfg, session):
    seen = []
    cb = Noop()
    cb.on_plan_updated = lambda p: seen.append(p)
    reg = _registry(cfg, session, callbacks=cb)
    reg.execute("update_plan", {"op": "create", "goal": "g",
                                "steps": [{"title": "a"}]})
    assert seen and seen[-1]["goal"] == "g"
    reg.shutdown()


def test_update_plan_is_registered(cfg):
    from agent.tools import ToolRegistry
    reg = ToolRegistry(config=cfg, callbacks=Noop())
    assert "update_plan" in reg.names()
    reg.shutdown()


# ---------------------------------------------------------------------------
# 任务边界：计划跨 run 的复位与保留
# ---------------------------------------------------------------------------

def _plan_call(op, **kw):
    return _tc("update_plan", dict(op=op, **kw))


def test_plan_resets_when_previous_task_finished(cfg):
    """上一个计划全部终态化后，新请求必须清空计划让它重新规划。

    不清的后果：旧计划一直通过 anchor 回注，模型沿着过期计划走、也不再调用
    update_plan —— 表现为「计划面板停在第一次那幅，新任务没人监督」。
    """
    mock = MockLLM([
        [_evt("done", tool_calls=_plan_call("create", goal="任务A",
                                             steps=[{"title": "a"}]), usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan_call("patch", step_id="1", status="done",
                                            note="ok"), usage={"total_tokens": 1})],
        [_evt("content", "A完成"), _evt("done", usage={"total_tokens": 1})],
        # 第二个任务：新请求
        [_evt("content", "B完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.kernel.run("任务A")
    assert agent.session.get_plan() is not None

    agent.kernel.run("任务B")          # 直接调内核不走门面
    assert agent.session.get_plan() is not None, "内核不管任务边界"

    # 走门面（真实入口）：第二个 run 开始时应清掉已完成的计划
    mock2 = MockLLM([[_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})]])
    agent.llm = mock2
    agent.run("任务C")
    assert agent.session.get_plan() is None, "已完成的任务计划应被清空"


def test_plan_survives_when_previous_task_unfinished(cfg):
    """有未完成步骤时保留计划 —— 用户说「继续」要能接着干。"""
    mock = MockLLM([
        [_evt("done", tool_calls=_plan_call("create", goal="任务A",
                                             steps=[{"title": "a"}, {"title": "b"}]),
          usage={"total_tokens": 1})],
        [_evt("done", tool_calls=_plan_call("patch", step_id="1", status="done",
                                            note="ok"), usage={"total_tokens": 1})],
        # 连续三次只想文字收尾 → 反思门提醒用尽后放行（step2 仍未完成）
        [_evt("content", "先这样"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "还是先这样"), _evt("done", usage={"total_tokens": 1})],
        [_evt("content", "算了"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent = Agent(llm=mock, config=cfg, callbacks=Noop())
    agent.max_iterations = 15
    agent.run("任务A")
    assert [s["id"] for s in agent.session.plan_open_steps()] == ["2"]

    # 「继续」：计划必须还在
    mock2 = MockLLM([
        [_evt("done", tool_calls=_plan_call("patch", step_id="2", status="done",
                                            note="补做"), usage={"total_tokens": 1})],
        [_evt("content", "完成"), _evt("done", usage={"total_tokens": 1})],
    ])
    agent.llm = mock2
    agent.run("继续")
    assert agent.session.get_plan() is not None
    assert agent.session.get_plan()["goal"] == "任务A"
    assert agent.session.plan_open_steps() == []


def test_run_start_notifies_plan_to_ui(cfg):
    """每轮开始都把当前计划同步给 UI —— 面板不能是上一轮的残影。"""
    seen = []
    cb = Noop()
    cb.on_plan_updated = lambda p: seen.append(p)
    mock = MockLLM([[_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})]])
    agent = Agent(llm=mock, config=cfg, callbacks=cb)
    agent.run("你好")
    assert seen, "run() 开始时应至少同步一次计划状态"
    assert seen[-1] is None      # 没有计划 → 面板清空
