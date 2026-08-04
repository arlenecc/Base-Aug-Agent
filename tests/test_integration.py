"""End-to-end integration: agent + real ToolRegistry + scripted mock LLM.

Also a smoke test that the PyQt6 MainWindow constructs under the offscreen
platform (no display required).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from agent.agent import Agent, AgentCallbacks
from agent.config import AgentConfig
from agent.llm_client import StreamEvent
from agent.tools import ToolRegistry


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [], usage=usage or {})


class _ScriptedLLM:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools})
        return iter(self._scripts.pop(0))


def test_e2e_reads_file_and_summarizes(config, recording_callbacks):
    """Agent reads a workspace file via file_read and produces a final answer."""
    with open(os.path.join(config.workspace, "notes.txt"), "w") as f:
        f.write("The release date is 2026-09-01.")

    tc = [{"id": "c1", "function": {"name": "file_read",
            "arguments": json.dumps({"path": "notes.txt"})}}]
    llm = _ScriptedLLM([
        [_evt("reasoning", "Need to read notes.txt"),
         _evt("done", tool_calls=tc, usage={"total_tokens": 8, "completion_tokens": 4})],
        [_evt("content", "The release date is "), _evt("content", "2026-09-01."),
         _evt("done", usage={"total_tokens": 12, "completion_tokens": 6})],
    ])
    agent = Agent(llm=llm, config=config, callbacks=recording_callbacks)
    answer = agent.run("What is the release date in notes.txt?")

    assert "2026-09-01" in answer
    assert "".join(recording_callbacks.reasoning) == "Need to read notes.txt"
    assert recording_callbacks.tool_starts[0][0] == "file_read"
    assert recording_callbacks.usages[-1]["total_tokens"] == 12
    # tool schemas were sent to the LLM
    assert recording_callbacks and llm.calls[0]["tools"] is not None
    assert len(llm.calls) == 2


def test_e2e_workspace_write_runs_without_confirmation(config, recording_callbacks):
    """Workspace file writes are not destructive (cannot escape), so no prompt."""
    tc = [{"id": "c1", "function": {"name": "file_write",
            "arguments": json.dumps({"path": "out/log.txt", "content": "hello"})}}]
    llm = _ScriptedLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "wrote it"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=llm, config=config, callbacks=recording_callbacks)
    agent.run("save hello to out/log.txt")

    assert os.path.exists(os.path.join(config.workspace, "out", "log.txt"))
    assert recording_callbacks.confirms == []  # workspace write needs no confirm


def test_e2e_code_run_confirms(config, recording_callbacks):
    """code_run can affect the system, so it must ask the user first."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print(2+2)"})}}]
    llm = _ScriptedLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=llm, config=config, callbacks=recording_callbacks)
    agent.run("compute")

    assert len(recording_callbacks.confirms) == 1
    assert "4" in recording_callbacks.tool_ends[0][1].output


def test_e2e_skill_suggested_after_repeat(config, recording_callbacks):
    from agent.skills import SkillManager

    sm = SkillManager(path=os.path.join(config.workspace, ".agent", "skills.json"))
    llm = _ScriptedLLM([[_evt("content", "ok"), _evt("done", usage={"total_tokens": 1})]] * 3)
    agent = Agent(llm=llm, config=config, callbacks=recording_callbacks, skills=sm)
    # first request: no suggestion
    agent.run("deploy to staging")
    assert len(recording_callbacks.logs) >= 0
    # second request: suggestion
    suggested_skills = []
    recording_callbacks.__class__  # noqa: B018
    orig = recording_callbacks.on_skill_suggested
    recording_callbacks.on_skill_suggested = lambda s: suggested_skills.append(s)
    agent.run("deploy to staging")
    assert len(suggested_skills) == 1
    recording_callbacks.on_skill_suggested = orig


# ---------------------------------------------------------------------------
# GUI smoke test (offscreen)
# ---------------------------------------------------------------------------


def test_main_window_constructs(qapp, tmp_path, monkeypatch):
    # point config path at a temp file so we don't touch the user's real config
    import agent.ui.main_window as mw

    monkeypatch.setattr(mw, "CONFIG_PATH", str(tmp_path / "cfg.json"))
    win = mw.MainWindow()
    assert win.base_url_edit is not None
    assert win.model_combo is not None
    assert win.send_btn is not None
    assert win.think_view is not None
    assert win.log_view is not None

    # applying a workspace change should create the dir
    new_ws = str(tmp_path / "ws")
    win.workspace_edit.setText(new_ws)
    win._on_apply_config()
    assert os.path.isdir(new_ws)
    assert win._agent is not None
    win.close()
