"""Tests for the agent main loop (LLM mocked)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator, List

import pytest

from agent.agent import Agent
from agent.config import AgentConfig
from agent.llm_client import StreamEvent


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [], usage=usage or {})


class MockLLM:
    """Replays a queue of canned event-lists, one per chat_stream call."""

    def __init__(self, scripts: List[List[StreamEvent]]):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools, "temperature": temperature})
        if not self._scripts:
            raise AssertionError("MockLLM ran out of scripted responses")
        return iter(self._scripts.pop(0))


# ---------------------------------------------------------------------------
# simple reply (no tools)
# ---------------------------------------------------------------------------

def test_agent_streams_content_and_stops(config, recording_callbacks):
    mock = MockLLM([[_evt("content", "He"), _evt("content", "llo"),
                     _evt("done", usage={"total_tokens": 5, "prompt_tokens": 2, "completion_tokens": 3})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")

    assert "".join(recording_callbacks.content) == "Hello"
    assert recording_callbacks.tool_starts == []
    assert recording_callbacks.usages == [{"total_tokens": 5, "prompt_tokens": 2, "completion_tokens": 3}]


def test_agent_streams_reasoning(config, recording_callbacks):
    mock = MockLLM([[_evt("reasoning", "thinking..."),
                     _evt("content", "answer"),
                     _evt("done", usage={"total_tokens": 4})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")

    assert "".join(recording_callbacks.reasoning) == "thinking..."
    assert "".join(recording_callbacks.content) == "answer"


# ---------------------------------------------------------------------------
# one tool call then final answer
# ---------------------------------------------------------------------------

def test_agent_executes_tool_then_answers(config, recording_callbacks):
    # Create a file the tool will read.
    with open(__import__("os").path.join(config.workspace, "data.txt"), "w") as f:
        f.write("42")

    tc = [{"id": "c1", "function": {"name": "file_read", "arguments": json.dumps({"path": "data.txt"})}}]
    mock = MockLLM([
        [_evt("reasoning", "I will read the file"),
         _evt("done", tool_calls=tc, usage={"total_tokens": 10})],
        [_evt("content", "The answer is 42"),
         _evt("done", usage={"total_tokens": 12})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("read data.txt")

    assert recording_callbacks.tool_starts == [("file_read", {"path": "data.txt"})]
    assert len(recording_callbacks.tool_ends) == 1
    name, res = recording_callbacks.tool_ends[0]
    assert name == "file_read"
    assert res.success and res.output == "42"
    assert "".join(recording_callbacks.content) == "The answer is 42"
    # 2 LLM calls
    assert len(mock.calls) == 2
    # second call must include the tool result message
    second_msgs = mock.calls[1]["messages"]
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in second_msgs)


# ---------------------------------------------------------------------------
# multi-step tool calls
# ---------------------------------------------------------------------------

def test_agent_multi_step_tool_chain(config, recording_callbacks):
    tc1 = [{"id": "c1", "function": {"name": "code_run", "arguments": json.dumps({"code": "x=6*7"})}}]
    tc2 = [{"id": "c2", "function": {"name": "ask_user", "arguments": json.dumps({"prompt": "ok?"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc1, usage={"total_tokens": 5})],
        [_evt("done", tool_calls=tc2, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("compute then ask")

    assert [n for n, _ in recording_callbacks.tool_starts] == ["code_run", "ask_user"]
    assert recording_callbacks.asks == ["ok?"]


# ---------------------------------------------------------------------------
# confirmation policy: only system-affecting tools (code_run) ask; workspace
# file ops do not.
# ---------------------------------------------------------------------------

def test_code_run_requires_confirmation(config, recording_callbacks):
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print('hi')"})}}]
    mock = MockLLM([[_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
                    [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)

    recording_callbacks._confirm_return = True
    agent.run("run code")

    assert len(recording_callbacks.confirms) == 1
    assert recording_callbacks.tool_starts == [("code_run", {"code": "print('hi')"})]


def test_code_run_aborted_when_user_declines(config, recording_callbacks):
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print('hi')"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        # after refusal, model should still get a tool message saying denied, then answer
        [_evt("content", "ok skipped"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    recording_callbacks._confirm_return = False
    agent.run("run code")

    assert recording_callbacks.tool_starts == []  # never started
    # tool result message should report denial
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "denied" in tool_msg["content"].lower() or "declined" in tool_msg["content"].lower()


def test_workspace_file_write_does_not_confirm(config, recording_callbacks):
    """Writing a file inside the workspace runs without a confirmation prompt."""
    tc = [{"id": "c1", "function": {"name": "file_write",
            "arguments": json.dumps({"path": "x.txt", "content": "y"})}}]
    mock = MockLLM([[_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
                    [_evt("content", "written"), _evt("done", usage={"total_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("write file")

    assert recording_callbacks.confirms == []  # no prompt
    assert recording_callbacks.tool_starts == [("file_write", {"path": "x.txt", "content": "y"})]
    import os

    assert os.path.exists(os.path.join(config.workspace, "x.txt"))


def test_workspace_file_modify_does_not_confirm(config, recording_callbacks):
    import os

    with open(os.path.join(config.workspace, "m.txt"), "w") as f:
        f.write("foo")
    tc = [{"id": "c1", "function": {"name": "file_modify",
            "arguments": json.dumps({"path": "m.txt", "old": "foo", "new": "bar"})}}]
    mock = MockLLM([[_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
                    [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("edit file")

    assert recording_callbacks.confirms == []
    with open(os.path.join(config.workspace, "m.txt")) as f:
        assert f.read() == "bar"


# ---------------------------------------------------------------------------
# safety: max iterations
# ---------------------------------------------------------------------------

def test_agent_stops_after_max_iterations(config, recording_callbacks):
    tc = [{"id": "c1", "function": {"name": "code_run", "arguments": json.dumps({"code": "print(1)"})}}]
    # always returns another tool call -> infinite loop
    loop = [_evt("done", tool_calls=tc, usage={"total_tokens": 5})]
    mock = MockLLM([loop] * 100)
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.max_iterations = 3
    agent.run("loop")

    assert len(mock.calls) <= 3


# ---------------------------------------------------------------------------
# tool error propagates back to model
# ---------------------------------------------------------------------------

def test_tool_error_is_returned_to_model(config, recording_callbacks):
    tc = [{"id": "c1", "function": {"name": "file_read",
            "arguments": json.dumps({"path": "missing.txt"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "could not read"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("read missing")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "error" in tool_msg["content"].lower() or "not exist" in tool_msg["content"].lower()


# ---------------------------------------------------------------------------
# logs & token speed reporting
# ---------------------------------------------------------------------------

def test_agent_emits_logs_and_speed(config, recording_callbacks):
    import time

    mock = MockLLM([[_evt("content", "hi"), _evt("done", usage={"total_tokens": 5, "completion_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")

    assert len(recording_callbacks.logs) > 0
    # at least one speed report
    assert len(recording_callbacks.speeds) >= 1


# ---------------------------------------------------------------------------
# reasoning streaming: real-time speed + token counting
# ---------------------------------------------------------------------------

def test_speed_emitted_during_reasoning_streaming(config, recording_callbacks):
    """While reasoning chunks are streaming in, the agent must emit real-time
    on_token_speed updates so the UI can show tokens/sec live -- not just a
    single update at the very end."""
    import time
    # A long reasoning stream followed by a short content + done.
    # Inter-chunk delay simulates real streaming so the 0.3s throttle triggers.
    reasoning_chunks = [_evt("reasoning", "thinking " * 20) for _ in range(5)]

    class SlowMockLLM(MockLLM):
        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls.append({"messages": messages, "tools": tools, "temperature": temperature})
            script = self._scripts.pop(0)
            def gen():
                for ev in script:
                    time.sleep(0.08)  # 80ms per chunk → 5 chunks = 400ms > 0.3s throttle
                    yield ev
            return gen()

    mock = SlowMockLLM([reasoning_chunks + [_evt("content", "answer"),
                                        _evt("done", usage={"total_tokens": 50, "completion_tokens": 50})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("think hard")

    # At least one speed update must arrive *during* reasoning, before the
    # final post-done update. We check there are >1 speed reports total.
    assert len(recording_callbacks.speeds) >= 2, (
        f"expected real-time speed updates during reasoning, got {recording_callbacks.speeds}"
    )


def test_speed_emitted_during_content_only_streaming(config, recording_callbacks):
    """For models that emit only content (no reasoning), the agent must still
    emit real-time on_token_speed updates during streaming — not just a single
    update at the end. Otherwise the UI looks frozen until the turn completes."""
    import time
    content_chunks = [_evt("content", "word " * 20) for _ in range(5)]

    class SlowMockLLM(MockLLM):
        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls.append({"messages": messages, "tools": tools, "temperature": temperature})
            script = self._scripts.pop(0)
            def gen():
                for ev in script:
                    time.sleep(0.08)
                    yield ev
            return gen()

    mock = SlowMockLLM([content_chunks + [_evt("done", usage={"total_tokens": 50, "completion_tokens": 50})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("write something")

    assert len(recording_callbacks.speeds) >= 2, (
        f"expected real-time speed updates during content streaming, got {recording_callbacks.speeds}"
    )


def test_stream_end_without_done_logs_warning(config, recording_callbacks):
    """If the LLM stream ends without a 'done' event (network cut / server
    truncation), the agent should log a warning so the issue is visible."""
    # Stream that just stops after content — no done event.
    mock = MockLLM([[_evt("content", "partial reply")]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("hi")

    assert any("without 'done'" in line for line in recording_callbacks.logs), (
        f"expected a warning about missing done event, got logs: {recording_callbacks.logs}"
    )


def test_reasoning_tokens_counted_in_speed(config, recording_callbacks):
    """When the API reports completion_tokens but the model also emitted a lot
    of reasoning text, the final speed must account for the reasoning tokens,
    not just completion_tokens. Otherwise reasoning-heavy models show an
    implausibly low tok/s."""
    # 500 chars of reasoning, but usage claims only 5 completion_tokens
    # (as if the API didn't count reasoning). Speed should not be 5/elapsed.
    reasoning = _evt("reasoning", "x" * 500)
    mock = MockLLM([[reasoning, _evt("content", "ok"),
                     _evt("done", usage={"total_tokens": 10, "completion_tokens": 5})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("think")

    # The final speed report should reflect more than just 5 tokens, because
    # the agent estimates reasoning tokens from the streamed text.
    last_total, last_speed = recording_callbacks.speeds[-1]
    assert last_total > 10, (
        f"total tokens should include estimated reasoning tokens, got {last_total}"
    )


def test_reasoning_tokens_added_to_total_when_api_omits_them(config, recording_callbacks):
    """Some local model APIs return usage that doesn't include reasoning tokens
    at all. The agent must add an estimate so the running total reflects
    actual consumption."""
    # Long reasoning, usage reports 0 total_tokens (API bug / omission)
    reasoning = _evt("reasoning", "y" * 400)
    mock = MockLLM([[reasoning, _evt("content", "done"),
                     _evt("done", usage={"total_tokens": 0, "completion_tokens": 0})]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("think")

    # Despite usage saying 0, the total should be non-zero (estimated).
    last_total, _ = recording_callbacks.speeds[-1]
    assert last_total > 0, "total tokens must include estimated reasoning tokens"


def test_api_reasoning_tokens_used_when_provided(config, recording_callbacks):
    """If the API helpfully provides completion_tokens_details.reasoning_tokens,
    use that exact number instead of a text-length estimate."""
    reasoning = _evt("reasoning", "z" * 300)
    usage = {
        "total_tokens": 100,
        "completion_tokens": 80,
        "completion_tokens_details": {"reasoning_tokens": 60},
    }
    mock = MockLLM([[reasoning, _evt("content", "ok"), _evt("done", usage=usage)]])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("think")

    last_total, _ = recording_callbacks.speeds[-1]
    # API said 100 total; agent should trust that, not inflate with estimates.
    assert last_total == 100


def test_estimate_tokens_helper():
    """The token estimation helper returns a positive int for non-empty text
    and 0 for empty text."""
    from agent.agent import _estimate_tokens

    assert _estimate_tokens("") == 0
    assert _estimate_tokens("hello world") > 0
    # CJK text: each char is roughly 1-2 tokens, so 10 chars >= 5 tokens
    assert _estimate_tokens("你好世界这是一段中文") >= 5
    # longer text -> more tokens
    assert _estimate_tokens("a" * 100) > _estimate_tokens("a" * 10)


# ---------------------------------------------------------------------------
# code_run end-to-end: model writes code -> agent executes -> result fed back
# ---------------------------------------------------------------------------

def test_code_run_result_fed_back_to_model(config, recording_callbacks):
    """The stdout from code_run must appear in the tool message sent back to the
    model so it can reason about the result in the next turn."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print(2+2)"})}}]
    mock = MockLLM([
        [_evt("reasoning", "I'll compute 2+2"), _evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "The answer is 4"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("compute 2+2")

    # tool actually ran and captured stdout
    name, res = recording_callbacks.tool_ends[0]
    assert name == "code_run"
    assert res.success
    assert "4" in res.output
    # the tool message in the second LLM call carries the stdout
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "4" in tool_msg["content"]


def test_code_run_error_traceback_fed_back_to_model(config, recording_callbacks):
    """When code_run raises, the traceback must reach the model so it can fix
    the code on the next turn."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "1/0"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "fixed"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("divide")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    assert "ZeroDivisionError" in res.error
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "ZeroDivisionError" in tool_msg["content"]


# ---------------------------------------------------------------------------
# robustness: malformed tool-call arguments must NOT silently run with {}
# ---------------------------------------------------------------------------

def test_malformed_json_args_reported_to_model(config, recording_callbacks):
    """If the model emits a tool_call whose arguments are not valid JSON, the
    agent must tell the model exactly that (not silently run with empty args
    and produce a confusing 'missing argument' error)."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": "print('hi')"  # not valid JSON, missing braces
           }}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "recovered"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    # the error must mention JSON so the model knows what to fix
    assert "json" in res.error.lower() or "parse" in res.error.lower()
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "json" in tool_msg["content"].lower() or "parse" in tool_msg["content"].lower()


def test_empty_args_for_required_param_reports_clear_error(config, recording_callbacks):
    """A tool call with empty arguments for a tool requiring params must produce
    a clear error naming the missing parameter."""
    tc = [{"id": "c1", "function": {"name": "file_read", "arguments": ""}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("read")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    assert "path" in res.error.lower()


# ---------------------------------------------------------------------------
# API compatibility: assistant messages and tool_call IDs
# ---------------------------------------------------------------------------

def test_assistant_message_has_content_null_when_only_tool_calls(config, recording_callbacks):
    """OpenAI-compatible APIs expect assistant messages with tool_calls to carry
    a `content` field (null). Omitting it causes HTTP 400 on some servers."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "print(1)"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run")

    # The assistant message in the second call's history must include content.
    second_msgs = mock.calls[1]["messages"]
    asst = [m for m in second_msgs if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst, "assistant message with tool_calls must be in history"
    assert "content" in asst[0], "assistant message must have content key (null allowed)"


def test_missing_tool_call_id_gets_generated(config, recording_callbacks):
    """Some local models omit tool_call.id. The agent must synthesize one so the
    tool result can be correlated back (tool_call_id must not be empty)."""
    recording_callbacks._confirm_return = True
    tc = [{"function": {"name": "code_run",
            "arguments": json.dumps({"code": "print(1)"})}}]  # no "id"
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run")

    second_msgs = mock.calls[1]["messages"]
    asst = [m for m in second_msgs if m.get("role") == "assistant" and m.get("tool_calls")][0]
    tool_msgs = [m for m in second_msgs if m.get("role") == "tool"]
    assert tool_msgs, "tool result message must exist"
    # the tool_call id on the assistant side must be non-empty...
    assert asst["tool_calls"][0].get("id"), "tool_call.id must be non-empty"
    # ...and match the tool_call_id on the tool side
    assert tool_msgs[0]["tool_call_id"] == asst["tool_calls"][0]["id"]


def test_multiple_tool_calls_results_all_sent_back(config, recording_callbacks):
    """When the model emits several tool_calls in one turn, every result must
    be in the next request's history, each with its own tool_call_id."""
    recording_callbacks._confirm_return = True
    tcs = [
        {"id": "c1", "function": {"name": "code_run",
         "arguments": json.dumps({"code": "print(1)"})}},
        {"id": "c2", "function": {"name": "file_write",
         "arguments": json.dumps({"path": "x.txt", "content": "y"})}},
    ]
    mock = MockLLM([
        [_evt("done", tool_calls=tcs, usage={"total_tokens": 5})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("do two things")

    second_msgs = mock.calls[1]["messages"]
    tool_msgs = [m for m in second_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    ids = {m["tool_call_id"] for m in tool_msgs}
    assert ids == {"c1", "c2"}


def test_code_run_non_string_code_coerced(config, recording_callbacks):
    """If the model emits code as a non-string (number/dict), code_run must
    coerce it to a string rather than crashing with a TypeError."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": 42})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run number")

    _, res = recording_callbacks.tool_ends[0]
    # Coercion succeeded: the tool ran without a TypeError crash.
    assert res.success
    assert "typeerror" not in res.error.lower()


# ---------------------------------------------------------------------------
# shell_run end-to-end: model calls shell_run -> agent executes -> result back
# ---------------------------------------------------------------------------

def test_shell_run_result_fed_back_to_model(config, recording_callbacks):
    """shell_run stdout must appear in the tool message sent back to the model."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "shell_run",
            "arguments": json.dumps({"command": "echo hello_from_shell"})}}]
    mock = MockLLM([
        [_evt("reasoning", "I'll run a shell command"), _evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "The shell said hello"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("run echo")

    name, res = recording_callbacks.tool_ends[0]
    assert name == "shell_run"
    assert res.success
    assert "hello_from_shell" in res.output
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "hello_from_shell" in tool_msg["content"]


def test_shell_run_requires_confirmation(config, recording_callbacks):
    """shell_run is destructive — it must ask the user before executing."""
    tc = [{"id": "c1", "function": {"name": "shell_run",
            "arguments": json.dumps({"command": "ls"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    recording_callbacks._confirm_return = True
    agent.run("list files")

    assert len(recording_callbacks.confirms) == 1
    assert recording_callbacks.tool_starts[0][0] == "shell_run"


def test_shell_run_error_fed_back_to_model(config, recording_callbacks):
    """A non-zero exit code must be reported back so the model can react."""
    recording_callbacks._confirm_return = True
    tc = [{"id": "c1", "function": {"name": "shell_run",
            "arguments": json.dumps({"command": "exit 7"})}}]
    mock = MockLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 5})],
        [_evt("content", "noted the failure"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    agent.run("fail")

    _, res = recording_callbacks.tool_ends[0]
    assert not res.success
    second_msgs = mock.calls[1]["messages"]
    tool_msg = [m for m in second_msgs if m.get("role") == "tool"][0]
    assert "7" in tool_msg["content"]


def test_code_run_then_shell_run_feedback_loop(config, recording_callbacks):
    """Multi-turn feedback loop: model writes a file via code_run, then reads
    it back via shell_run, then produces a final answer. Verifies the full
    tool-calling cycle: parse -> confirm -> execute -> result -> next call."""
    recording_callbacks._confirm_return = True
    tc1 = [{"id": "c1", "function": {"name": "code_run",
            "arguments": json.dumps({"code": "open('loop.txt','w').write('data123')"})}}]
    tc2 = [{"id": "c2", "function": {"name": "shell_run",
            "arguments": json.dumps({"command": "cat loop.txt"})}}]
    mock = MockLLM([
        [_evt("reasoning", "write file"), _evt("done", tool_calls=tc1, usage={"total_tokens": 5})],
        [_evt("reasoning", "read it back"), _evt("done", tool_calls=tc2, usage={"total_tokens": 5})],
        [_evt("content", "The file contains data123"), _evt("done", usage={"total_tokens": 5})],
    ])
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    answer = agent.run("write and read back")

    assert "data123" in answer
    assert [n for n, _ in recording_callbacks.tool_starts] == ["code_run", "shell_run"]
    # 3 LLM calls: write -> read -> final answer
    assert len(mock.calls) == 3
    # The second LLM call sees code_run's result; the third sees both results
    # (history accumulates). The latest tool result (shell_run) has data123.
    second_tool_msgs = [m for m in mock.calls[1]["messages"] if m.get("role") == "tool"]
    third_tool_msgs = [m for m in mock.calls[2]["messages"] if m.get("role") == "tool"]
    assert len(second_tool_msgs) == 1
    assert len(third_tool_msgs) == 2  # code_run + shell_run results
    # the shell_run result (tool_call_id c2) should carry the file content
    shell_msg = [m for m in third_tool_msgs if m["tool_call_id"] == "c2"][0]
    assert "data123" in shell_msg["content"]


# ---------------------------------------------------------------------------
# context shrink: proactive (90% threshold) + reactive (context_length error)
# ---------------------------------------------------------------------------

def test_is_context_too_long_error_detects_common_patterns():
    """The error-pattern detector should match OpenAI/Anthropic/local server
    variants of context-length-exceeded errors."""
    from agent.agent import Agent

    assert Agent._is_context_too_long_error(
        "chat_stream failed: HTTP 400 {'error': {'message': 'context_length_exceeded'}}"
    )
    assert Agent._is_context_too_long_error("This model's maximum context length is 8192 tokens")
    assert Agent._is_context_too_long_error("prompt is too long")
    assert Agent._is_context_too_long_error("too many tokens in prompt")
    # Non-context errors must not match.
    assert not Agent._is_context_too_long_error("HTTP 500 internal server error")
    assert not Agent._is_context_too_long_error("rate limit exceeded")
    assert not Agent._is_context_too_long_error("invalid api key")


def test_estimate_history_tokens_counts_content_and_tool_calls(config, recording_callbacks):
    """_estimate_history_tokens must include both message content and
    tool_call function names/arguments, since both count against the model's
    context window."""
    from agent.agent import Agent

    agent = Agent(llm=MockLLM([]), config=config, callbacks=recording_callbacks)
    base = agent._estimate_history_tokens()
    # Add a long user message.
    agent._history.append({"role": "user", "content": "x" * 400})
    after_user = agent._estimate_history_tokens()
    assert after_user > base
    # Add an assistant message with a tool_call — its name and arguments
    # must contribute additional tokens beyond just the role overhead.
    agent._history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "c1",
            "function": {"name": "code_run", "arguments": json.dumps({"code": "y" * 400})},
        }],
    })
    after_tool = agent._estimate_history_tokens()
    assert after_tool > after_user


def test_proactive_shrink_triggers_at_threshold(config, recording_callbacks):
    """When estimated context size exceeds 90% of max_context_tokens, the
    agent must proactively summarize older messages before calling the LLM."""
    # Build a long history so the estimated tokens exceed the small budget.
    long_text = "x" * 200
    # Summary LLM call: returns a fixed summary; main call: short answer.
    summary_script = [_evt("content", "用户想做任务A，已调用 code_run"), _evt("done", usage={})]
    final_script = [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})]
    mock = MockLLM([summary_script, final_script])

    # Set a tiny budget so threshold triggers immediately.
    config.max_context_tokens = 50  # tiny: any history will exceed 90% = 45
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    # Pre-populate history with enough messages to be shrinkable (>keep_recent+2).
    for i in range(10):
        agent._history.append({"role": "user", "content": f"msg {i}: {long_text}"})
        agent._history.append({"role": "assistant", "content": f"reply {i}: {long_text}"})
    agent._history.append({"role": "user", "content": "final request"})

    agent.run("final request")

    # on_context_shrunk must have fired with a reason mentioning 主动收缩.
    assert len(recording_callbacks.context_shrinks) >= 1, (
        f"expected proactive shrink event, got {recording_callbacks.context_shrinks}"
    )
    summary, reason = recording_callbacks.context_shrinks[0]
    assert "主动收缩" in reason
    assert summary  # non-empty


def test_proactive_shrink_keeps_recent_messages(config, recording_callbacks):
    """After a shrink, the most recent N messages must remain verbatim in
    history so the agent retains immediate task context. The first history
    message must be the summary marker (not a dangling tool result)."""
    summary_script = [_evt("content", "summary of old convo"), _evt("done", usage={})]
    final_script = [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})]
    mock = MockLLM([summary_script, final_script])
    config.max_context_tokens = 30
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    # Pre-populate: 12 older messages + 1 final user request.
    for i in range(12):
        agent._history.append({"role": "user", "content": f"old msg {i} " + "y" * 100})
    recent_user_msg = {"role": "user", "content": "current question"}
    agent._history.append(recent_user_msg)

    agent.run("current question")

    # The first history message must now be the summary marker.
    assert agent._history[0]["role"] == "user"
    assert "上下文摘要" in agent._history[0]["content"]
    # The most recent user request must still be present verbatim somewhere
    # in the kept tail of history.
    contents = [m.get("content", "") for m in agent._history]
    assert any("current question" in (c or "") for c in contents)
    # And the very first history message is NOT a dangling tool result.
    assert agent._history[0].get("role") != "tool"


def test_shrink_no_dangling_tool_at_boundary(config, recording_callbacks):
    """If the recent-messages slice starts with a tool result (because the
    slice boundary falls right after the assistant that made the tool call),
    the shrink must pull the preceding assistant back so the tool message
    is not dangling. A dangling tool message (no preceding assistant with
    matching tool_calls) causes the API to reject the request."""
    summary_script = [_evt("content", "summary"), _evt("done", usage={})]
    final_script = [_evt("content", "ok"), _evt("done", usage={"total_tokens": 5})]
    mock = MockLLM([summary_script, final_script])
    config.max_context_tokens = 30
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    # Fill old history with large user messages to trigger shrink.
    for i in range(10):
        agent._history.append({"role": "user", "content": f"old {i} " + "x" * 100})
    # Now craft recent messages so the slice boundary falls on a tool result.
    # _shrink_keep_recent = 6. We place: assistant(tool_calls) + tool(result)
    # right at the boundary so tool(result) would be the first of recent.
    agent._history.append({"role": "user", "content": "do something"})
    agent._history.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "file_read", "arguments": '{"path":"a.txt}'}}],
    })
    agent._history.append({
        "role": "tool", "tool_call_id": "call_1", "name": "file_read",
        "content": "file contents here",
    })
    # Pad the tail so the tool message lands at position -6 (start of recent).
    for i in range(4):
        agent._history.append({"role": "user", "content": f"tail {i}"})

    agent.run("final question")

    # After shrink, the first message is the summary marker (user role).
    assert agent._history[0]["role"] == "user"
    assert "上下文摘要" in agent._history[0]["content"]
    # No dangling tool message: every tool message must be preceded by an
    # assistant message with tool_calls.
    for i, msg in enumerate(agent._history):
        if msg.get("role") == "tool":
            assert i > 0, "tool message at position 0 is dangling"
            prev = agent._history[i - 1]
            assert prev.get("role") == "assistant", (
                f"tool message at {i} preceded by {prev.get('role')}, not assistant"
            )
            assert prev.get("tool_calls"), (
                f"tool message at {i} has no matching tool_calls in preceding assistant"
            )


def test_proactive_shrink_appends_to_memory_md(config, recording_callbacks):
    """After a shrink, the summary must be appended to workspace/memory.md
    with a timestamp header, for long-term/time-based recall."""
    import os
    summary_script = [_evt("content", "用户的关键需求是部署到生产环境"), _evt("done", usage={})]
    final_script = [_evt("content", "done"), _evt("done", usage={"total_tokens": 5})]
    mock = MockLLM([summary_script, final_script])
    config.max_context_tokens = 40
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    for i in range(10):
        agent._history.append({"role": "user", "content": f"old {i} " + "z" * 80})
    agent._history.append({"role": "user", "content": "go"})

    agent.run("go")

    memory_path = os.path.join(config.workspace, "memory.md")
    assert os.path.exists(memory_path), "memory.md must be created after shrink"
    with open(memory_path, encoding="utf-8") as f:
        content = f.read()
    # Must contain a timestamped header.
    assert "上下文摘要" in content
    assert "触发原因" in content
    # Must contain the LLM-produced summary text.
    assert "部署到生产环境" in content
    # No leftover temp file.
    assert not os.path.exists(memory_path + ".tmp")


def test_reactive_shrink_on_context_length_error(config, recording_callbacks):
    """If the LLM returns a context_length_exceeded error, the agent must
    shrink the context and retry the same iteration (not fail the turn)."""
    # First call raises a context-length error; second (post-shrink) call
    # returns a summary; third is the actual retry that produces the answer.
    # The summary call is a separate chat_stream invocation.
    summary_script = [_evt("content", "summarized old context"), _evt("done", usage={})]
    final_script = [_evt("content", "recovered"), _evt("done", usage={"total_tokens": 5})]

    class ErrorThenRecoverLLM:
        def __init__(self):
            self.calls = []
            self._scripts = iter([
                RuntimeError("chat_stream failed: HTTP 400 context_length_exceeded"),
                summary_script,
                final_script,
            ])

        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls.append({"messages": messages})
            nxt = next(self._scripts)
            if isinstance(nxt, Exception):
                raise nxt
            return iter(nxt)

    mock = ErrorThenRecoverLLM()
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    # Pre-populate enough history so shrink has something to work with.
    for i in range(10):
        agent._history.append({"role": "user", "content": f"old {i} " + "a" * 100})
    agent._history.append({"role": "user", "content": "retry me"})

    answer = agent.run("retry me")

    # The turn must have recovered, not failed.
    assert answer == "recovered"
    # A shrink event must have fired (reactive, mentioning context_length_exceeded).
    assert len(recording_callbacks.context_shrinks) >= 1
    _, reason = recording_callbacks.context_shrinks[0]
    assert "context_length_exceeded" in reason or "被动收缩" in reason
    # on_error must NOT have been emitted (we recovered).
    # (recording_callbacks has no on_error list, but the turn returned content,
    # which itself proves the error path wasn't taken.)


def test_reactive_shrink_max_attempts_cap(config, recording_callbacks):
    """If the LLM keeps returning context_length_exceeded even after shrinking,
    the agent must give up after _max_shrinks_per_run attempts and surface the
    error, rather than looping forever."""
    class AlwaysTooLongLLM:
        def __init__(self):
            self.calls = 0

        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls += 1
            raise RuntimeError("context_length_exceeded")

    mock = AlwaysTooLongLLM()
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    for i in range(20):
        agent._history.append({"role": "user", "content": f"old {i} " + "b" * 100})
    agent._history.append({"role": "user", "content": "stuck"})

    # Must not hang; must return (empty final_content is OK).
    agent.max_iterations = 5
    agent.run("stuck")

    # Should have attempted at most _max_shrinks_per_run shrinks before giving up.
    assert len(recording_callbacks.context_shrinks) <= agent._max_shrinks_per_run


def test_shrink_skipped_when_history_too_small(config, recording_callbacks):
    """If history is too small to shrink (< keep_recent + 2 messages), the
    shrink must be a no-op so we don't summarize the only context we have."""
    mock = MockLLM([[_evt("content", "answer"), _evt("done", usage={"total_tokens": 5})]])
    config.max_context_tokens = 10  # tiny, will trigger threshold
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    # Only 2 messages — too small to shrink.
    agent._history.append({"role": "user", "content": "hi"})

    agent.run("hi")

    # No shrink should have fired.
    assert recording_callbacks.context_shrinks == []


def test_fallback_summary_when_llm_unavailable(config, recording_callbacks):
    """If the summarize LLM call itself fails (raises), the agent must fall
    back to a structural summary (user messages + tool names) rather than
    leaving the summary empty."""
    class FailingSummarizerLLM:
        def __init__(self):
            self.calls = 0

        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls += 1
            # First call = summarize; raise. Second call = real reply.
            if self.calls == 1:
                raise RuntimeError("summarize endpoint down")
            return iter([_evt("content", "recovered"), _evt("done", usage={"total_tokens": 5})])

    mock = FailingSummarizerLLM()
    config.max_context_tokens = 30
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks)
    # Old messages with a user request and a tool_call.
    agent._history.append({"role": "user", "content": "please deploy to staging"})
    agent._history.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "shell_run", "arguments": "{}"}}],
    })
    agent._history.append({"role": "tool", "tool_call_id": "c1", "name": "shell_run", "content": "ok"})
    for i in range(8):
        agent._history.append({"role": "user", "content": f"filler {i} " + "c" * 60})
        agent._history.append({"role": "assistant", "content": f"r{i}"})
    agent._history.append({"role": "user", "content": "go"})

    agent.run("go")

    # The shrink must have fired, and the summary must mention the user request
    # (extracted via the fallback path, not the LLM).
    assert len(recording_callbacks.context_shrinks) == 1
    summary, _ = recording_callbacks.context_shrinks[0]
    assert "deploy to staging" in summary or "shell_run" in summary


# ---------------------------------------------------------------------------
# RAG integration: agent discovers and calls rag_status / rag_search
# ---------------------------------------------------------------------------

def test_agent_calls_rag_tools_when_knowledge_base_configured(
    config, recording_callbacks, tmp_path, fake_ef, monkeypatch
):
    """End-to-end RAG tool invocation through the agent loop.

    When a knowledge_base is configured, the agent must:
    1. Expose rag_status / rag_search / rag_ingest in the tool schema list
       passed to the LLM.
    2. Actually dispatch LLM-emitted tool_calls to the RAGEngine.
    3. Feed the rag_search result back to the LLM as a tool message so the
       next turn can ground its answer in retrieved context.

    Uses fake_ef (no FastEmbed model download) by monkeypatching
    RAGEngine in the engine module — ToolRegistry's _register_rag_tools
    imports RAGEngine lazily inside the function, so the patch is picked up.
    """
    import json as _json

    from agent.rag import engine as engine_mod
    from agent.tools.base import ToolRegistry

    # 1. Set up a knowledge base with a unique fact the LLM cannot know.
    kb = tmp_path / "kb"
    kb.mkdir()
    unique_fact = "ZetaCloud 内部代号 ProjectAurora2024"
    (kb / "company_faq.md").write_text(
        f"# 公司内部 FAQ\n\n"
        f"## 项目代号\n\n{unique_fact}\n"
        f"该项目于 2024 年第三季度启动，负责人是张工。\n",
        encoding="utf-8",
    )

    # 2. Patch RAGEngine so the registry's _register_rag_tools uses fake_ef
    #    (no FastEmbed / ONNX model download).
    class _FakeRAGEngine(engine_mod.RAGEngine):
        def __init__(self, workspace, knowledge_base="", **kwargs):
            kwargs.setdefault("embedding_function", fake_ef)
            super().__init__(workspace=workspace, knowledge_base=knowledge_base, **kwargs)

    monkeypatch.setattr(engine_mod, "RAGEngine", _FakeRAGEngine)

    # 3. Configure knowledge_base and build the registry.
    config.knowledge_base = str(kb)
    registry = ToolRegistry(config=config, callbacks=recording_callbacks)

    # 4. Verify RAG tools are exposed to the LLM via schemas().
    schema_names = {s["function"]["name"] for s in registry.schemas()}
    assert "rag_search" in schema_names, "rag_search must be in tool schemas"
    assert "rag_status" in schema_names, "rag_status must be in tool schemas"
    assert "rag_ingest" in schema_names, "rag_ingest must be in tool schemas"

    # 5. Ingest synchronously so the vector store has data to search.
    rag_engine = registry.get_rag_engine()
    assert rag_engine is not None, "RAG engine was not initialized"
    stats = rag_engine.ingest(force=True)
    assert stats["chunks"] > 0, f"ingest produced no chunks: {stats}"
    # Refresh the LanceDB table handle so the upcoming search sees new rows.
    registry.reload_rag()

    # 6. Script the LLM: turn 1 calls rag_status, turn 2 calls rag_search,
    #    turn 3 produces the final grounded answer.
    tc_status = [{"id": "c1", "function": {"name": "rag_status",
                  "arguments": "{}"}}]
    tc_search = [{"id": "c2", "function": {"name": "rag_search",
                  "arguments": _json.dumps({"query": "项目代号是什么"})}}]
    mock = MockLLM([
        [_evt("reasoning", "First check what's in the knowledge base."),
         _evt("done", tool_calls=tc_status, usage={"total_tokens": 5})],
        [_evt("reasoning", "Now search for the project codename."),
         _evt("done", tool_calls=tc_search, usage={"total_tokens": 5})],
        [_evt("content", f"根据知识库，{unique_fact}。"),
         _evt("done", usage={"total_tokens": 20})],
    ])

    # 7. Run the agent.
    agent = Agent(llm=mock, config=config, callbacks=recording_callbacks,
                  tool_registry=registry)
    agent.run("公司项目的代号是什么？")

    # 8. Both RAG tools must have been dispatched, in order.
    called = [n for n, _ in recording_callbacks.tool_starts]
    assert called == ["rag_status", "rag_search"], (
        f"expected [rag_status, rag_search], got {called}"
    )

    # 9. Both tool calls must have succeeded.
    for name, res in recording_callbacks.tool_ends:
        assert res.success, f"{name} failed: {res.error}"

    # 10. The rag_search result must carry the unique fact.
    search_name, search_result = recording_callbacks.tool_ends[1]
    assert search_name == "rag_search"
    assert unique_fact in search_result.output, (
        f"rag_search result missing unique fact: {search_result.output[:300]}"
    )

    # 11. The rag_search result must be in the third LLM call's history so
    #     the model can ground its final answer in retrieved context.
    third_msgs = mock.calls[2]["messages"]
    rag_tool_msgs = [m for m in third_msgs
                     if m.get("role") == "tool" and m.get("name") == "rag_search"]
    assert rag_tool_msgs, "rag_search result not in third LLM call history"
    assert unique_fact in rag_tool_msgs[0]["content"], (
        "RAG context not propagated to LLM for the final answer"
    )

    # 12. The agent's final visible reply should reference the retrieved fact.
    assert unique_fact in "".join(recording_callbacks.content), (
        "agent final reply did not include the retrieved knowledge base fact"
    )

    # 13. Release RAG engine resources (no leak across tests).
    registry.shutdown()


def test_agent_registers_rag_tools_with_default_knowledge_base(
    config, recording_callbacks
):
    """When knowledge_base is empty, the agent defaults to
    <workspace>/knowledge_base and registers RAG tools so the agent can
    discover the local knowledge base."""
    from agent.tools.base import ToolRegistry

    config.knowledge_base = ""
    registry = ToolRegistry(config=config, callbacks=recording_callbacks)
    schema_names = {s["function"]["name"] for s in registry.schemas()}
    assert "rag_search" in schema_names
    assert "rag_status" in schema_names
    assert "rag_ingest" in schema_names
    registry.shutdown()


# ---------------------------------------------------------------------------
# Daily auto-extracted facts (short-term memory) — 短期记忆只保留当天事实
# ---------------------------------------------------------------------------

def test_parse_daily_facts_valid():
    """_parse_daily_facts 解析合法的当天桶。"""
    from agent.agent import Agent
    raw = '{"date": "2026-08-15", "facts": ["f1", "f2"]}'
    date, facts = Agent._parse_daily_facts(raw)
    assert date == "2026-08-15"
    assert facts == ["f1", "f2"]


def test_parse_daily_facts_malformed():
    """_parse_daily_facts 对空/非法输入返回 ("", [])，视为需重置。"""
    from agent.agent import Agent
    assert Agent._parse_daily_facts("") == ("", [])
    assert Agent._parse_daily_facts("not json") == ("", [])
    assert Agent._parse_daily_facts('{"date": 123, "facts": "oops"}') == ("", [])


def test_parse_daily_facts_filters_non_strings():
    """facts 里的非字符串项被过滤。"""
    from agent.agent import Agent
    raw = '{"date": "2026-08-15", "facts": ["ok", 123, null, "  ", "also"]}'
    date, facts = Agent._parse_daily_facts(raw)
    assert date == "2026-08-15"
    assert facts == ["ok", "also"]


def test_append_daily_facts_writes_and_dedups(config, recording_callbacks):
    """_append_daily_facts 写入当天桶并去重。"""
    import datetime
    from agent.agent import Agent
    from agent.tools.memory import _work_memory

    agent = Agent(llm=MockLLM([[_evt("content", "x"), _evt("done")]]),
                  config=config, callbacks=recording_callbacks)

    agent._append_daily_facts(["事实A", "事实B"])
    agent._append_daily_facts(["事实B", "事实C"])  # B 重复，应去重

    wm = _work_memory(config)
    date, facts = agent._parse_daily_facts(wm.get("__auto_facts__") or "")
    assert date == datetime.date.today().isoformat()
    assert facts == ["事实A", "事实B", "事实C"]


def test_append_daily_facts_resets_on_new_day(config, recording_callbacks):
    """跨天后追加，旧当天桶应被清空（历史归长期记忆）。"""
    import datetime
    from agent.agent import Agent
    from agent.tools.memory import _work_memory

    agent = Agent(llm=MockLLM([[_evt("content", "x"), _evt("done")]]),
                  config=config, callbacks=recording_callbacks)

    # 预置一个「昨天」的桶。
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    _work_memory(config).set(
        "__auto_facts__",
        json.dumps({"date": yesterday, "facts": ["旧事实"]}, ensure_ascii=False),
    )

    # 今天追加新事实 → 旧事实应被丢弃。
    agent._append_daily_facts(["新事实"])

    wm = _work_memory(config)
    date, facts = agent._parse_daily_facts(wm.get("__auto_facts__") or "")
    assert date == datetime.date.today().isoformat()
    assert facts == ["新事实"], f"跨天应只保留当天事实，got {facts}"

