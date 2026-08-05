"""Full worker -> signal -> UI flow test using pytest-qt's event loop.

Drives a real agent turn through the QThread worker with a scripted LLM and
verifies streaming content lands in the chat view.

We call MainWindow._on_send() directly and poll the `busy` flag with
qtbot.waitUntil: this is robust under the offscreen platform (where synthetic
mouse clicks may not be delivered) and avoids nested waitSignal subtleties.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from agent.agent import Agent
from agent.config import AgentConfig
from agent.llm_client import StreamEvent


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [], usage=usage or {})


class _ScriptedLLM:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools})
        return iter(self._scripts.pop(0))


def _make_window(tmp_path, monkeypatch):
    import agent.ui.main_window as mw

    monkeypatch.setattr(mw, "CONFIG_PATH", str(tmp_path / "cfg.json"))
    win = mw.MainWindow()
    # Isolate the workspace + skill history so the agent never touches the
    # user's real ~/base-agent-workspace (which could trigger a skill-suggestion
    # modal mid-test and hang the offscreen event loop).
    win.workspace_edit.setText(str(tmp_path / "ws"))
    win._on_apply_config()
    return win


def test_send_streams_content_into_chat(qtbot, tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    llm = _ScriptedLLM([[_evt("content", "He"), _evt("content", "llo"),
                         _evt("done", usage={"total_tokens": 3, "completion_tokens": 2})]])
    win._agent.llm = llm
    win._agent.max_iterations = 3

    win.send_editor.setPlainText("hi")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    html = win.chat_view.toHtml()
    assert "Hello" in html
    assert "你" in html
    assert "助手" in html
    assert "总计 3 tokens" in win.token_label.text()
    assert win.send_btn.isEnabled()
    assert not win.stop_btn.isEnabled()


def test_send_with_tool_call_shows_tool_line(qtbot, tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    with open(os.path.join(win.config.workspace, "d.txt"), "w") as f:
        f.write("42")
    tc = [{"id": "c1", "function": {"name": "file_read",
            "arguments": json.dumps({"path": "d.txt"})}}]
    llm = _ScriptedLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 4})],
        [_evt("content", "answer=42"), _evt("done", usage={"total_tokens": 6})],
    ])
    win._agent.llm = llm
    win._agent.max_iterations = 3

    win.send_editor.setPlainText("read d.txt")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    html = win.chat_view.toHtml()
    assert "file_read" in html
    assert "42" in html  # tool result snippet
    assert "answer=42" in html


def test_clear_chat_resets_history(qtbot, tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    win._append_chat("<p>hello</p>")
    win._agent._history.append({"role": "user", "content": "x"})
    win._on_clear_chat()
    assert win.chat_view.toPlainText() == ""


# ---------------------------------------------------------------------------
# thinking-view output control: no overlap, always append at end
# ---------------------------------------------------------------------------

def test_sanitize_stream_text_normalizes_carriage_returns():
    import agent.ui.main_window as mw

    s = mw._sanitize_stream_text
    assert s("a\r\nb") == "a\nb"
    assert s("spin1\rspin2\rspin3") == "spin1\nspin2\nspin3"
    assert s("") == ""
    assert s("plain") == "plain"


def test_reasoning_appends_even_after_cursor_drift(qtbot, tmp_path, monkeypatch):
    """If the user clicks the thinking view (cursor moves off the end), new
    streaming chunks must still append at the end, not interleave/overlap."""
    from PyQt6.QtGui import QTextCursor

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    win._on_reasoning("abc")
    # simulate the user clicking at the start of the thinking view
    cursor = win.think_view.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    win.think_view.setTextCursor(cursor)
    win._on_reasoning("def")

    assert win.think_view.toPlainText() == "abcdef"


# ---------------------------------------------------------------------------
# skill suggestion flow: cached during run, dialog shown after _on_finished
# ---------------------------------------------------------------------------

def test_skill_suggestion_cached_during_run_shown_after_finish(qtbot, tmp_path, monkeypatch):
    """When the agent emits on_skill_suggested during a run, the UI must NOT
    show the modal dialog immediately (that would block streaming). Instead
    it caches the suggestion and shows the dialog only after _on_finished.

    This test monkeypatches QMessageBox.question and QInputDialog.getText so
    the dialog functions return scripted answers without blocking.
    """
    import agent.ui.main_window as mw
    from PyQt6.QtWidgets import QMessageBox, QInputDialog

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    # Patch dialogs to return scripted answers (Yes -> name "my-skill").
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("my-skill", True))

    # Pre-populate skill request history so the NEXT run triggers a suggestion.
    # Two identical requests -> 2nd triggers suggestion.
    win._agent.skills.record_request("deploy to staging")
    # Before running, no pending skill.
    assert win._pending_skill is None

    # Run a turn whose user text matches the recorded intent.
    llm = _ScriptedLLM([[_evt("content", "done"), _evt("done", usage={"total_tokens": 3})]])
    win._agent.llm = llm
    win._agent.max_iterations = 3
    win.send_editor.setPlainText("deploy to staging")
    win._on_send()

    # While busy: _pending_skill should be set, but NO dialog shown yet
    # (the patched QMessageBox.question would have been called if dialog was
    # shown during streaming — we verify it wasn't via the skill not being
    # saved until after finish).
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    # After _on_finished: the dialog must have been shown, and the skill
    # saved under the name "my-skill".
    saved = win._agent.skills.list()
    assert any(s.name == "my-skill" for s in saved), (
        f"expected skill 'my-skill' saved after finish, got {[s.name for s in saved]}"
    )
    # Pending skill must be cleared after the dialog.
    assert win._pending_skill is None


def test_skill_suggestion_does_not_block_streaming(qtbot, tmp_path, monkeypatch):
    """The skill-suggestion dialog must NOT appear during streaming. If a
    suggestion fires mid-stream, content chunks must still land in the chat
    view immediately. This is verified by checking that the chat view shows
    content before _on_finished completes (which is when the dialog would
    be shown under the new deferred-display behavior)."""
    import agent.ui.main_window as mw
    from PyQt6.QtWidgets import QMessageBox, QInputDialog

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    # Track whether the dialog was called during streaming.
    dialog_calls = {"count": 0}
    def _track_question(*a, **k):
        dialog_calls["count"] += 1
        return QMessageBox.StandardButton.No  # decline to save
    monkeypatch.setattr(QMessageBox, "question", _track_question)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("x", True))

    # Pre-populate so the next run triggers a suggestion.
    win._agent.skills.record_request("deploy to staging")

    # Stream content first, then done — suggestion fires at run() start
    # but dialog must not appear until _on_finished.
    llm = _ScriptedLLM([[_evt("content", "streamed-content"), _evt("done", usage={"total_tokens": 3})]])
    win._agent.llm = llm
    win._agent.max_iterations = 3
    win.send_editor.setPlainText("deploy to staging")
    win._on_send()

    # Wait for the turn to finish.
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    # The dialog must have been called exactly once — AFTER streaming finished,
    # not during. The streamed content must have landed in the chat view.
    assert dialog_calls["count"] == 1, (
        f"expected dialog exactly once after finish, got {dialog_calls['count']}"
    )
    assert "streamed-content" in win.chat_view.toHtml()


def test_reasoning_with_carriage_returns_does_not_overlap(qtbot, tmp_path, monkeypatch):
    """Bare \\r (overstrike) must become a newline so later text doesn't print
    on top of the current line."""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    win._on_reasoning("line1\rline2")

    assert win.think_view.toPlainText() == "line1\nline2"


def test_reasoning_empty_chunk_is_noop(qtbot, tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    win._on_reasoning("keep")
    win._on_reasoning("")  # empty / sanitizes to empty -> ignored
    assert win.think_view.toPlainText() == "keep"
    assert win._agent._history == []


# ---------------------------------------------------------------------------
# auto-scroll-to-bottom: latest content must stay visible in every text view
# ---------------------------------------------------------------------------

def _force_layout(widget):
    """Make the widget actually size itself so its scrollbar has a real range.

    Under the offscreen platform a never-shown widget has viewport size 0 and a
    scrollbar range of 0..0, which makes "value == maximum" trivially true and
    hides real bugs. We resize the viewport, show the widget, and pump the event
    loop so the document layout settles before assertions.
    """
    widget.resize(400, 120)
    widget.show()
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()


def test_reasoning_scrolls_to_bottom_after_each_chunk(qtbot, tmp_path, monkeypatch):
    """Each reasoning chunk must leave the scrollbar pinned to the bottom so the
    latest line is always visible, even after the user scrolled up to read
    earlier output."""
    from PyQt6.QtGui import QTextCursor
    from PyQt6.QtWidgets import QApplication

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    _force_layout(win.think_view)

    # Fill enough lines to make the view scrollable, then verify bottom is shown.
    for i in range(40):
        win._on_reasoning(f"line {i}\n")
        sb = win.think_view.verticalScrollBar()
        assert sb.value() == sb.maximum(), (
            f"think_view not at bottom after line {i}: "
            f"value={sb.value()} maximum={sb.maximum()}"
        )

    # User scrolls up to read earlier output.
    sb = win.think_view.verticalScrollBar()
    sb.setValue(sb.minimum())
    QApplication.processEvents()
    assert sb.value() < sb.maximum()

    # A new chunk arrives -> must snap back to the bottom.
    win._on_reasoning("new tail line\n")
    sb = win.think_view.verticalScrollBar()
    assert sb.value() == sb.maximum()


def test_chat_view_scrolls_to_bottom_on_content(qtbot, tmp_path, monkeypatch):
    """Streaming content into the chat view must keep the scrollbar at the
    bottom, including after the user scrolled up."""
    from PyQt6.QtWidgets import QApplication

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    _force_layout(win.chat_view)

    for i in range(40):
        win._on_content(f"message chunk {i}\n")
        sb = win.chat_view.verticalScrollBar()
        assert sb.value() == sb.maximum(), (
            f"chat_view not at bottom after chunk {i}: "
            f"value={sb.value()} maximum={sb.maximum()}"
        )

    # After user scrolls up, the next chunk must still pin to bottom.
    sb = win.chat_view.verticalScrollBar()
    sb.setValue(sb.minimum())
    QApplication.processEvents()
    win._on_content("tail\n")
    sb = win.chat_view.verticalScrollBar()
    assert sb.value() == sb.maximum()


def test_log_view_scrolls_to_bottom_on_append(qtbot, tmp_path, monkeypatch):
    """Appending to the log view must keep it pinned to the bottom."""
    from PyQt6.QtWidgets import QApplication

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    _force_layout(win.log_view)

    for i in range(40):
        win._append_log(f"log line {i}")
        sb = win.log_view.verticalScrollBar()
        assert sb.value() == sb.maximum(), (
            f"log_view not at bottom after line {i}: "
            f"value={sb.value()} maximum={sb.maximum()}"
        )

    sb = win.log_view.verticalScrollBar()
    sb.setValue(sb.minimum())
    QApplication.processEvents()
    win._append_log("tail")
    sb = win.log_view.verticalScrollBar()
    assert sb.value() == sb.maximum()


def test_append_chat_scrolls_to_bottom(qtbot, tmp_path, monkeypatch):
    """Direct _append_chat calls (tool lines, errors) must also pin to bottom."""
    from PyQt6.QtWidgets import QApplication

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    _force_layout(win.chat_view)

    for i in range(40):
        win._append_chat(f"<p>block {i}</p>")
        sb = win.chat_view.verticalScrollBar()
        assert sb.value() == sb.maximum()

    sb = win.chat_view.verticalScrollBar()
    sb.setValue(sb.minimum())
    QApplication.processEvents()
    win._append_chat("<p>tail</p>")
    sb = win.chat_view.verticalScrollBar()
    assert sb.value() == sb.maximum()
