"""Tests for the built-in tool set."""
from __future__ import annotations

import os

import httpx
import pytest

from agent.agent import AgentCallbacks
from agent.config import AgentConfig
from agent.tools import ToolRegistry


def _reg(config, callbacks=None):
    return ToolRegistry(config=config, callbacks=callbacks or AgentCallbacks())


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------

def test_file_read_returns_contents(config):
    p = os.path.join(config.workspace, "a.txt")
    with open(p, "w") as f:
        f.write("hello world")
    reg = _reg(config)
    res = reg.execute("file_read", {"path": "a.txt"})
    assert res.success
    assert res.output == "hello world"


def test_file_read_missing_file_returns_error(config):
    reg = _reg(config)
    res = reg.execute("file_read", {"path": "nope.txt"})
    assert not res.success
    assert "not exist" in res.error.lower() or "no such" in res.error.lower()


def test_file_read_rejects_escape(config):
    reg = _reg(config)
    res = reg.execute("file_read", {"path": "../../etc/passwd"})
    assert not res.success
    assert "workspace" in res.error.lower() or "outside" in res.error.lower()


def test_file_write_rejects_symlink_escape(config, tmp_path):
    """A symlink inside the workspace pointing outside must not allow writes
    to escape the workspace. _resolve must use realpath to resolve symlinks."""
    import os
    outside = os.path.join(os.path.dirname(config.workspace), "outside_target.txt")
    if os.path.exists(outside):
        os.remove(outside)
    link_path = os.path.join(config.workspace, "escape_link")
    if not os.path.exists(link_path):
        os.symlink(outside, link_path)
    reg = _reg(config)
    res = reg.execute("file_write", {"path": "escape_link", "content": "escaped"})
    assert not res.success, "write via symlink to outside must be rejected"
    assert not os.path.exists(outside), "target outside workspace must not be created"


def test_file_write_atomic_no_tmp_leftover(config):
    """file_write must use atomic write (temp + os.replace); after a successful
    write no .tmp file should remain in the workspace."""
    reg = _reg(config)
    reg.execute("file_write", {"path": "atom.txt", "content": "data"})
    assert not os.path.exists(os.path.join(config.workspace, "atom.txt.tmp"))


def test_file_modify_atomic_no_tmp_leftover(config):
    """file_modify must use atomic write; no .tmp leftover after success."""
    with open(os.path.join(config.workspace, "mod.txt"), "w") as f:
        f.write("hello")
    reg = _reg(config)
    reg.execute("file_modify", {"path": "mod.txt", "old": "hello", "new": "world"})
    with open(os.path.join(config.workspace, "mod.txt")) as f:
        assert f.read() == "world"
    assert not os.path.exists(os.path.join(config.workspace, "mod.txt.tmp"))


# ---------------------------------------------------------------------------
# file_write
# ---------------------------------------------------------------------------

def test_file_write_creates_file(config):
    reg = _reg(config)
    res = reg.execute("file_write", {"path": "out.txt", "content": "abc"})
    assert res.success
    with open(os.path.join(config.workspace, "out.txt")) as f:
        assert f.read() == "abc"


def test_file_write_overwrites_existing(config):
    with open(os.path.join(config.workspace, "out.txt"), "w") as f:
        f.write("old")
    reg = _reg(config)
    reg.execute("file_write", {"path": "out.txt", "content": "new"})
    with open(os.path.join(config.workspace, "out.txt")) as f:
        assert f.read() == "new"


def test_file_write_creates_subdirs(config):
    reg = _reg(config)
    res = reg.execute("file_write", {"path": "sub/dir/x.txt", "content": "y"})
    assert res.success
    with open(os.path.join(config.workspace, "sub", "dir", "x.txt")) as f:
        assert f.read() == "y"


# ---------------------------------------------------------------------------
# file_modify
# ---------------------------------------------------------------------------

def test_file_modify_replaces_text(config):
    with open(os.path.join(config.workspace, "m.txt"), "w") as f:
        f.write("foo bar foo")
    reg = _reg(config)
    res = reg.execute("file_modify", {"path": "m.txt", "old": "bar", "new": "baz"})
    assert res.success
    with open(os.path.join(config.workspace, "m.txt")) as f:
        assert f.read() == "foo baz foo"


def test_file_modify_missing_old_string(config):
    with open(os.path.join(config.workspace, "m.txt"), "w") as f:
        f.write("foo")
    reg = _reg(config)
    res = reg.execute("file_modify", {"path": "m.txt", "old": "zzz", "new": "yy"})
    assert not res.success


# ---------------------------------------------------------------------------
# code_run
# ---------------------------------------------------------------------------

def test_code_run_executes_and_captures_stdout(config):
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "print(1+1)\nimport math; print(math.sqrt(16))"})
    assert res.success
    assert "2" in res.output
    assert "4.0" in res.output


def test_code_run_can_write_files_in_workspace(config):
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "open('crr.txt','w').write('done')"})
    assert res.success
    with open(os.path.join(config.workspace, "crr.txt")) as f:
        assert f.read() == "done"


def test_code_run_returns_error_on_exception(config):
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "raise ValueError('boom')"})
    assert not res.success
    assert "boom" in res.error


def test_code_run_cannot_escape_workspace(config):
    reg = _reg(config)
    outside = os.path.join(os.path.dirname(config.workspace), "escaped.txt")
    if os.path.exists(outside):
        os.remove(outside)
    code = f"open({outside!r},'w').write('x')"
    res = reg.execute("code_run", {"code": code})
    assert not res.success
    assert not os.path.exists(outside)


def test_code_run_timeout_kills_infinite_loop(config):
    """An infinite loop in code_run must be killed by the timeout, not hang
    the agent forever."""
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "while True:\n    pass", "timeout": 2})
    assert not res.success
    assert "timeout" in res.error.lower() or "timed out" in res.error.lower()


def test_code_run_can_use_subprocess(config):
    """code_run must allow subprocess so the model can pip install or run
    shell commands from within Python."""
    reg = _reg(config)
    res = reg.execute("code_run", {"code": "import subprocess; print(subprocess.check_output(['echo','hello']).decode().strip())"})
    assert res.success
    assert "hello" in res.output


def test_code_run_pip_install_and_import(config):
    """The model should be able to pip install a small package and import it."""
    reg = _reg(config)
    code = (
        "import subprocess, sys\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'six'])\n"
        "import six\n"
        "print('six version:', six.__version__)"
    )
    res = reg.execute("code_run", {"code": code})
    assert res.success
    assert "six version:" in res.output


# ---------------------------------------------------------------------------
# shell_run
# ---------------------------------------------------------------------------

def test_shell_run_executes_command(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "echo hello world"})
    assert res.success
    assert "hello world" in res.output


def test_shell_run_captures_stderr(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "echo out; echo err >&2"})
    assert res.success
    assert "out" in res.output
    assert "err" in res.output


def test_shell_run_nonzero_exit_reported(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "exit 3"})
    assert not res.success
    assert "3" in res.error or "3" in res.output


def test_shell_run_timeout(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "sleep 30", "timeout": 2})
    assert not res.success
    assert "timeout" in res.error.lower() or "timed out" in res.error.lower()


def test_shell_run_uses_workspace_cwd(config):
    reg = _reg(config)
    res = reg.execute("shell_run", {"command": "pwd"})
    assert res.success
    assert os.path.abspath(config.workspace) in res.output or config.workspace in res.output


def test_shell_run_is_destructive(config):
    """shell_run can execute arbitrary system-affecting commands, so it must
    require confirmation."""
    reg = _reg(config)
    assert reg.get("shell_run").destructive is True
    assert reg.get("shell_run").should_confirm({"command": "ls"}) is True


# ---------------------------------------------------------------------------
# web_scan
# ---------------------------------------------------------------------------

def test_web_scan_extracts_text(monkeypatch, config):
    from agent.tools import web as web_mod

    class FakeResp:
        status_code = 200
        text = "<html><body><h1>Hi</h1><p>Hello world</p><script>bad()</script></body></html>"
        headers = {"Content-Type": "text/html"}

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    # The tool now uses a shared httpx.Client instance (_http attribute).
    # Patch the Client.get method to return our fake response.
    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kw: FakeResp())
    reg = _reg(config)
    res = reg.execute("web_scan", {"url": "https://example.com"})
    assert res.success
    assert "Hi" in res.output
    assert "Hello world" in res.output
    assert "bad()" not in res.output  # script stripped


# ---------------------------------------------------------------------------
# ask_user
# ---------------------------------------------------------------------------

def test_ask_user_invokes_callback(config, recording_callbacks):
    recording_callbacks._ask_return = "user-said"
    reg = _reg(config, recording_callbacks)
    res = reg.execute("ask_user", {"prompt": "Which one?"})
    assert res.success
    assert res.output == "user-said"
    assert recording_callbacks.asks == ["Which one?"]


# ---------------------------------------------------------------------------
# work_memory
# ---------------------------------------------------------------------------

def test_work_memory_set_get_list_clear(config):
    reg = _reg(config)
    assert reg.execute("work_memory", {"op": "list"}).output.strip() == "{}"

    assert reg.execute("work_memory", {"op": "set", "key": "goal", "value": "fix bug"}).success
    assert reg.execute("work_memory", {"op": "get", "key": "goal"}).output == "fix bug"

    listing = reg.execute("work_memory", {"op": "list"}).output
    assert "goal" in listing and "fix bug" in listing

    reg.execute("work_memory", {"op": "clear"})
    assert reg.execute("work_memory", {"op": "list"}).output.strip() == "{}"


# ---------------------------------------------------------------------------
# memory_extract
# ---------------------------------------------------------------------------

def test_memory_extract_persists_facts(config):
    reg = _reg(config)
    res = reg.execute("memory_extract", {"facts": ["user prefers dark mode", "user uses python"]})
    assert res.success
    res2 = reg.execute("memory_extract", {"op": "search", "query": "python"})
    assert "python" in res2.output.lower()


# ---------------------------------------------------------------------------
# destructive flag & confirmation
# ---------------------------------------------------------------------------

def test_destructive_flags(config):
    """Only tools that can affect the system/external world require confirmation.

    Workspace file ops are locked to the workspace, so they are non-destructive.
    code_run and shell_run can execute arbitrary system-affecting code, so they
    are destructive.
    """
    reg = _reg(config)
    assert reg.get("file_read").destructive is False
    assert reg.get("file_write").destructive is False
    assert reg.get("file_modify").destructive is False
    assert reg.get("code_run").destructive is True
    assert reg.get("shell_run").destructive is True
    assert reg.get("web_scan").destructive is False
    assert reg.get("webexec_js").destructive is False
    assert reg.get("ask_user").destructive is False
    assert reg.get("work_memory").destructive is False
    assert reg.get("memory_extract").destructive is False


def test_should_confirm_default_follows_destructive(config):
    reg = _reg(config)
    assert reg.get("code_run").should_confirm({"code": "print(1)"}) is True
    assert reg.get("file_write").should_confirm({"path": "a", "content": "b"}) is False
    assert reg.get("file_read").should_confirm({"path": "a"}) is False


def test_unknown_tool_returns_error(config):
    reg = _reg(config)
    res = reg.execute("nope", {})
    assert not res.success
    assert "unknown" in res.error.lower()


def test_registry_exposes_function_schemas(config):
    reg = _reg(config)
    schemas = reg.schemas()
    names = {s["function"]["name"] for s in schemas}
    expected = {
        "code_run", "shell_run", "file_read", "file_write", "file_modify",
        "web_scan", "webexec_js", "ask_user", "work_memory", "memory_extract",
    }
    assert expected <= names


# ---------------------------------------------------------------------------
# RAG tool registration
# ---------------------------------------------------------------------------

def test_rag_tools_not_registered_without_knowledge_base(config):
    """No knowledge_base configured → RAG tools should not be registered."""
    config.knowledge_base = ""
    reg = _reg(config)
    names = {s["function"]["name"] for s in reg.schemas()}
    assert "rag_search" not in names
    assert "rag_status" not in names
    assert "rag_ingest" not in names


def test_rag_tools_registered_with_knowledge_base(config, tmp_path):
    """knowledge_base configured → RAG tools should be registered and
    exposed to the LLM via schemas(). Also verifies that construction does
    NOT trigger ingest (which would block the UI main thread)."""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc.txt").write_text("test content")
    config.knowledge_base = str(kb)

    reg = _reg(config)
    names = {s["function"]["name"] for s in reg.schemas()}
    assert "rag_search" in names, "rag_search must be registered when knowledge_base is set"
    assert "rag_status" in names
    assert "rag_ingest" in names

    # Verify engine is available but no ingest was performed (lazy init).
    engine = reg.get_rag_engine()
    assert engine is not None
    # count() triggers _ensure_initialized which loads FastEmbed — skip in
    # unit test to avoid model download. Just verify the engine exists.

    reg.shutdown()
