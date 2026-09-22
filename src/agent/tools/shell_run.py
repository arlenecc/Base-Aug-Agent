"""shell_run tool: execute shell/terminal commands in the workspace.

Runs the command with the workspace as cwd, captures stdout+stderr, and
enforces a timeout so a hung command cannot block the agent indefinitely.
Marked destructive because shell commands can affect the system arbitrarily;
the agent's confirmation policy prompts the user before execution.
"""
from __future__ import annotations

import collections
import logging
import os
import signal
import subprocess
import threading
import time

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Default timeout (seconds) for shell commands. Can be overridden per-call via
# the `timeout` argument. Long enough for pip installs, short enough that a
# hung command doesn't freeze the agent.
_DEFAULT_TIMEOUT = 120

# 单条命令 stdout/stderr 各保留的最大字符数。旧实现用 capture_output 全量
# 读入内存——一条 `cat 大日志` / `find /` 就能把进程内存打满。
_MAX_OUTPUT_CHARS = 200_000

# 超时后给进程组留出优雅退出的时间，之后再 SIGKILL。
_TERM_GRACE_SECONDS = 3.0


class ShellRunTool(Tool):
    name = "shell_run"
    description = (
        "Run a shell command in the workspace, return stdout+stderr. "
        "For CLI tools, git, pip, build scripts, etc."
    )
    destructive = True
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute (may include pipes, redirects, etc.).",
            },
            "timeout": {
                "type": "number",
                "description": f"Timeout in seconds (default {_DEFAULT_TIMEOUT}).",
            },
        },
        "required": ["command"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, command: str, timeout: float = None) -> ToolResult:
        # Tolerate non-string command (model might emit a list or number).
        if not isinstance(command, str):
            command = str(command)

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = float(_DEFAULT_TIMEOUT)

        ws = os.path.abspath(self.config.workspace)
        try:
            returncode, out, err, timed_out = _run_capture(
                command, cwd=ws, timeout=timeout
            )
        except Exception as e:
            return ToolResult(False, error=f"{type(e).__name__}: {e}")

        if timed_out:
            return ToolResult(
                False,
                error=f"Command timed out after {timeout:.0f}s: {command[:200]}",
                output=_combine(out, err),
            )

        # Combine stdout and stderr so the model sees everything.
        output = _combine(out, err)

        if returncode == 0:
            return ToolResult(True, output=output or "(no output)")
        # Non-zero exit: report as failure but still include output so the
        # model can diagnose what went wrong.
        return ToolResult(
            False,
            output=output,
            error=f"Exit code {returncode}",
        )


def _combine(out: str, err: str) -> str:
    if err:
        return out + ("\n" if out else "") + "[stderr]\n" + err
    return out


class _Collector:
    """Bounded stdout/stderr collector that keeps the head *and* the tail.

    超长输出里最有价值的信息往往在末尾（报错摘要、最后几行结果）。
    只保留开头会让模型看不到命令为什么失败。这里用 deque 保留尾部、
    单独字符串保留头部，中间用省略标记衔接。
    """

    _TAIL_RATIO = 0.25  # 尾部至少保留 25% 的预算

    def __init__(self, limit: int = _MAX_OUTPUT_CHARS):
        self.limit = limit
        self._head_budget = limit - int(limit * self._TAIL_RATIO)
        self._head = []
        self._head_size = 0
        self._tail = collections.deque()
        self._tail_size = 0
        self.truncated = False

    def _drop_tail_overflow(self) -> None:
        while self._tail_size > int(self.limit * self._TAIL_RATIO) and self._tail:
            dropped = self._tail.popleft()
            self._tail_size -= len(dropped)

    def feed(self, text: str) -> None:
        if not text:
            return
        if not self.truncated:
            if self._head_size + len(text) <= self._head_budget:
                self._head.append(text)
                self._head_size += len(text)
                return
            # 头部预算用尽 → 进入尾部模式
            keep = self._head_budget - self._head_size
            if keep > 0:
                self._head.append(text[:keep])
                rest = text[keep:]
            else:
                rest = text
            self.truncated = True
            self._feed_tail(rest)
            return
        self._feed_tail(text)

    def _feed_tail(self, text: str) -> None:
        if not text:
            return
        self._tail.append(text)
        self._tail_size += len(text)
        self._drop_tail_overflow()

    def value(self) -> str:
        text = "".join(self._head)
        if self.truncated:
            text += f"\n… [输出超过 {self.limit} 字符，中间已截断] …"
        text += "".join(self._tail)
        return text


def _reader(stream, collector: _Collector) -> None:
    """Read a pipe in a thread, appending into a bounded collector."""
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            collector.feed(chunk)
    except Exception:  # pragma: no cover - 管道可能在关闭时被中断
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the whole process group.

    ``shell=True`` spawns ``sh -c …``; killing only that PID leaves any
    grandchild (``tail -f``, a dev server, a pip subprocess) running forever
    as an orphan.  We start the child in its own session and signal the entire
    group so nothing survives a timeout.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    deadline = time.monotonic() + _TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def _run_capture(command: str, cwd: str, timeout: float):
    """Run ``command`` in its own process group, capturing bounded output.

    Returns ``(returncode, stdout, stderr, timed_out)``.  On timeout the
    entire process group is killed (see :func:`_kill_process_group`) and
    whatever output was produced before the timeout is still returned so the
    model can see where it got stuck.
    """
    # start_new_session=True puts the shell (and everything it spawns) into a
    # new process group/session so we can reap the whole tree on timeout.
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    out = _Collector()
    err = _Collector()
    t_out = threading.Thread(target=_reader, args=(proc.stdout, out), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, err), daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        try:
            proc.wait(timeout=_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - 极端情况
            pass

    # Give the reader threads a moment to drain whatever is left in the pipes.
    t_out.join(timeout=2.0)
    t_err.join(timeout=2.0)
    # Make sure the pipes are closed even if a reader thread is wedged.
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream and not stream.closed:
                stream.close()
        except Exception:
            pass

    return proc.returncode, out.value(), err.value(), timed_out
