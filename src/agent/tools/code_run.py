"""code_run tool: execute arbitrary Python in a sandboxed namespace.

Executes with the workspace as cwd. File writes outside the workspace are
blocked by a guarded builtins.open wrapper, while still allowing pip installs
and network access (the whole point of code_run). A timeout prevents infinite
loops from hanging the agent.

Timeout handling: Python threads cannot be forcefully killed, but we use a
threading.Event-based cooperative cancellation to signal the worker thread
to stop as soon as possible. For truly stuck code (infinite C extension loop),
the daemon thread lingers until process exit, but we track leaked threads and
log a warning so the user is aware.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from typing import List, Optional

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Default timeout (seconds). Long enough for pip installs and network calls,
# short enough that a stuck script doesn't freeze the agent.
_DEFAULT_TIMEOUT = 30
# timeout 上下限：模型可能传 0 / 负数（会导致 join(timeout=0) 立即误判超时），
# 也可能传个天文数字把 agent 挂死。
_MIN_TIMEOUT = 1.0
_MAX_TIMEOUT = 600.0

# stdout/stderr 每路保留的最大字符数。旧实现用无限 StringIO——一段
# `while True: print(...)` 在超时窗口内能产出数 GB 字符串。
_MAX_STREAM_CHARS = 200_000

# Track leaked daemon threads from timed-out code_run calls.
# We can't kill them, but we can warn the user and track the count.
# Cleaned up when the process exits (daemon threads).
#
# Cap the tracked list: a truly stuck thread (tight C loop / blocking I/O
# without timeout) lives until process exit and holds a strong reference to
# its namespace/StringIO via its closure.  Tracking every such thread forever
# would grow this list without bound across many timeouts.  We only need a
# bounded sample for the warning, not an exhaustive registry.
_leaked_threads: List[threading.Thread] = []
_leaked_threads_lock = threading.Lock()
_MAX_TRACKED_LEAKED = 10


class CodeRunTool(Tool):
    name = "code_run"
    description = (
        "Execute Python code (multi-line). stdout/stderr captured. "
        "Can import and pip install packages, call APIs, write files in workspace. "
        f"Killed after {int(_DEFAULT_TIMEOUT)}s unless overridden."
    )
    destructive = True
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."},
            "timeout": {"type": "number", "description": f"Timeout in seconds (default {int(_DEFAULT_TIMEOUT)})."},
        },
        "required": ["code"],
    }

    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self, code: str, timeout: Optional[float] = None) -> ToolResult:
        import builtins as _b

        # Tolerate non-string `code` (some models emit a number or a list).
        # exec() requires str/bytes; coercing here avoids a confusing TypeError
        # that the model would have to debug blind.
        if not isinstance(code, str):
            code = str(code)

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = float(_DEFAULT_TIMEOUT)
        timeout = max(_MIN_TIMEOUT, min(timeout, _MAX_TIMEOUT))

        ws = os.path.realpath(self.config.workspace)
        _real_open = _b.open

        def _guard_open(file, mode="r", *a, **kw):
            if isinstance(file, int):
                raise PermissionError("File descriptor access is not allowed in code_run")
            # 用 realpath 而不是 abspath：workspace 里的 symlink 指向外部文件时
            # （如 os.symlink("/etc/cron.d/x", "pwn") 后 open("pwn","w")），
            # abspath 仍落在 workspace 前缀内，防护被绕过。
            path = os.path.realpath(file if isinstance(file, (str, os.PathLike)) else "")
            if any(m in str(mode) for m in ("w", "a", "x", "+")):
                if not (path == ws or path.startswith(ws + os.sep)):
                    raise PermissionError(f"Refusing to write outside workspace: {file}")
            return _real_open(file, mode, *a, **kw)

        namespace: dict = {"__name__": "__code_run__", "__file__": "<code_run>"}

        class _BoundedIO(io.StringIO):
            """有界 StringIO：超出上限后丢弃后续写入（只标记一次）。"""

            def __init__(self, limit: int = _MAX_STREAM_CHARS):
                super().__init__()
                self._limit = limit
                self._truncated = False

            def write(self, s) -> int:
                if self._truncated:
                    return 0
                if self.tell() + len(s) > self._limit:
                    super().write("\n… [输出超限，已截断]")
                    self._truncated = True
                    return len(s)
                return super().write(s)

        out = _BoundedIO()
        err = _BoundedIO()

        # 主线程侧的进程级状态快照（cwd / stdout / stderr），join 后恢复。
        _prev_cwd = os.getcwd()
        _prev_stdout, _prev_stderr = sys.stdout, sys.stderr

        # Cooperative cancellation event: set by the main thread on timeout,
        # checked by user code via an injected `_cancelled()` function so
        # well-behaved scripts can exit early. For uncooperative code (tight
        # C loop, blocking I/O without timeout), the thread will linger.
        _cancel_evt = threading.Event()

        # Result holder for the worker thread.
        result: dict = {"exc": None, "done": False}

        def _worker():
            try:
                # os.chdir / sys.stdout 替换是**进程级**全局状态，不是线程级的：
                # 执行期间其它线程的相对路径与 print 都会受影响。这里接受这段
                # 时间的短暂影响（相对路径语义是工具契约的一部分）。
                #
                # 恢复动作**全部**由主线程在 join 之后完成（见下方注释），
                # worker 自己绝不恢复——尤其不能碰 sys.stdout：超时泄漏的线程
                # 会在未来任意时刻退出，若它在 finally 里把 stdout 恢复成
                # 「自己启动时的值」，就可能覆盖掉下一次 code_run 刚换上的
                # StringIO，把那次执行的输出整个吞掉。
                os.chdir(ws)
                # Inject cancellation checker into the sandbox namespace so
                # user code can call `_cancelled()` to poll and exit early.
                namespace["__builtins__"] = {**vars(_b), "open": _guard_open}
                namespace["_cancelled"] = _cancel_evt.is_set
                sys.stdout, sys.stderr = out, err
                try:
                    exec(compile(code, "<code_run>", "exec"), namespace)
                finally:
                    result["done"] = True
            except PermissionError as e:
                result["exc"] = e
            except SystemExit as e:
                result["exc"] = SystemExit(f"SystemExit: {e.code}")
            except BaseException as e:
                result["exc"] = e
            finally:
                result["done"] = True

        # Run exec in a daemon thread so a timeout can return control to the
        # agent. The thread cannot be forcefully killed in Python, but as a
        # daemon it won't block process exit.
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)

        # 无论是否超时都由主线程统一恢复进程级全局状态。恢复权**只**属于
        # 主线程：worker 不做任何恢复（见 _worker 注释），因此超时泄漏的
        # 线程无论何时退出都不会再污染 cwd / sys.stdout。
        try:
            os.chdir(_prev_cwd)
            sys.stdout, sys.stderr = _prev_stdout, _prev_stderr
        except OSError as e:
            logger.warning("code_run: failed to restore process cwd/stdout: %s", e)

        if not result["done"]:
            # Thread is still running (infinite loop or slow code).
            # Signal cooperative cancellation so the worker can exit when
            # it next checks `_cancelled()`. Track the leaked thread.
            _cancel_evt.set()
            # 泄漏线程此后只会继续往自己的 StringIO 写（不会再碰全局
            # stdout/cwd），因此并发执行下一次 code_run 是安全的。
            with _leaked_threads_lock:
                # Prune completed threads, then append the new one, capped.
                _leaked_threads[:] = [lt for lt in _leaked_threads if lt.is_alive()]
                _leaked_threads.append(t)
                if len(_leaked_threads) > _MAX_TRACKED_LEAKED:
                    _leaked_threads[:] = _leaked_threads[-_MAX_TRACKED_LEAKED:]
                leaked_count = len(_leaked_threads)
            if leaked_count > 1:
                logger.warning(
                    "code_run: daemon thread leaked due to timeout (total tracked: %d). "
                    "Stuck threads are cleaned up on process exit.",
                    leaked_count,
                )
            return ToolResult(
                False,
                error=f"Code execution timed out after {timeout:.0f}s",
            )

        # Worker completed within timeout — check if the thread we just
        # joined is still in the leaked list and remove it.
        with _leaked_threads_lock:
            if t in _leaked_threads:
                _leaked_threads.remove(t)

        exc = result["exc"]
        if exc is not None:
            if isinstance(exc, PermissionError):
                return ToolResult(False, error=str(exc))
            if isinstance(exc, SystemExit):
                return ToolResult(False, error=str(exc))
            # Reconstruct traceback from the exception for the model to debug.
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            return ToolResult(False, error="".join(tb))

        output = out.getvalue()
        if err.getvalue():
            output += ("\n" if output else "") + "[stderr]\n" + err.getvalue()
        return ToolResult(True, output=output or "(no output)")

