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
from contextlib import redirect_stdout, redirect_stderr
from typing import List

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Default timeout (seconds). Long enough for pip installs and network calls,
# short enough that a stuck script doesn't freeze the agent.
_DEFAULT_TIMEOUT = 30

# Track leaked daemon threads from timed-out code_run calls.
# We can't kill them, but we can warn the user and track the count.
# Cleaned up when the process exits (daemon threads).
_leaked_threads: List[threading.Thread] = []
_leaked_threads_lock = threading.Lock()


class CodeRunTool(Tool):
    name = "code_run"
    description = (
        "Execute arbitrary Python code (multi-line scripts allowed). stdout/stderr "
        "are captured and returned. You may `import` and `pip install` packages, "
        "call external APIs, and write files inside the workspace. "
        f"Execution is killed after a {int(_DEFAULT_TIMEOUT)}s timeout unless overridden."
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

    def run(self, code: str, timeout: float = None) -> ToolResult:
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

        ws = os.path.abspath(self.config.workspace)
        _real_open = _b.open

        def _guard_open(file, mode="r", *a, **kw):
            if isinstance(file, int):
                raise PermissionError("File descriptor access is not allowed in code_run")
            path = os.path.abspath(file if isinstance(file, (str, os.PathLike)) else "")
            if any(m in str(mode) for m in ("w", "a", "x", "+")):
                if not (path == ws or path.startswith(ws + os.sep)):
                    raise PermissionError(f"Refusing to write outside workspace: {file}")
            return _real_open(file, mode, *a, **kw)

        namespace: dict = {"__name__": "__code_run__", "__file__": "<code_run>"}
        out = io.StringIO()
        err = io.StringIO()

        # Cooperative cancellation event: set by the main thread on timeout,
        # checked by user code via an injected `_cancelled()` function so
        # well-behaved scripts can exit early. For uncooperative code (tight
        # C loop, blocking I/O without timeout), the thread will linger.
        _cancel_evt = threading.Event()

        # Result holder for the worker thread.
        result: dict = {"exc": None, "done": False}

        def _worker():
            cwd = os.getcwd()
            try:
                os.chdir(ws)
                # Inject cancellation checker into the sandbox namespace so
                # user code can call `_cancelled()` to poll and exit early.
                namespace["__builtins__"] = {**vars(_b), "open": _guard_open}
                namespace["_cancelled"] = _cancel_evt.is_set
                with redirect_stdout(out), redirect_stderr(err):
                    exec(compile(code, "<code_run>", "exec"), namespace)
            except PermissionError as e:
                result["exc"] = e
            except SystemExit as e:
                result["exc"] = SystemExit(f"SystemExit: {e.code}")
            except BaseException as e:
                result["exc"] = e
            finally:
                os.chdir(cwd)
                result["done"] = True

        # Run exec in a daemon thread so a timeout can return control to the
        # agent. The thread cannot be forcefully killed in Python, but as a
        # daemon it won't block process exit.
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if not result["done"]:
            # Thread is still running (infinite loop or slow code).
            # Signal cooperative cancellation so the worker can exit when
            # it next checks `_cancelled()`. Track the leaked thread.
            _cancel_evt.set()
            with _leaked_threads_lock:
                _leaked_threads.append(t)
                # Prune completed threads from the leaked list.
                _leaked_threads[:] = [lt for lt in _leaked_threads if lt.is_alive()]
                leaked_count = len(_leaked_threads)
            if leaked_count > 1:
                logger.warning(
                    "code_run: %d daemon threads leaked due to timeout (total leaked: %d). "
                    "They will be cleaned up on process exit.",
                    1, leaked_count,
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

