"""code_run tool: execute arbitrary Python in a sandboxed namespace.

Executes with the workspace as cwd. File writes outside the workspace are
blocked by a guarded builtins.open wrapper, while still allowing pip installs
and network access (the whole point of code_run). A timeout prevents infinite
loops from hanging the agent.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult


# Default timeout (seconds). Long enough for pip installs and network calls,
# short enough that a stuck script doesn't freeze the agent.
_DEFAULT_TIMEOUT = 30


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
            path = os.path.abspath(file if isinstance(file, (str, os.PathLike)) else "")
            if any(m in str(mode) for m in ("w", "a", "x", "+")):
                if not (path == ws or path.startswith(ws + os.sep)):
                    raise PermissionError(f"Refusing to write outside workspace: {file}")
            return _real_open(file, mode, *a, **kw)

        namespace: dict = {"__name__": "__code_run__", "__file__": "<code_run>"}
        out = io.StringIO()
        err = io.StringIO()

        # Result holder for the worker thread. Using a list so the closure can
        # mutate it without `nonlocal` (which complicates the daemon pattern).
        result = {"exc": None, "done": False}

        def _worker():
            cwd = os.getcwd()
            try:
                os.chdir(ws)
                namespace["__builtins__"] = {**vars(_b), "open": _guard_open}
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
        # daemon it won't block process exit. For an infinite loop this means
        # the thread lingers in the background -- acceptable for a local tool.
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if not result["done"]:
            # Thread is still running (infinite loop or slow code).
            return ToolResult(
                False,
                error=f"Code execution timed out after {timeout:.0f}s",
            )

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

