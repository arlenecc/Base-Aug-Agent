"""code_run tool: execute arbitrary Python in a sandboxed namespace.

Executes with the workspace as cwd. File writes outside the workspace are
blocked by a guarded builtins.open wrapper, while still allowing pip installs
and network access (the whole point of code_run). A timeout prevents infinite
loops from hanging the agent.

写入护栏只是**提示性的**，不是沙箱：它只替换了命名空间里的 ``open``，
``io.open`` / ``pathlib.Path.write_text`` / ``os.remove`` / ``shutil`` /
``subprocess`` 都不受约束，``__import__`` 也可绕过。真正的隔离只能靠
容器或独立进程。

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
        "This is the capability core: anything not covered by a dedicated tool can "
        "be done here. For system commands / CLI tools prefer shell_run "
        "(or call it from here via subprocess). "
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

    def _cancel_checker(self):
        """协作式取消探针（UI「终止对话」）。拿不到 callbacks 时恒 False。"""
        cb = getattr(self.registry, "callbacks", None)
        fn = getattr(cb, "is_cancelled", None)
        return fn if callable(fn) else (lambda: False)

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
        # 轮询式 join：旧实现一次性 join(timeout) 在用户点「终止对话」后仍要
        # 跑满整个 timeout 才返回（harness 虽立刻合成失败结果，但这个线程
        # 继续占 CPU、且 sys.stdout 仍被替换成 StringIO 直到 join 返回）。
        # 0.2s 粒度轮询取消标志，命中立即收尾。
        deadline = time.monotonic() + timeout
        cancelled = False
        while t.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            t.join(timeout=min(0.2, remaining))
            if not t.is_alive():
                break
            if self._cancel_checker()():
                cancelled = True
                break

        # 无论是否超时都由主线程统一恢复进程级全局状态。恢复权**只**属于
        # 主线程：worker 不做任何恢复（见 _worker 注释），因此超时泄漏的
        # 线程无论何时退出都不会再污染 cwd / sys.stdout。
        try:
            os.chdir(_prev_cwd)
            sys.stdout, sys.stderr = _prev_stdout, _prev_stderr
        except OSError as e:
            logger.warning("code_run: failed to restore process cwd/stdout: %s", e)

        def _partial_output() -> str:
            partial = ""
            try:
                partial = (out.getvalue() or "") + (
                    ("\n[stderr]\n" + err.getvalue()) if err.getvalue() else "")
            except Exception:
                partial = ""
            return partial

        if cancelled:
            # 与超时同法：先读出已产出的内容，再关缓冲区止住 stdout 污染
            # （getvalue 在 close 之后会抛 ValueError，顺序不能反）。
            partial = _partial_output()
            _cancel_evt.set()
            for buf in (out, err):
                try:
                    buf.close()
                except Exception:
                    pass
            msg = "用户取消了对话，代码执行被中断"
            if partial.strip():
                msg += f"\n[取消前已产生的输出]\n{partial[-3000:]}"
            return ToolResult(False, error=msg)

        if not result["done"]:
            # Thread is still running (infinite loop or slow code).
            # Signal cooperative cancellation so the worker can exit when
            # it next checks `_cancelled()`. Track the leaked thread.
            _cancel_evt.set()
            #
            # 泄漏线程随后 print() 时会**动态**查找 sys.stdout，而上面主线程
            # 已经把它恢复成真实 stdout —— 不处理的话它的输出会一直打到进程
            # 真实 stdout（不受 _BoundedIO 截断、也不会进工具结果）。
            #
            # 这里把缓冲区读出来再 close()：泄漏线程的下一次写入会抛
            # ValueError: I/O operation on closed file，从而自行退出（既止住
            # stdout 污染，也顺带回收了线程与其命名空间）。
            partial = ""
            try:
                partial = (out.getvalue() or "") + (
                    ("\n[stderr]\n" + err.getvalue()) if err.getvalue() else "")
            except Exception:
                partial = ""
            for buf in (out, err):
                try:
                    buf.close()
                except Exception:
                    pass
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
            msg = f"Code execution timed out after {timeout:.0f}s"
            if partial.strip():
                tail = partial[-3000:]
                msg += f"\n[超时前已产生的输出]\n{tail}"
            return ToolResult(False, error=msg)

        # Worker completed within timeout — check if the thread we just
        # joined is still in the leaked list and remove it.
        with _leaked_threads_lock:
            if t in _leaked_threads:
                _leaked_threads.remove(t)

        exc = result["exc"]
        if exc is not None:
            # 统一带 traceback：PermissionError（写护栏拦截）/SystemExit 旧实现
            # 只给裸字符串，模型看不到出错代码行，与其他异常类型不一致、难调试。
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            detail = "".join(tb)
            if isinstance(exc, (PermissionError, SystemExit)):
                # 护栏/退出类异常消息本身就是结论，放最前面再附栈。
                return ToolResult(False, error=f"{exc}\n{detail}")
            return ToolResult(False, error=detail)

        output = out.getvalue()
        if err.getvalue():
            output += ("\n" if output else "") + "[stderr]\n" + err.getvalue()
        return ToolResult(True, output=output or "(no output)")

