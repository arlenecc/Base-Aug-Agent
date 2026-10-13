"""code_run tool: execute arbitrary Python in a sandboxed namespace.

Executes with the workspace as cwd. File writes outside the workspace are
blocked by a guarded builtins.open wrapper, while still allowing pip installs
and network access (the whole point of code_run). A timeout prevents infinite
loops from hanging the agent.

**这不是安全边界**：限制只覆盖 ``builtins.open``。``pathlib.Path.write_text``、
``os.open``、``os.remove``、``shutil`` 等路径都能绕过它。真正的隔离需要在
受限子进程里执行。因此 code_run 被标记为 destructive，每次执行都必须经用户
确认（见 AgentCallbacks.confirm）—— 确认弹窗才是这里唯一的实际防线。

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
import shutil
import subprocess
import sys
import tempfile
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

# 单行长度上限：一次 print 一个无换行的超长字符串，父进程侧的 line 同样
# 会撑到那么大（_MAX_STREAM_CHARS 只约束累计量，约束不了单行）。
_MAX_LINE_CHARS = 64_000

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

# code_run 的执行互斥锁：进程内兜底路径下 os.chdir / sys.stdout 替换是进程级
# 全局状态，两个并发在飞的 code_run 会互相覆盖 stdout/cwd（A 的 print 写进 B 的
# StringIO，B 的相对路径被 A 的 cwd 影响）。串行化整个「换状态 → 执行 →
# 恢复」区间；超时泄漏的线程此后只碰自己的 StringIO，不再阻塞后续调用。
_exec_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 受限子进程沙箱
# ---------------------------------------------------------------------------
#
# 进程内执行有两个无法用工程手段修掉的问题：
#   1) 线程杀不掉——超时后死循环的线程会一直存活到进程退出；
#   2) 写路径拦不全——只 patch builtins.open 时，pathlib.Path.write_text、
#      os.open/os.remove/os.rename、shutil.* 都能绕过。
# 子进程里两者都能解决：超时可以 killpg(SIGKILL) 真正杀掉整棵进程树，
# 且补丁可以覆盖到 pathlib/os/shutil 层面。

_SANDBOX_TMP_NAME = "base-agent-code-run"

# 子进程引导脚本。用法：python -c <bootstrap> <workspace_realpath> <sandbox_tmp>
_SANDBOX_BOOTSTRAP = r'''
import sys, os, io, builtins, tempfile

_WS = os.path.realpath(sys.argv[1])
_TMP = os.path.realpath(sys.argv[2])
try:
    os.makedirs(_TMP, exist_ok=True)
    tempfile.tempdir = _TMP
except Exception:
    pass

_code = sys.stdin.read()
try:
    sys.stdin = open(os.devnull, "r")
except Exception:
    pass


def _norm(p):
    if isinstance(p, (bytes, bytearray)):
        p = os.fsdecode(p)
    try:
        return os.path.realpath(os.fspath(p))
    except Exception:
        raise PermissionError("Path is not usable in code_run: %r" % (p,))


def _check(p, op):
    rp = _norm(p)
    if not (
        rp == _WS or rp.startswith(_WS + os.sep)
        or rp == _TMP or rp.startswith(_TMP + os.sep)
    ):
        raise PermissionError("Refusing to %s outside workspace: %s" % (op, p))


# ---- builtins.open / io.open ----
_real_open = builtins.open


def _guarded_open(file, mode="r", *a, **kw):
    # int fd 直接放行：fd 本身已经是打开状态，不构成新的文件访问；且
    # subprocess / os.fdopen 等标准库内部大量走这条路径（拦掉会让
    # subprocess.check_output 这类完全合法的调用失败）。写路径的防护在
    # os.open 上，fd 只能从那里来。
    if isinstance(file, int):
        return _real_open(file, mode, *a, **kw)
    if any(m in str(mode) for m in ("w", "a", "x", "+")):
        _check(file, "write")
    return _real_open(file, mode, *a, **kw)


builtins.open = _guarded_open
io.open = _guarded_open


# ---- os 写类操作 ----
# rename / replace / link / symlink 的写副作用发生在**第二个参数（目标）**上：
# 只校验源路径的话，「在 workspace 内写好内容 → rename 到 ~/Library/...」
# 就能把文件落在外面而完全不触发 PermissionError。
_DST_OPS = ("rename", "replace", "link", "symlink")


def _wrap(_orig, _op):
    def _f(path, *a, **kw):
        _check(path, _op)
        if _op in _DST_OPS:
            dst = kw.get("dst", a[0] if a else None)
            if dst is not None:
                _check(dst, _op + " target")
        return _orig(path, *a, **kw)
    return _f


for _n in ("remove", "unlink", "rmdir", "mkdir", "makedirs", "rename",
           "replace", "truncate", "link", "symlink", "chmod", "utime"):
    _orig_fn = getattr(os, _n, None)
    if _orig_fn is not None:
        setattr(os, _n, _wrap(_orig_fn, _n))

_real_os_open = os.open


def _guarded_os_open(path, flags, *a, **kw):
    # O_WRONLY / O_RDWR 才是写；只带 O_CREAT 的只读打开无害。
    if flags & (os.O_WRONLY | os.O_RDWR):
        _check(path, "open for write")
    return _real_os_open(path, flags, *a, **kw)


os.open = _guarded_os_open


# ---- pathlib ----
try:
    import pathlib
    _P = pathlib.Path
    for _m, _op in (("open", "write"), ("write_text", "write"),
                    ("write_bytes", "write"), ("mkdir", "mkdir"),
                    ("rename", "rename"), ("replace", "replace"),
                    ("unlink", "remove"), ("rmdir", "rmdir"),
                    ("touch", "touch"), ("symlink_to", "symlink")):
        _orig_m = getattr(_P, _m, None)
        if _orig_m is None:
            continue

        def _mk_path(_orig_m=_orig_m, _op=_op, _m=_m):
            def _f(self, *a, **kw):
                if _m != "open":
                    _check(self, _op)
                else:
                    _mode = kw.get("mode", a[0] if a else "r")
                    if any(x in str(_mode) for x in ("w", "a", "x", "+")):
                        _check(self, _op)
                # rename / replace / symlink_to 的目标才是真正落点，必须校验。
                if _m in ("rename", "replace", "symlink_to"):
                    _target = kw.get("target", a[0] if a else None)
                    if _target is None:
                        _target = kw.get("new_name")
                    if _target is not None:
                        _check(_target, _m + " target")
                return _orig_m(self, *a, **kw)
            return _f
        setattr(_P, _m, _mk_path())
except Exception as _e:
    # 补丁失败必须出声：只 patch 上一半的话沙箱会带着残缺的防护继续跑，
    # 而日志里什么都看不到（"确认弹窗是唯一防线"的设计就被悄悄削弱了）。
    sys.stderr.write("code_run 沙箱: pathlib 补丁失败，写路径保护降级: %r\\n" % (_e,))


# ---- shutil ----
try:
    import shutil as _sh
    for _m in ("copy", "copy2", "copyfile", "move", "rmtree", "copytree"):
        _orig_m = getattr(_sh, _m, None)
        if _orig_m is None:
            continue

        def _mk_sh(_orig_m=_orig_m, _m=_m):
            def _f(src, dst, *a, **kw):
                _check(dst, _m)
                if _m in ("move", "rmtree"):
                    _check(src, _m)
                return _orig_m(src, dst, *a, **kw)
            return _f
        setattr(_sh, _m, _mk_sh())
except Exception as _e:
    sys.stderr.write("code_run 沙箱: shutil 补丁失败，写路径保护降级: %r\\n" % (_e,))


# 协作式取消：父进程超时后创建 argv[3] 指向的标志文件，用户代码可用
# _cancelled() 轮询并主动退出（与进程内执行路径的契约保持一致）。
_CANCEL_FILE = sys.argv[3] if len(sys.argv) > 3 else ""


def _cancelled():
    return bool(_CANCEL_FILE) and os.path.exists(_CANCEL_FILE)


_ns = {"__name__": "__code_run__", "__file__": "<code_run>",
       "_cancelled": _cancelled}
exec(compile(_code, "<code_run>", "exec"), _ns)
'''


def _python_executable() -> Optional[str]:
    """返回可用于 ``python -c ...`` 的解释器；找不到返回 None。

    冻结包（PyInstaller）里 ``sys.executable`` 是 GUI 二进制本身——把它传给
    ``subprocess.Popen([..., "-c"])`` 会再启动一个 GUI 应用。这种情况退回
    在 PATH 里找同名版本的 python。
    """
    if not getattr(sys, "frozen", False):
        return sys.executable
    for _name in (
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        f"python{sys.version_info.major}",
        "python3",
        "python",
    ):
        found = shutil.which(_name)
        if found:
            return found
    return None


def _kill_sandbox(proc) -> None:
    """杀掉整个进程组：先 SIGTERM 给清理机会，再 SIGKILL 保证一定死。"""
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=3)
        return
    except Exception:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class _BoundedCollector(io.StringIO):
    """有界 StringIO：超出上限后丢弃后续写入（只标记一次）。"""

    def __init__(self, limit: int = _MAX_STREAM_CHARS):
        super().__init__()
        self._limit = limit
        self.truncated = False

    def write(self, s) -> int:
        if self.truncated:
            return 0
        if self.tell() + len(s) > self._limit:
            super().write("\n… [输出超限，已截断]")
            self.truncated = True
            # 必须返回「实际写入的字符数」。返回 len(s) 是对调用方的
            # 谎报（print/f.write 依赖返回值判断进度），截断场景下会
            # 让上层误以为全部写入成功。
            return 0
        return super().write(s)


def _run_in_subprocess(code: str, timeout: float, ws: str) -> Optional[ToolResult]:
    """在受限子进程里执行 ``code``。

    返回 ``None`` 表示本路径不可用（找不到解释器 / 无法启动），调用方应回退
    到进程内执行。
    """
    py = _python_executable()
    if not py:
        return None

    # 每次执行一个独立的临时目录：固定可预测路径（/tmp/base-agent-code-run）
    # 是多用户/多实例共享的，存在被预置 symlink 劫持的风险。
    try:
        sandbox_tmp = tempfile.mkdtemp(prefix=_SANDBOX_TMP_NAME + "-")
    except OSError as e:
        logger.warning("code_run: 无法创建沙箱临时目录 (%s)，回退进程内执行", e)
        return None
    cancel_file = os.path.join(sandbox_tmp, "cancel.flag")

    env = os.environ.copy()
    # 子进程（及其再 fork 的 pip/构建脚本）的临时文件也落在沙箱 tmp 内，
    # 与引导脚本里的写路径放行规则一致。
    env["TMPDIR"] = sandbox_tmp
    env["TMP"] = sandbox_tmp
    env["TEMP"] = sandbox_tmp
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            [py, "-c", _SANDBOX_BOOTSTRAP, ws, sandbox_tmp, cancel_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ws,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
    except Exception as e:
        logger.warning("code_run: 子进程沙箱不可用 (%s)，回退进程内执行", e)
        shutil.rmtree(sandbox_tmp, ignore_errors=True)
        return None

    def _cleanup():
        """删除本次执行的临时目录（用户代码可能往 tempfile 里写了大量数据）。"""
        shutil.rmtree(sandbox_tmp, ignore_errors=True)

    def _feed_stdin():
        try:
            proc.stdin.write(code)
            proc.stdin.close()
        except Exception:
            pass

    feeder = threading.Thread(target=_feed_stdin, daemon=True)
    feeder.start()

    out = _BoundedCollector()
    err = _BoundedCollector()

    def _pump(stream, sink):
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                # 单行长度也要封顶：一次 print 一个 500MB 的无换行字符串，
                # 父进程侧的 line 同样是 500MB。
                if len(line) > _MAX_LINE_CHARS:
                    line = line[:_MAX_LINE_CHARS] + "…[单行过长已截断]\n"
                # 截断后仍继续读，避免子进程写满管道卡死。
                if not sink.truncated:
                    sink.write(line)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    readers = [
        threading.Thread(target=_pump, args=(proc.stdout, out), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, err), daemon=True),
    ]
    for t in readers:
        t.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # 先立取消标志，让协作式脚本在 SIGTERM 宽限期内自己退出。
        try:
            open(cancel_file, "w").close()
        except OSError:
            pass
        _kill_sandbox(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    except BaseException:
        _kill_sandbox(proc)
        raise
    finally:
        feeder.join(timeout=2.0)

    # 子进程已退出，但它 fork 出的孙进程（后台 dev server 之类）可能仍持有
    # 管道写端 → readline() 永不返回 b"" → pump 线程永久挂起并泄漏两个 fd。
    # 先给一个短宽限让它们把缓冲排干，再**从父进程侧关闭管道**强制唤醒。
    for t in readers:
        t.join(timeout=0.5)
    for t, stream in zip(readers, (proc.stdout, proc.stderr)):
        if t.is_alive():
            try:
                stream.close()
            except Exception:
                pass
    for t in readers:
        t.join(timeout=0.5)

    if timed_out:
        partial = out.getvalue()
        if err.getvalue():
            partial += ("\n" if partial else "") + "[stderr]\n" + err.getvalue()
        _cleanup()
        return ToolResult(
            False,
            output=partial,
            error=(
                f"Code execution timed out after {timeout:.0f}s"
                + ("（以下为超时前已产生的输出）" if partial else "")
            ),
        )

    rc = proc.returncode
    out_s = out.getvalue()
    err_s = err.getvalue()
    if rc == 0:
        output = out_s
        if err_s:
            output += ("\n" if output else "") + "[stderr]\n" + err_s
        _cleanup()
        return ToolResult(True, output=output or "(no output)")
    # 非零退出：stderr 里就是 traceback（或 SystemExit 提示）。
    msg = err_s.strip() or f"Process exited with code {rc}"
    if out_s.strip():
        msg = out_s.rstrip() + "\n" + msg
    _cleanup()
    return ToolResult(False, error=msg, output=out_s)


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

        # 优先走受限子进程：能真正 kill 超时代码，且写路径防护覆盖
        # pathlib/os/shutil（进程内路径拦不全、线程杀不掉）。
        sandboxed = _run_in_subprocess(code, timeout, ws)
        if sandboxed is not None:
            return sandboxed
        return self._run_in_process(code, timeout)

    def _run_in_process(self, code: str, timeout: float) -> ToolResult:
        """进程内兜底路径（找不到可用解释器时）。

        语义与子进程路径一致，但有两个已知短板：超时线程杀不掉、写路径只拦
        builtins.open。仅当子进程不可用时启用。
        """
        import builtins as _b

        ws = os.path.realpath(self.config.workspace)
        _real_open = _b.open

        def _guard_open(file, mode="r", *a, **kw):
            if isinstance(file, int):
                raise PermissionError("File descriptor access is not allowed in code_run")
            # bytes 路径完全合法（open(b"report.txt", "w")），旧实现走 else ""
            # 分支 → realpath("") = 进程 cwd → 对 workspace 内的合法写误判为
            # 「workspace 外部」而误报 PermissionError。
            if isinstance(file, bytes):
                file = os.fsdecode(file)
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
                    # 必须返回「实际写入的字符数」。返回 len(s) 是对调用方的
                    # 谎报（print/f.write 依赖返回值判断进度），截断场景下会
                    # 让上层误以为全部写入成功。
                    return 0
                return super().write(s)

        out = _BoundedIO()
        err = _BoundedIO()

        # 主线程侧的进程级状态快照（cwd / stdout / stderr），join 后恢复。
        # getcwd 在进程 cwd 已被删除时抛 FileNotFoundError——捕获并回退到
        # workspace（恢复时 chdir 失败也会安全地告警，不会中断流程）。
        try:
            _prev_cwd = os.getcwd()
        except OSError:
            _prev_cwd = ws
        _prev_stdout, _prev_stderr = sys.stdout, sys.stderr

        # 串行化整个「换全局状态 → 执行 → 恢复」区间：os.chdir/sys.stdout 是
        # 进程级状态，两个并发在飞的 code_run 会互相覆盖（见模块级 _exec_lock
        # 注释）。锁的粒度覆盖 join（含超时等待）——这是有意为之。
        # 拿到锁之后到「恢复 + release」之间的任何异常都不能把锁留在手里：
        # 否则此后**所有**兜底 code_run 都会永久挂死在 acquire() 上。
        _exec_lock.acquire()

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
        try:
            t = threading.Thread(target=_worker, daemon=True)
            # start() 可能抛 RuntimeError（线程数耗尽）：此时必须放锁，否则
            # 后续所有 code_run 都会永久卡在 acquire() 上。
            t.start()
        except BaseException:
            _exec_lock.release()
            raise
        t.join(timeout=timeout)

        # 无论是否超时都由主线程统一恢复进程级全局状态。恢复权**只**属于
        # 主线程：worker 不做任何恢复（见 _worker 注释），因此超时泄漏的
        # 线程无论何时退出都不会再污染 cwd / sys.stdout。
        try:
            os.chdir(_prev_cwd)
            sys.stdout, sys.stderr = _prev_stdout, _prev_stderr
        except OSError as e:
            logger.warning("code_run: failed to restore process cwd/stdout: %s", e)
        finally:
            # 全局状态已恢复（含超时路径——泄漏线程此后只碰自己的 StringIO），
            # 释放执行锁，允许下一个 code_run 进入。
            _exec_lock.release()

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
            # 超时前已经产生的输出必须回传：否则模型看不到「卡在哪一步」，
            # 只能瞎猜重试（比如一个跑到 90% 才卡住的循环会完全丢失线索）。
            partial = out.getvalue()
            if err.getvalue():
                partial += ("\n" if partial else "") + "[stderr]\n" + err.getvalue()
            return ToolResult(
                False,
                output=partial,
                error=(
                    f"Code execution timed out after {timeout:.0f}s"
                    + ("（以下为超时前已产生的输出）" if partial else "")
                ),
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

