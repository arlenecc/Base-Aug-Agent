"""shell_run tool: execute shell/terminal commands in the workspace.

Runs the command with the workspace as cwd, captures stdout+stderr, and
enforces a timeout so a hung command cannot block the agent indefinitely.

确认策略：普通命令（ls、git、pip、构建脚本等）直接执行；仅当命令可能
破坏系统、写入重要系统/配置文件或影响系统稳定性/安全性时（见
_is_risky_command）才需要用户确认。
"""
from __future__ import annotations

import collections
import logging
import os
import re
import signal
import subprocess
import threading
import time
from typing import Any, Dict

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Default timeout (seconds) for shell commands. Can be overridden per-call via
# the `timeout` argument. Long enough for pip installs, short enough that a
# hung command doesn't freeze the agent.
_DEFAULT_TIMEOUT = 120
# 上下限钳制：模型可能传 99999（agent 线程会阻塞 27 小时且无法被「终止对话」
# 打断，因为协作式取消只在工具边界轮询），也可能传 0/负数（立刻判定超时并
# kill 掉刚启动的命令）。与 code_run 的 _MIN/_MAX_TIMEOUT 同策略。
_MIN_TIMEOUT = 1.0
_MAX_TIMEOUT = 600.0

# 单条命令 stdout/stderr 各保留的最大字符数。旧实现用 capture_output 全量
# 读入内存——一条 `cat 大日志` / `find /` 就能把进程内存打满。
_MAX_OUTPUT_CHARS = 200_000

# 超时后给进程组留出优雅退出的时间，之后再 SIGKILL。
_TERM_GRACE_SECONDS = 3.0


# ---------------------------------------------------------------------------
# 风险评估：仅当命令可能破坏系统、写入重要系统/配置文件或影响系统稳定性/
# 安全性时才需要用户确认；其余（ls、git、pip、构建脚本、工作区内 rm 等）
# 直接执行。宁可对边界场景多确认一次，也不让普通命令反复打断用户。
# ---------------------------------------------------------------------------

# 命令段首词命中即确认：磁盘/引导/电源/内核/系统服务操作。
_RISKY_HEADS = frozenset((
    "mkfs", "fdisk", "gdisk", "diskutil",
    "shutdown", "reboot", "halt", "poweroff",
    "csrutil", "nvram", "kextload", "kextunload", "pmset", "launchctl",
))

# 系统目录前缀：凡「写入」这些位置（重定向 / tee / sed -i / rm 目标）即确认。
_SYSTEM_DIRS = (
    "/etc", "/usr", "/bin", "/sbin", "/System", "/Library",
    "/private/etc", "/private/var", "/boot", "/opt", "/dev",
)
_SYS_DIR_ALT = "(" + "|".join(re.escape(d) for d in _SYSTEM_DIRS) + r")\b"

# 提权：sudo 后面的子命令可改任意系统状态，一律确认。
_RE_SUDO = re.compile(r"(?:^|[;&|]\s*|\bthen\s+|\bdo\s+)\bsudo\b")
# 远程代码执行：下载内容直接管道给 shell 解释。
_RE_PIPE_SH = re.compile(r"\b(?:curl|wget)\b[^|;&]*\|\s*(?:sudo\s+)?(?:ba|z|fi)?sh\b")
# 写系统路径的两种形态：> / >> 重定向、tee 目标（sed -i 的场景较少见且
# 通常伴随 tee/sudo，这里保持规则简单）。
_RE_WRITE_SYS = re.compile(r"(?:>{1,2}\s*|\btee\s+(?:-\w+\s+)*)" + _SYS_DIR_ALT)


def _rm_targets_outside(seg: str) -> bool:
    """rm 的目标是否指向工作区之外（绝对路径 / home / 父目录逃逸）。

    shell_run 的 cwd 是 workspace，纯相对路径（含工作区内 rm -rf build）
    只影响工作区文件，不确认；绝对路径、~/$HOME、../ 一律确认。
    """
    for tok in seg.split():
        if tok.startswith("-"):
            continue
        if tok == "~" or tok.startswith(("/", "~/", "$HOME", "../", "..\\")):
            return True
    return False


def _is_risky_command(command: str) -> bool:
    """判定一条 shell 命令是否需要用户确认（True = 弹确认框）。"""
    if not command or not command.strip():
        return False
    text = command.strip()

    if _RE_PIPE_SH.search(text):
        return True
    if _RE_WRITE_SYS.search(text):
        return True
    if _RE_SUDO.search(text):
        return True

    # 按命令段（| ; & 分隔）逐段检查首词与 rm 的目标。
    for seg in re.split(r"[|;&]+", text):
        words = seg.strip().split()
        if not words:
            continue
        head = words[0]
        if head in _RISKY_HEADS:
            return True
        if head == "rm" and _rm_targets_outside(seg.strip()):
            return True
    return False


class ShellRunTool(Tool):
    name = "shell_run"
    description = (
        "Run a shell command in the workspace, return stdout+stderr. "
        "For CLI tools, git, pip, build scripts, etc. "
        "Destructive commands (system files, sudo, disk ops) ask for confirmation first."
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

    def should_confirm(self, args: Dict[str, Any]) -> bool:
        command = args.get("command")
        if not isinstance(command, str):
            # 参数异常（缺 command / 非字符串）时保守确认。
            return True
        return _is_risky_command(command)

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
        # 钳制到 [_MIN_TIMEOUT, _MAX_TIMEOUT]：过大阻塞 agent 线程无法取消，
        # 过小（0/负数）会立刻 kill 掉刚启动的命令。
        timeout = max(_MIN_TIMEOUT, min(timeout, _MAX_TIMEOUT))

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
        # stdin=DEVNULL：子进程默认继承 agent 的 stdin——模型执行 cat / read /
        # python REPL / ssh 这类读 stdin 的命令会挂到超时（且可能真实消费宿主
        # 输入）。没有交互式输入是这个工具的正确语义。
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # 显式 utf-8 + errors="replace"：text=True 默认按 locale 编码且
        # errors="strict"，命令输出含一个非法字节（curl 抓到的二进制、损坏
        # 文件名）就会让 _reader 线程的 read 抛 UnicodeDecodeError，被 except
        # 吞掉后**该流剩余输出全部静默丢失**，模型看到的是截断且无标注的结果。
        encoding="utf-8",
        errors="replace",
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
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc)
            try:
                proc.wait(timeout=_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:  # pragma: no cover - 极端情况
                pass
    except BaseException:
        # Popen 成功到 wait() 之间出现的任何异常（含 KeyboardInterrupt）都不
        # 能让子进程变成孤儿：这里必须把整个进程组杀掉再往上抛。
        _kill_process_group(proc)
        raise

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

    returncode = proc.returncode
    if returncode is None:
        # SIGKILL 之后仍然 wait 不到（极端情况下内核尚未回收）→ 不能再等，
        # 否则子进程一直不被 reap 变成僵尸。按被强杀处理并记日志。
        logger.warning(
            "shell_run: process still unreaped after kill — treating as SIGKILL (-9)"
        )
        returncode = -9
    return returncode, out.value(), err.value(), timed_out
