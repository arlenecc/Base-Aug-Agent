"""MCP (Model Context Protocol) JSON-RPC client over stdio.

Manages a subprocess that runs an MCP server, communicates via JSON-RPC 2.0
over stdin/stdout, and exposes tools/list and tools/call.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# tools/call 的下限超时（秒）。握手/枚举用实例的 timeout（默认 30s）足够，
# 但工具调用本身可能是构建、批量处理、爬取等长任务——共用 30s 会让长任务
# 必然超时，且服务端还在跑，结果被当迟到响应丢弃。
_TOOL_CALL_TIMEOUT = 300.0

# 单次 MCP 工具返回保留的最大字符数。MCP 工具是外部代码，输出体积不受我们
# 控制（例如让它 dump 一个大文件/数据库表），而这段文本会原样进入模型上下文
# ——不设上限一次调用就能把窗口撑爆。与 file_read / shell_run 的 200K 策略
# 保持一致，保留头部并明确标注截断。
_MAX_TOOL_OUTPUT_CHARS = 200_000

# ------------------------------------------------------------------
# MCP JSON-RPC wire types
# ------------------------------------------------------------------


@dataclass
class JSONRPCRequest:
    jsonrpc: str = "2.0"
    id: int = 0
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JSONRPCResponse:
    jsonrpc: str = "2.0"
    id: int = 0
    result: Any = None
    error: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------------
# MCP Client
# ------------------------------------------------------------------


class MCPClient:
    """Manages one MCP server subprocess and JSON-RPC communication."""

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()
        # Separate lock for stdin writes. _lock guards _pending/_results/_request_id
        # but is released while waiting on the response event — without a dedicated
        # write lock, two concurrent _call()s could interleave bytes on the stdin
        # pipe, corrupting JSON-RPC messages (>PIPE_BUF splits).
        self._write_lock = threading.Lock()
        self._pending: Dict[int, threading.Event] = {}
        self._results: Dict[int, JSONRPCResponse] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._running = False
        self._server_info: Dict[str, Any] = {}
        self._tools: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the MCP server subprocess and perform the initialize handshake."""
        if self._running:
            return

        env = None
        if self.env:
            import os
            env = os.environ.copy()
            env.update(self.env)

        try:
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
                # 独立进程组：很多 MCP server 由 `npx` 启动，npx 会再 fork 出
                # 真正的 node server。不建新会话的话 terminate() 只杀掉 npx，
                # 真正的 server 会变成孤儿进程一直存活到关机。有了独立进程组，
                # stop() 可以 killpg 一次性回收整棵树。
                start_new_session=True,
            )
        except FileNotFoundError:
            raise MCPError(f"MCP server '{self.name}': command not found: {self.command}")
        except Exception as e:
            raise MCPError(f"MCP server '{self.name}': failed to start: {e}")

        self._running = True
        # Drain stderr in a background thread so the pipe buffer never fills up
        # and blocks the subprocess.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        # Initialize handshake
        try:
            init_resp = self._call("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "base-agent", "version": "1.0"},
            })
            self._server_info = init_resp.result or {}
            # Send initialized notification
            self._send_notification("notifications/initialized", {})
            # Fetch tools
            tools_resp = self._call("tools/list", {})
            self._tools = tools_resp.result.get("tools", []) if tools_resp.result else []
            logger.info(
                "MCP server '%s' initialized: %d tools available",
                self.name, len(self._tools),
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Terminate the MCP server subprocess and wait for threads.

        Cleanup is driven by ``self._process``, NOT by ``self._running``.
        The reader thread sets ``_running = False`` in its ``finally`` block
        whenever the server dies on its own (crash, stdout closed).  If we
        bailed out on ``not self._running``, that path would skip
        terminate()/wait() entirely and leave a zombie child forever — and a
        still-alive server (only its stdout closed) would become an orphan.
        """
        # Signal the reader thread to stop first. We set _running=False under
        # the lock so the reader sees it on its next loop iteration.
        with self._lock:
            self._running = False
            process = self._process
            self._process = None
        # Close stdin / terminate the process OUTSIDE the lock so the reader
        # thread (which may be in its finally block trying to acquire the lock)
        # doesn't deadlock waiting for us to release it.
        if process:
            for pipe in (process.stdin, process.stdout, process.stderr):
                try:
                    if pipe:
                        pipe.close()
                except Exception:
                    pass
            try:
                # 整棵进程组一起收：npx 之类的包装进程会把真正的 server 挂成
                # 孙进程，只 terminate 顶层 PID 会留下孤儿。
                try:
                    import os
                    import signal
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception:
                    process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    import os
                    import signal
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:
                    process.kill()
                # SIGKILL 后仍设超时：进程处于不可中断状态（D，常见于 NFS/
                # FUSE I/O）时 wait() 会永久挂起，进而卡死 shutdown/重建。
                # 这种情况下只能放弃等待，让 init 在进程退出后收尸。
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "MCP server '%s': pid %d unkillable (D state?), giving up wait",
                        self.name, process.pid,
                    )
            except Exception:
                pass
        # Clear pending requests under the lock.
        with self._lock:
            self._process = None
            # Wake up any waiters
            for ev in self._pending.values():
                ev.set()
            self._pending.clear()
            self._results.clear()
        # Wait for reader threads to finish (they should exit once _running is
        # False and stdin is closed, causing readline() to return "").
        for t in (self._reader_thread, self._stderr_thread):
            if t and t.is_alive():
                t.join(timeout=3)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the list of tools exposed by this MCP server."""
        return list(self._tools)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server and return the text content."""
        # tools/call 与 initialize/tools/list 不能共用同一个 30s 超时：构建、
        # 批量处理、爬取这类工具跑几分钟是常态。旧实现让它们共用初始化超时，
        # 长任务必然超时失败——而服务端还在继续跑，结果被当迟到响应丢弃。
        resp = self._call(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=max(self.timeout, _TOOL_CALL_TIMEOUT),
        )
        if resp.error:
            raise MCPError(
                f"MCP tool '{name}' error: {resp.error.get('message', 'unknown')}"
            )
        result = resp.result
        if not isinstance(result, dict):
            return str(result)
        # MCP tools return content as a list of content blocks
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "resource":
                        texts.append(f"[resource: {block.get('resource', {})}]")
                    else:
                        texts.append(json.dumps(block, ensure_ascii=False))
                else:
                    texts.append(str(block))
            return _cap_output("\n".join(texts))
        return _cap_output(json.dumps(result, ensure_ascii=False))

    # ------------------------------------------------------------------
    # JSON-RPC internals
    # ------------------------------------------------------------------

    def _call(
        self,
        method: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> JSONRPCResponse:
        """Send a JSON-RPC request and wait for the response.

        ``timeout`` 缺省用初始化超时（适合握手/枚举类调用）；长任务方法应
        传入更长的值（见 call_tool）。
        """
        with self._lock:
            req_id = self._request_id
            self._request_id += 1
            ev = threading.Event()
            self._pending[req_id] = ev

        req = JSONRPCRequest(id=req_id, method=method, params=params or {})
        try:
            self._send(json.dumps(req.__dict__))
        except MCPError:
            with self._lock:
                self._pending.pop(req_id, None)
            raise

        if not ev.wait(timeout=timeout if timeout is not None else self.timeout):
            with self._lock:
                self._pending.pop(req_id, None)
            raise MCPError(f"MCP server '{self.name}': timeout waiting for '{method}'")

        with self._lock:
            resp = self._results.pop(req_id, None)
            self._pending.pop(req_id, None)

        if resp is None:
            raise MCPError(f"MCP server '{self.name}': no response for '{method}'")
        if resp.error:
            raise MCPError(
                f"MCP server '{self.name}' RPC error: {resp.error.get('message', 'unknown')}"
            )
        return resp

    def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._send(json.dumps(msg))

    def _send(self, line: str) -> None:
        """Write a single JSON-RPC message line to the subprocess stdin.

        Guarded by _write_lock so concurrent _call() invocations cannot
        interleave bytes on the pipe (writes >PIPE_BUF are not atomic).

        进程引用在锁内读取：stop() 会在锁内置 ``_process = None``，旧实现
        在锁外判空、锁内解引用，与 stop() 竞态时抛 AttributeError 而不是
        清晰的 MCPError。
        """
        with self._write_lock:
            process = self._process
            if not process or not process.stdin:
                raise MCPError(f"MCP server '{self.name}': not running")
            try:
                process.stdin.write(line + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:
                raise MCPError(f"MCP server '{self.name}': write failed: {e}")

    def _drain_stderr(self) -> None:
        """Continuously read stderr to prevent buffer deadlock."""
        if not self._process or not self._process.stderr:
            return
        try:
            for line in self._process.stderr:
                logger.debug("MCP '%s' stderr: %s", self.name, line.rstrip())
        except Exception:
            pass

    def _read_loop(self) -> None:
        """Continuously read JSON-RPC responses from the subprocess stdout."""
        try:
            while self._running and self._process and self._process.stdout:
                line = self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("MCP server '%s' non-JSON line: %s", self.name, line[:200])
                    continue

                # Skip notifications (no id)
                if "id" not in obj:
                    continue

                resp = JSONRPCResponse(
                    jsonrpc=obj.get("jsonrpc", "2.0"),
                    id=obj["id"],
                    result=obj.get("result"),
                    error=obj.get("error"),
                )
                with self._lock:
                    ev = self._pending.get(resp.id)
                    if ev is None:
                        # 迟到的响应：对应的 _call() 已超时/被取消，没人会再来
                        # pop 它。旧实现无条件写入 _results，这些条目会永久
                        # 驻留（内存泄漏）。直接丢弃。
                        continue
                    self._results[resp.id] = resp
                ev.set()
        except Exception as e:
            logger.debug("MCP server '%s' reader exited: %s", self.name, e)
        finally:
            # The reader is exiting (server crashed, stdin closed, or stop()).
            # Wake up any pending callers so they don't wait the full timeout.
            # Store an error response so _call() reports a clear message instead
            # of a generic "no response".
            with self._lock:
                self._running = False
                for req_id, ev in self._pending.items():
                    if req_id not in self._results:
                        self._results[req_id] = JSONRPCResponse(
                            id=req_id,
                            error={"message": "MCP server connection closed", "code": -1},
                        )
                    ev.set()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _cap_output(text: str) -> str:
    """把 MCP 工具输出钳制到 _MAX_TOOL_OUTPUT_CHARS，超长部分截断并标注。

    截断标记必须让模型知道「结果被裁了」，否则它会把半截输出当成完整结果
    继续推理。
    """
    if len(text) <= _MAX_TOOL_OUTPUT_CHARS:
        return text
    return (
        text[:_MAX_TOOL_OUTPUT_CHARS]
        + f"\n… [MCP 工具输出共 {len(text)} 字符，已截断到 {_MAX_TOOL_OUTPUT_CHARS}]"
    )


# ------------------------------------------------------------------
# Error type
# ------------------------------------------------------------------


class MCPError(Exception):
    """Raised when MCP communication fails."""
    pass