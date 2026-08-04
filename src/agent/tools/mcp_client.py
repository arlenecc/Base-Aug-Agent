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
        self._pending: Dict[int, threading.Event] = {}
        self._results: Dict[int, JSONRPCResponse] = {}
        self._reader_thread: Optional[threading.Thread] = None
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
        """Terminate the MCP server subprocess and wait for threads."""
        with self._lock:
            self._running = False
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except Exception:
                pass
            self._process = None
        # Wake up any waiters
        with self._lock:
            for ev in self._pending.values():
                ev.set()
            self._pending.clear()
            self._results.clear()
        # Wait for reader threads to finish
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
        resp = self._call("tools/call", {"name": name, "arguments": arguments})
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
            return "\n".join(texts)
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # JSON-RPC internals
    # ------------------------------------------------------------------

    def _call(self, method: str, params: Dict[str, Any]) -> JSONRPCResponse:
        """Send a JSON-RPC request and wait for the response."""
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

        if not ev.wait(timeout=self.timeout):
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
        """Write a single JSON-RPC message line to the subprocess stdin."""
        if not self._process or not self._process.stdin:
            raise MCPError(f"MCP server '{self.name}': not running")
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
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
                    self._results[resp.id] = resp
                    ev = self._pending.get(resp.id)
                if ev:
                    ev.set()
        except Exception as e:
            logger.debug("MCP server '%s' reader exited: %s", self.name, e)
        finally:
            self._running = False


# ------------------------------------------------------------------
# Error type
# ------------------------------------------------------------------


class MCPError(Exception):
    """Raised when MCP communication fails."""
    pass