"""File operation tools: read, write, modify."""
from __future__ import annotations

import os
import tempfile

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult

# file_read 单次返回的最大字符数。模型把一次 file_read 的输出原样放进上下文，
# 没有上限的话一个几 GB 的日志/数据文件就能撑爆内存和模型窗口。与 shell_run
# (200K)、web (8MB) 的上限策略保持一致。
_MAX_READ_CHARS = 200_000


def _atomic_write(full: str, content: str, encoding: str = "utf-8") -> None:
    """Temp + os.replace 原子写。

    临时文件名必须唯一：固定 ``full + ".tmp"`` 在两个线程并发写同一目标时会
    互相覆盖、落盘混合内容；写失败时固定名还会把 .tmp 残留在 workspace 里被
    后续读取当成数据。用 mkstemp 生成唯一名，finally 里清理。
    """
    dir_name = os.path.dirname(full) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp, full)
    finally:
        # os.replace 成功后 tmp 已不存在；失败时清理掉半成品。
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


class _FileTool(Tool):
    config: AgentConfig
    registry: ToolRegistry

    def bind(self, config: AgentConfig, registry: ToolRegistry) -> None:
        self.config = config
        self.registry = registry

    def _resolve(self, path: str) -> str:
        # Use realpath (not abspath) so symlinks pointing outside the workspace
        # are resolved and rejected — abspath leaves symlinks unresolved, which
        # means a symlink inside the workspace to /etc would pass the prefix
        # check and allow escaping the workspace.
        ws = os.path.realpath(self.config.workspace)
        full = path if os.path.isabs(path) else os.path.join(ws, path)
        full = os.path.realpath(full)
        if not (full == ws or full.startswith(ws + os.sep)):
            raise PermissionError(f"Path '{path}' is outside the workspace")
        return full


class FileReadTool(_FileTool):
    name = "file_read"
    description = "Read the textual contents of a file inside the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within the workspace (or absolute inside it)."},
        },
        "required": ["path"],
    }

    def run(self, path: str) -> ToolResult:
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not os.path.isfile(full):
            return ToolResult(False, error=f"File does not exist: {path}")
        try:
            size = os.path.getsize(full)
            truncated = size > _MAX_READ_CHARS
            with open(full, "r", encoding="utf-8") as f:
                # 只读需要的部分，而不是把整个文件读进内存再截断。
                data = f.read(_MAX_READ_CHARS) if truncated else f.read()
            if truncated:
                data += f"\n… [文件共 {size} 字节，仅显示前 {_MAX_READ_CHARS} 字符]"
            return ToolResult(True, output=data)
        except UnicodeDecodeError:
            import base64

            with open(full, "rb") as f:
                # 二进制同样限制体积：base64 会再放大 4/3，不截断的话一个
                # 大二进制文件就能把内存和模型上下文一起撑爆。
                raw = f.read(_MAX_READ_CHARS // 2)
            return ToolResult(True, output="(binary) " + base64.b64encode(raw).decode())


class FileWriteTool(_FileTool):
    name = "file_write"
    description = "Write text content to a file inside the workspace, creating parent directories."
    # Confined to the workspace by _resolve(); cannot touch system/other files.
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> ToolResult:
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        try:
            _atomic_write(full, content)
        except OSError as e:
            return ToolResult(False, error=f"Write failed: {e}")
        return ToolResult(True, output=f"Wrote {len(content)} chars to {path}")


class FileModifyTool(_FileTool):
    name = "file_modify"
    description = "Replace the first occurrence of `old` with `new` in a workspace file."
    # Confined to the workspace by _resolve(); cannot touch system/other files.
    destructive = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)."},
        },
        "required": ["path", "old", "new"],
    }

    def run(self, path: str, old: str, new: str, replace_all: bool = False) -> ToolResult:
        try:
            full = self._resolve(path)
        except PermissionError as e:
            return ToolResult(False, error=str(e))
        if not os.path.isfile(full):
            return ToolResult(False, error=f"File does not exist: {path}")
        with open(full, "r", encoding="utf-8") as f:
            text = f.read()
        if old not in text:
            return ToolResult(False, error=f"'old' string not found in {path}")
        if replace_all:
            text = text.replace(old, new)
        else:
            text = text.replace(old, new, 1)
        # Atomic write: write to temp then os.replace.
        try:
            _atomic_write(full, text)
        except OSError as e:
            return ToolResult(False, error=f"Modify failed: {e}")
        return ToolResult(True, output=f"Modified {path}")
