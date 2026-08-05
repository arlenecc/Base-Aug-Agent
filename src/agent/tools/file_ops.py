"""File operation tools: read, write, modify."""
from __future__ import annotations

import os
from typing import Any

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult


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
            with open(full, "r", encoding="utf-8") as f:
                return ToolResult(True, output=f.read())
        except UnicodeDecodeError:
            import base64

            with open(full, "rb") as f:
                return ToolResult(True, output="(binary) " + base64.b64encode(f.read()).decode())


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
        # Atomic write: write to temp then os.replace. A crash mid-write leaves
        # the original file intact rather than a truncated/partial one.
        tmp = full + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, full)
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
        tmp = full + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, full)
        return ToolResult(True, output=f"Modified {path}")
