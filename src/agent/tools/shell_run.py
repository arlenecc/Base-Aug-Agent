"""shell_run tool: execute shell/terminal commands in the workspace.

Runs the command with the workspace as cwd, captures stdout+stderr, and
enforces a timeout so a hung command cannot block the agent indefinitely.
Marked destructive because shell commands can affect the system arbitrarily;
the agent's confirmation policy prompts the user before execution.
"""
from __future__ import annotations

import os
import subprocess

from ..config import AgentConfig
from .base import Tool, ToolRegistry, ToolResult


# Default timeout (seconds) for shell commands. Can be overridden per-call via
# the `timeout` argument. Long enough for pip installs, short enough that a
# hung command doesn't freeze the agent.
_DEFAULT_TIMEOUT = 120


class ShellRunTool(Tool):
    name = "shell_run"
    description = (
        "Execute a shell command in the workspace and return stdout+stderr. "
        "Use this for running CLI tools, git, pip, build scripts, etc. "
        "The command runs with the workspace as the working directory."
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

        ws = os.path.abspath(self.config.workspace)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=ws,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                False,
                error=f"Command timed out after {timeout:.0f}s: {command[:200]}",
            )
        except Exception as e:
            return ToolResult(False, error=f"{type(e).__name__}: {e}")

        # Combine stdout and stderr so the model sees everything.
        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + "[stderr]\n" + proc.stderr

        if proc.returncode == 0:
            return ToolResult(True, output=output or "(no output)")
        # Non-zero exit: report as failure but still include output so the
        # model can diagnose what went wrong.
        return ToolResult(
            False,
            output=output,
            error=f"Exit code {proc.returncode}",
        )
