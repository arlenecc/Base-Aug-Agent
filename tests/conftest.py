"""Shared pytest fixtures."""
import os
import sys
import json
import tempfile
from pathlib import Path

import pytest

# Make src importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture()
def tmp_workspace(tmp_path):
    """An isolated working directory used as the agent's filesystem root."""
    return tmp_path


@pytest.fixture()
def config(tmp_workspace):
    from agent.config import AgentConfig

    cfg = AgentConfig()
    cfg.workspace = str(tmp_workspace)
    cfg.base_url = "https://api.example.com/v1"
    cfg.api_key = "test-key"
    cfg.model = "gpt-test"
    return cfg


@pytest.fixture()
def recording_callbacks():
    """An AgentCallbacks implementation that records every call for assertions."""
    from agent.agent import AgentCallbacks

    class Rec(AgentCallbacks):
        def __init__(self):
            self.content = []
            self.reasoning = []
            self.logs = []
            self.tool_starts = []
            self.tool_ends = []
            self.usages = []
            self.speeds = []
            self.confirms = []
            self.asks = []
            self._confirm_return = True
            self._ask_return = "ok"

        def on_content(self, text):
            self.content.append(text)

        def on_reasoning(self, text):
            self.reasoning.append(text)

        def on_tool_start(self, name, args):
            self.tool_starts.append((name, args))

        def on_tool_end(self, name, result):
            self.tool_ends.append((name, result))

        def on_log(self, line):
            self.logs.append(line)

        def on_usage(self, usage):
            self.usages.append(usage)

        def on_token_speed(self, tokens, speed):
            self.speeds.append((tokens, speed))

        def confirm(self, message):
            self.confirms.append(message)
            return self._confirm_return

        def ask_user(self, prompt):
            self.asks.append(prompt)
            return self._ask_return

    return Rec()
