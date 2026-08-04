"""Tests for the OpenAI-compatible LLM client."""
from __future__ import annotations

import json
from typing import Iterator

import pytest

from agent.llm_client import LLMClient, StreamEvent


def _sse_lines(*chunks: dict) -> list[bytes]:
    """Encode dict chunks as OpenAI-style SSE lines."""
    out = []
    for ch in chunks:
        out.append(b"data: " + json.dumps(ch).encode() + b"\n\n")
    out.append(b"data: [DONE]\n\n")
    return out


class _FakeResponse:
    """Mimics the parts of httpx.Response that LLMClient uses for streaming."""

    def __init__(self, chunks: list[bytes], status_code: int = 200, text: str = None):
        self._chunks = chunks
        self.status_code = status_code
        self.text = text if text is not None else b"".join(chunks).decode()
        self.headers = {}

    def iter_lines(self):
        for ch in self._chunks:
            for line in ch.split(b"\n"):
                if line:
                    yield line

    def iter_bytes(self) -> Iterator[bytes]:
        for ch in self._chunks:
            yield ch

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def read(self):
        return self.text.encode()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeClient:
    """A drop-in httpx.Client replacement driven by a queue of canned responses."""

    def __init__(self, responses):
        # responses: list of (method, url_substring, _FakeResponse)
        self._responses = list(responses)
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self._match("GET", url)

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self._match("POST", url)

    def stream(self, method, url, **kw):
        self.calls.append((method, url, kw))
        resp = self._match(method, url)
        return resp  # _FakeResponse is its own context manager

    def _match(self, method, url):
        for i, (m, sub, resp) in enumerate(self._responses):
            if m == method and sub in url:
                return self._responses.pop(i)[2]
        raise AssertionError(f"No canned response for {method} {url}")

    def close(self):
        pass


def _make_client(fake_responses, **kw) -> LLMClient:
    c = LLMClient(base_url="https://api.example.com/v1", api_key="k", model="m", **kw)
    c._http = _FakeClient(fake_responses)
    return c


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def test_list_models_returns_sorted_ids():
    body = {
        "object": "list",
        "data": [
            {"id": "gpt-4", "object": "model"},
            {"id": "gpt-3.5-turbo", "object": "model"},
            {"id": "text-embedding-ada-002", "object": "model"},
        ],
    }
    resp = _FakeResponse([json.dumps(body).encode()])
    client = _make_client([("GET", "/models", resp)])

    models = client.list_models()

    assert models == ["gpt-3.5-turbo", "gpt-4", "text-embedding-ada-002"]


def test_list_models_raises_on_http_error():
    resp = _FakeResponse([b"{}"], status_code=401)
    client = _make_client([("GET", "/models", resp)])
    with pytest.raises(Exception):
        client.list_models()


# ---------------------------------------------------------------------------
# chat_stream: content + reasoning + usage
# ---------------------------------------------------------------------------

def test_chat_stream_emits_content_deltas_then_done():
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}, "index": 0}]},
        {"choices": [{"delta": {"content": "lo"}, "index": 0}]},
        {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
    ]
    resp = _FakeResponse(_sse_lines(*chunks))
    client = _make_client([("POST", "/chat/completions", resp)])

    events = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    contents = [e.content for e in events if e.type == "content"]
    assert "".join(contents) == "Hello"
    done = [e for e in events if e.type == "done"][0]
    assert done.usage["total_tokens"] == 7
    assert done.tool_calls == []


def test_chat_stream_emits_reasoning_content():
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "think"}, "index": 0}]},
        {"choices": [{"delta": {"reasoning_content": "ing"}, "index": 0}]},
        {"choices": [{"delta": {"content": "answer"}, "index": 0, "finish_reason": "stop"}]},
    ]
    resp = _FakeResponse(_sse_lines(*chunks))
    client = _make_client([("POST", "/chat/completions", resp)])

    events = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    reasoning = "".join(e.content for e in events if e.type == "reasoning")
    content = "".join(e.content for e in events if e.type == "content")
    assert reasoning == "thinking"
    assert content == "answer"


# ---------------------------------------------------------------------------
# chat_stream: tool_calls accumulation across deltas
# ---------------------------------------------------------------------------

def test_chat_stream_accumulates_tool_calls():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "function": {"name": "file_read", "arguments": ""}
        }]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": "{\"path\":"}
        }]}, "index": 0}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": " \"a.txt\"}"}
        }]}, "index": 0, "finish_reason": "tool_calls"}]},
    ]
    resp = _FakeResponse(_sse_lines(*chunks))
    client = _make_client([("POST", "/chat/completions", resp)])

    events = list(client.chat_stream([{"role": "user", "content": "hi"}]))
    done = [e for e in events if e.type == "done"][0]

    assert len(done.tool_calls) == 1
    tc = done.tool_calls[0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "file_read"
    assert json.loads(tc["function"]["arguments"]) == {"path": "a.txt"}


def test_chat_stream_passes_tools_in_request():
    resp = _FakeResponse(_sse_lines({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}))
    client = _make_client([("POST", "/chat/completions", resp)])
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    list(client.chat_stream([{"role": "user", "content": "hi"}], tools=tools))

    _, _, kw = client._http.calls[0]
    body = json.loads(kw["content"])
    assert body["tools"] == tools
    assert body["stream"] is True


def test_chat_stream_handles_empty_done_marker_only():
    resp = _FakeResponse([b"data: [DONE]\n\n"])
    client = _make_client([("POST", "/chat/completions", resp)])
    events = list(client.chat_stream([{"role": "user", "content": "hi"}]))
    # No content or reasoning deltas; only a final empty completion signal.
    assert [e for e in events if e.type in ("content", "reasoning")] == []
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert done[0].tool_calls == [] and done[0].usage == {}
