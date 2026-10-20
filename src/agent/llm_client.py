"""OpenAI-compatible streaming chat client.

Supports:
- listing models (GET /v1/models)
- streaming chat completions with tool_calls and reasoning_content (deepseek-style)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """A single parsed streaming event.

    type:
        "content"   - a delta of the visible assistant reply (content field)
        "reasoning" - a delta of the model's reasoning (reasoning_content field)
        "done"      - emitted once at the end; carries accumulated tool_calls + usage
    """

    type: str
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    # 服务端的结束原因（"stop" / "length" / "tool_calls" / ...）。
    # finish_reason="length" 说明回复被 max_tokens 截断——上层需要能识别，
    # 否则半截回答会被当成完整答案写进历史。
    finish_reason: str = ""


class LLMError(RuntimeError):
    pass


class _RetryWithoutStreamOptions(RuntimeError):
    """内部信号：服务端拒绝了 ``stream_options``，需降级重试一次。"""


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_tokens: int = 32768,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        # <think>…</think> 跨 chunk 的解析状态（见 _split_think / _flush_think）。
        self._think_in = False
        self._think_pending = ""
        # Use a structured timeout so that streaming reads have a per-chunk
        # deadline.  A plain `timeout=float` applies only to the *whole*
        # request (and, for streaming, once the first byte arrives, an idle
        # server that stalls mid-generation would block `iter_lines()`
        # FOREVER — the "reply stops halfway and never resumes" symptom).
        # `read` here is the max gap between chunks; local models can take
        # tens of seconds on a hard reasoning step, so we use a generous but
        # finite bound to fail fast instead of hanging the UI.
        if isinstance(timeout, httpx.Timeout):
            self._http: httpx.Client = httpx.Client(timeout=timeout)
        else:
            read_timeout = max(timeout, 60.0)
            self._http: httpx.Client = httpx.Client(
                timeout=httpx.Timeout(connect=timeout, read=read_timeout,
                                      write=timeout, pool=timeout)
            )

    # ------------------------------------------------------------------
    # set http (used by tests; also lets the UI inject a configured client)
    # ------------------------------------------------------------------
    def _set_http(self, client: Any) -> None:  # pragma: no cover - test helper
        self._http = client

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # ------------------------------------------------------------------
    # list models
    # ------------------------------------------------------------------
    def list_models(self) -> List[str]:
        url = f"{self.base_url}/models"
        resp = self._http.get(url, headers=self._headers())
        try:
            if resp.status_code >= 400:
                raise LLMError(f"list_models failed: HTTP {resp.status_code} {resp.text[:200]}")
            data = resp.json()
        except ValueError as e:
            # 服务端返回了非 JSON（HTML 错误页、网关拦截页）。裸 ValueError 会让
            # 「获取模型」按钮抛一句没有上下文的异常。
            raise LLMError(
                f"list_models: 响应不是合法 JSON ({e})：{resp.text[:200]!r}"
            )
        finally:
            resp.close()
        items = data.get("data", []) if isinstance(data, dict) else data
        ids = [it["id"] for it in items if isinstance(it, dict) and it.get("id")]
        return sorted(ids)

    # ------------------------------------------------------------------
    # chat (streaming)
    # ------------------------------------------------------------------
    # 瞬时错误的重试策略。旧实现零重试：一次网络抖动 / 服务端 5xx / 429 就
    # 让整轮对话失败。注意只在**尚未产出任何事件**时重试——已经流式输出过内容
    # 再重试会造成重复输出。
    _MAX_RETRIES = 2
    _RETRY_BASE_DELAY = 0.5

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """判断是否为值得重试的瞬时错误。"""
        if isinstance(exc, (httpx.ConnectError, httpx.ReadError,
                            httpx.WriteError, httpx.RemoteProtocolError,
                            httpx.PoolTimeout, httpx.ConnectTimeout)):
            return True
        # httpx.ReadTimeout 是 TimeoutException 的子类：读超时意味着还没收到
        # 完整响应，重试是安全的（前提是还没有消费过任何 chunk）。
        if isinstance(exc, httpx.ReadTimeout):
            return True
        if isinstance(exc, LLMError):
            msg = str(exc)
            # 服务端瞬时故障：5xx / 429。
            if "HTTP 5" in msg or "HTTP 429" in msg or "429" in msg:
                return True
        return False

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Iterator[StreamEvent]:
        last_err: Optional[BaseException] = None
        for attempt in range(self._MAX_RETRIES + 1):
            produced = False
            try:
                for ev in self._chat_stream_once(messages, tools, temperature):
                    if ev.type != "done":
                        produced = True
                    yield ev
                return
            except _RetryWithoutStreamOptions:
                # 降级后立刻重试（不计入普通重试次数预算的等待）。
                if attempt >= self._MAX_RETRIES:
                    raise LLMError("chat_stream: 服务端拒绝了 stream_options，降级后仍失败")
                continue
            except Exception as e:
                # 已经吐出过内容 → 重试会造成重复输出，直接上抛。
                if produced or attempt >= self._MAX_RETRIES or not self._is_retryable(e):
                    raise
                last_err = e
                delay = self._RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "chat_stream: %s，%.1fs 后重试 (%d/%d)",
                    type(e).__name__, delay, attempt + 1, self._MAX_RETRIES,
                )
                time.sleep(delay)
        if last_err is not None:  # pragma: no cover - 防御性
            raise last_err

    def _chat_stream_once(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Iterator[StreamEvent]:
        url = f"{self.base_url}/chat/completions"
        # raw_content 是本地 MLX 后端专用的历史镜像（见 Agent._run 的「双视图」
        # 设计）：只有本地后端读它重建忠实的 KV 前缀序列。HTTP 服务端不认识这
        # 个字段——宽容的会忽略，严格的会以 400 拒绝。发出去之前剥掉。
        api_messages = [
            {k: v for k, v in m.items() if k != "raw_content"}
            if isinstance(m, dict) else m
            for m in (messages or [])
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": True,
        }
        # 部分本地服务端（旧版 llama.cpp 等）不认识 stream_options，会直接
        # 400。首次因此失败后降级重试不带该字段（见 _send_once 的调用方）。
        if getattr(self, "_stream_options_ok", True):
            payload["stream_options"] = {"include_usage": True}
        if self.min_p > 0:
            payload["min_p"] = self.min_p
        if self.top_k > 0:
            payload["top_k"] = self.top_k
        if self.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.repetition_penalty
        if tools:
            payload["tools"] = tools

        tool_acc: Dict[int, Dict[str, Any]] = {}
        final_usage: Dict[str, int] = {}
        final_finish_reason = ""
        # 每轮对话重置 <think> 解析状态（client 实例跨调用复用）。
        self._think_in = False
        self._think_pending = ""

        with self._http.stream("POST", url, headers=self._headers(), content=json.dumps(payload)) as resp:
            if resp.status_code >= 400:
                try:
                    body = resp.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                # 服务端不认识 stream_options → 永久降级并重试一次，而不是让
                # 整个对话能力在这个服务端上不可用。
                if (
                    resp.status_code == 400
                    and "stream_options" in body
                    and payload.get("stream_options")
                ):
                    self._stream_options_ok = False
                    raise _RetryWithoutStreamOptions()
                raise LLMError(f"chat_stream failed: HTTP {resp.status_code} {body[:300]}")

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if obj.get("usage"):
                    final_usage = obj["usage"]

                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {}) or {}
                # 记录结束原因（"stop" / "length" / "tool_calls"）。最后一个
                # 非空值即为本轮结束原因——中途 chunk 通常不带。
                fr = choice.get("finish_reason")
                if fr:
                    final_finish_reason = fr

                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    yield StreamEvent("reasoning", content=rc)

                ct = delta.get("content")
                if ct:
                    # 本地推理模型（Qwen3 / DeepSeek-R1 经 Ollama 等）常把思考过程
                    # 放在 content 的 <think>…</think> 标签里，而非独立 reasoning 字段。
                    # 拆出来单独送「思考过程」面板，避免思考与正式回复混进对话区。
                    for r_text, c_text in self._split_think(ct):
                        if r_text:
                            yield StreamEvent("reasoning", content=r_text)
                        if c_text:
                            yield StreamEvent("content", content=c_text)

                for tc in delta.get("tool_calls", []) or []:
                    idx = tc.get("index", 0)
                    slot = tool_acc.setdefault(idx, {"id": "", "type": "function",
                                                    "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    if tc.get("type"):
                        slot["type"] = tc["type"]
                    fn = tc.get("function", {}) or {}
                    if fn.get("name"):
                        name_delta = fn["name"]
                        cur = slot["function"]["name"]
                        if not cur:
                            slot["function"]["name"] = name_delta
                        elif name_delta not in cur:
                            # 只在确实是「续传片段」时才拼接。部分服务端会在每个
                            # chunk 重复下发完整 name，无条件 += 会拼成
                            # "read_fileread_file" 这类鬼名字。
                            slot["function"]["name"] = cur + name_delta
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]

        # 流结束：把 <think> 缓冲里残留的尾部（模型被截断、未闭合等）落盘。
        for r_text, c_text in self._flush_think():
            if r_text:
                yield StreamEvent("reasoning", content=r_text)
            if c_text:
                yield StreamEvent("content", content=c_text)

        yield StreamEvent(
            "done",
            tool_calls=[tool_acc[k] for k in sorted(tool_acc)],
            usage=final_usage,
            finish_reason=final_finish_reason,
        )

    # ------------------------------------------------------------------
    # <think>…</think> 思考片段提取（本地推理模型常把思考塞进 content）
    # ------------------------------------------------------------------
    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"

    def _split_think(self, text):
        """把 content 增量里 <think>…</think> 包裹的推理拆出来。

        返回 [(reasoning_text, content_text), ...]（空串表示对应部分为空）。
        跨 chunk 的标签通过 _think_pending 缓冲处理，避免把 '<thi' 之类误判成
        未闭合标签；_think_in 记录当前是否处于思考区间内。
        """
        self._think_pending += text
        buf = self._think_pending
        results = []
        i = 0
        n = len(buf)
        while i < n:
            if self._think_in:
                close = self._THINK_CLOSE
                j = buf.find(close, i)
                if j == -1:
                    # 还没遇到闭合标签：留出末尾可能拆分标签的字符先缓冲。
                    keep = len(close) - 1
                    emit_len = max(0, n - i - keep)
                    if emit_len > 0:
                        results.append((buf[i:i + emit_len], ""))
                    self._think_pending = buf[i + emit_len:]
                    break
                results.append((buf[i:j], ""))
                self._think_in = False
                i = j + len(close)
            else:
                open_ = self._THINK_OPEN
                j = buf.find(open_, i)
                if j == -1:
                    keep = len(open_) - 1
                    emit_len = max(0, n - i - keep)
                    if emit_len > 0:
                        results.append(("", buf[i:i + emit_len]))
                    self._think_pending = buf[i + emit_len:]
                    break
                results.append(("", buf[i:j]))
                self._think_in = True
                i = j + len(open_)
        return results

    def _flush_think(self):
        """流结束时的尾缓冲落盘；返回 [(reasoning_text, content_text), ...]。"""
        pending = self._think_pending
        self._think_pending = ""
        if not pending:
            return []
        if self._think_in:
            return [(pending, "")]
        return [("", pending)]

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass
