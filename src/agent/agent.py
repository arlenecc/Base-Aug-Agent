"""The agent: reasoning loop, tool dispatch, streaming, confirmation, skills."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from .config import AgentConfig
from .llm_client import LLMClient, StreamEvent
from .skills import SkillManager
from .tools import ToolRegistry, parse_args


# ---------------------------------------------------------------------------
# Callbacks protocol
# ---------------------------------------------------------------------------


class AgentCallbacks:
    """Override these methods to drive the UI. Default impls are no-ops."""

    def on_content(self, text: str) -> None: ...
    def on_reasoning(self, text: str) -> None: ...
    def on_tool_start(self, name: str, args: Dict[str, Any]) -> None: ...
    def on_tool_end(self, name: str, result) -> None: ...
    def on_log(self, line: str) -> None: ...
    def on_usage(self, usage: Dict[str, int]) -> None: ...
    def on_token_speed(self, total_tokens: int, speed: float) -> None: ...
    def on_skill_suggested(self, skill) -> None: ...

    def confirm(self, message: str) -> bool:
        return True

    def ask_user(self, prompt: str) -> Optional[str]:
        return None

    def on_finished(self) -> None: ...


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are base-agent, a local autonomous agent that completes the user's task by reasoning, planning, and calling tools.

Operating principles:
1. Analyze the user's request and any prior context. Plan the steps needed.
2. Call tools to gather information and take actions. After each tool result, decide the next step.
3. Prefer the least destructive path. For any risky/irreversible action, the user will be asked to confirm.
4. If information is missing or ambiguous, use the `ask_user` tool rather than guessing.
5. Use `work_memory` to keep short-term notes across steps; use `memory_extract` to persist reusable facts.
6. When you have the final answer, reply with a concise summary in plain text (no tool calls). The visible reply comes from your `content` field; your `reasoning_content` (if any) is shown separately as the thinking trace.

Tools available operate inside the workspace directory. File paths are relative to the workspace.

Confirmation policy: operations confined to the workspace (file_read, file_write, file_modify, work_memory, memory_extract, web_scan, webexec_js, ask_user) run immediately without asking. Only `code_run` and `shell_run` prompt the user first, because they can execute arbitrary system-affecting code. If you intend a genuinely destructive or irreversible action, still prefer to ask via `ask_user` first.

Tool calling: when you need to take an action, emit a tool_call with the tool name and JSON arguments. The agent will execute it locally and return the result (stdout, stderr, errors, or file contents) as a tool message in the next turn. Use this feedback to decide the next step. You may chain multiple tool calls across turns until the task is done.

Knowledge base: if a local knowledge base is configured and indexed, you can use `rag_search` to find relevant information from the user's documents. Use `rag_status` to check what's available, and `rag_ingest` to re-index after new files are added to the knowledge base directory. Always prefer searching the knowledge base when the user's question relates to their own documents or specialized knowledge."""


# ---------------------------------------------------------------------------
# Token estimation: approximate token count from text length.
# Used for real-time speed display during reasoning streaming (where we don't
# yet have the API's usage figure) and as a fallback when the API's usage
# omits reasoning tokens entirely.
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token-count estimate from text.

    Heuristic: ASCII text averages ~4 chars/token; CJK characters are denser
    (~1.5 chars/token). We split the text into CJK and non-CJK runs and apply
    the appropriate ratio. This is only for live speed display and as a
    fallback when the API underreports -- the exact count is not critical.
    """
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        # CJK Unified Ideographs + common CJK extension ranges
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or  # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or  # CJK Extension A
                0x3000 <= cp <= 0x30FF or  # CJK symbols + Japanese kana
                0xFF00 <= cp <= 0xFFEF):  # Fullwidth forms
            cjk += 1
        else:
            other += 1
    # CJK: ~1.5 chars/token; ASCII/other: ~4 chars/token
    return max(1, int(cjk / 1.5 + other / 4.0))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    def __init__(
        self,
        llm: "Any",
        config: AgentConfig,
        callbacks: AgentCallbacks,
        tool_registry: Optional[ToolRegistry] = None,
        skills: Optional[SkillManager] = None,
    ):
        self.llm = llm  # object with chat_stream(messages, tools, temperature)
        self.config = config
        self.callbacks = callbacks
        self.tools = tool_registry or ToolRegistry(config=config, callbacks=callbacks)
        self.skills = skills or SkillManager(path=os.path.join(config.workspace, ".agent", "skills.json"))
        self.max_iterations = int(getattr(config, "max_iterations", 15))
        self._history: List[Dict[str, Any]] = []
        self._total_tokens = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._history = []
        self._total_tokens = 0

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """Process one user turn. Returns the final assistant content."""
        self._log(f"USER: {user_input}")

        # skill suggestion (固化) - non-blocking
        try:
            suggested = self.skills.record_request(user_input)
            if suggested is not None:
                self.callbacks.on_skill_suggested(suggested)
        except Exception as e:  # pragma: no cover
            self._log(f"[skills] record_request error: {e}")

        self._history.append({"role": "user", "content": user_input})

        final_content = ""
        for i in range(self.max_iterations):
            messages = self._build_messages()
            self._log(f"[iter {i+1}] -> LLM ({self.config.model})")

            content_buf = []
            reasoning_buf = []
            tool_calls: List[Dict[str, Any]] = []
            usage: Dict[str, int] = {}
            t0 = time.time()
            completion_tokens = 0
            # Accumulated streamed text for live token estimation. Updated on
            # every reasoning/content chunk; used to emit real-time speed updates
            # before the API's final usage figure arrives.
            streamed_chars = 0
            last_speed_emit = 0.0  # timestamp of last on_token_speed emission

            try:
                for ev in self.llm.chat_stream(
                    messages, tools=self.tools.schemas(), temperature=self.config.temperature
                ):
                    if ev.type == "content":
                        content_buf.append(ev.content)
                        self.callbacks.on_content(ev.content)
                        streamed_chars += len(ev.content)
                    elif ev.type == "reasoning":
                        reasoning_buf.append(ev.content)
                        self.callbacks.on_reasoning(ev.content)
                        streamed_chars += len(ev.content)
                        now = time.time()
                        if now - last_speed_emit >= 0.3:
                            elapsed_so_far = max(now - t0, 1e-6)
                            est_tokens = _estimate_tokens("".join(reasoning_buf))
                            live_speed = est_tokens / elapsed_so_far
                            self.callbacks.on_token_speed(
                                self._total_tokens + est_tokens, live_speed
                            )
                            last_speed_emit = now
                    elif ev.type == "done":
                        tool_calls = ev.tool_calls
                        usage = ev.usage
                        completion_tokens = usage.get("completion_tokens", 0)
            except Exception as e:
                self._log(f"[agent] LLM stream error: {e}")
                self.callbacks.on_error(str(e))
                self.callbacks.on_finished()
                return final_content or ""

            elapsed = max(time.time() - t0, 1e-6)

            # Determine reasoning token count for this turn.
            # 1. If the API provides completion_tokens_details.reasoning_tokens,
            #    trust that exact figure.
            # 2. Otherwise estimate from the streamed reasoning text so that
            #    reasoning-heavy models still show a realistic tok/s and the
            #    running total reflects actual consumption.
            details = usage.get("completion_tokens_details") or {}
            api_reasoning_tokens = details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0
            reasoning_text = "".join(reasoning_buf)
            estimated_reasoning_tokens = _estimate_tokens(reasoning_text)

            if api_reasoning_tokens:
                reasoning_tokens = api_reasoning_tokens
                # API already includes reasoning in completion_tokens; don't add again
                extra_reasoning_tokens = 0
            else:
                # API's completion_tokens may or may not include reasoning.
                # If the API reported completion_tokens but we streamed a lot of
                # reasoning text that wasn't counted, add our estimate on top.
                reasoning_tokens = estimated_reasoning_tokens
                extra_reasoning_tokens = estimated_reasoning_tokens

            # Total tokens for this turn: use the API's total if it's non-zero
            # and includes reasoning (i.e. api_reasoning_tokens > 0 or no
            # reasoning text was streamed). Otherwise add our estimate.
            api_total = usage.get("total_tokens", 0)
            if api_total > 0 and (api_reasoning_tokens or not reasoning_text):
                # API total is trustworthy
                self._total_tokens += api_total
                turn_tokens = api_total
            else:
                # API total is missing or doesn't include reasoning: use
                # api_total + estimated reasoning tokens
                self._total_tokens += api_total + extra_reasoning_tokens
                turn_tokens = api_total + extra_reasoning_tokens

            if usage:
                self.callbacks.on_usage(usage)
            # Final speed: use the turn's total token count (including reasoning)
            # divided by elapsed time. Fall back to completion_tokens if we
            # somehow have no estimate.
            speed_tokens = turn_tokens if turn_tokens else completion_tokens
            speed = (speed_tokens / elapsed) if speed_tokens else 0.0
            self.callbacks.on_token_speed(self._total_tokens, speed)
            self._log(
                f"[iter {i+1}] <- {len(''.join(content_buf))} content chars, "
                f"{len(reasoning_text)} reasoning chars (~{reasoning_tokens} tok), "
                f"{len(tool_calls)} tool_calls, usage={usage}, "
                f"{speed_tokens} tok / {elapsed:.2f}s = {speed:.1f} tok/s"
            )

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if content_buf:
                assistant_msg["content"] = "".join(content_buf)
            else:
                # OpenAI-compatible APIs require the `content` field to be present
                # on assistant messages even when only tool_calls are emitted;
                # omitting it triggers HTTP 400 on some servers. Use None (null)
                # rather than "" so the model sees "no text reply", not "empty reply".
                assistant_msg["content"] = None
            if tool_calls:
                # Ensure every tool_call has a non-empty id. Some local models
                # omit it; without an id the tool result cannot be correlated
                # back (tool_call_id must match), and the API may reject the
                # request. We synthesize stable ids before recording history.
                for idx, tc in enumerate(tool_calls):
                    if not tc.get("id"):
                        tc["id"] = f"call_{idx}_{id(tc):x}"
                assistant_msg["tool_calls"] = tool_calls
            self._history.append(assistant_msg)
            final_content = assistant_msg.get("content") or ""

            if not tool_calls:
                # no further action -> done
                self.callbacks.on_finished()
                return final_content

            # execute each tool call
            for tc in tool_calls:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "")

                # Parse arguments. Malformed JSON must be reported to the model
                # as a parse error (not silently treated as {} which would
                # surface as a confusing "missing required argument" later).
                try:
                    args = parse_args(raw_args)
                except ValueError as e:
                    self._log(f"[tool] {name} arg-parse error: {e}")
                    from .tools import ToolResult
                    res = ToolResult(False, error=str(e))
                    self.callbacks.on_tool_end(name, res)
                    self._history.append(self._tool_message(tc, res))
                    continue

                tool = self.tools.get(name)

                needs_confirm = bool(tool and tool.should_confirm(args)) if tool else False
                self._log(f"[tool] {name}({raw_args}) needs_confirm={needs_confirm}")

                if tool is None:
                    res = self.tools.execute(name, args)  # returns unknown error
                    self.callbacks.on_tool_start(name, args)
                    self.callbacks.on_tool_end(name, res)
                    self._history.append(self._tool_message(tc, res))
                    continue

                if needs_confirm:
                    confirm_msg = self._confirm_message(name, args)
                    if not self.callbacks.confirm(confirm_msg):
                        self._log(f"[tool] {name} DENIED by user")
                        from .tools import ToolResult

                        res = ToolResult(False, error=f"User denied {name} ({confirm_msg})")
                        # Tool never actually started; only report the denial result.
                        self.callbacks.on_tool_end(name, res)
                        self._history.append(self._tool_message(tc, res))
                        continue

                self.callbacks.on_tool_start(name, args)
                res = self.tools.execute(name, args)
                self.callbacks.on_tool_end(name, res)
                self._log(f"[tool] {name} -> success={res.success} "
                          f"out_len={len(res.output)} err={(res.error or '')[:120]}")
                self._history.append(self._tool_message(tc, res))

        self._log("[agent] max iterations reached, stopping.")
        self.callbacks.on_finished()
        return final_content

    # ------------------------------------------------------------------
    def _build_messages(self) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        msgs.extend(self._history)
        return msgs

    def _system_prompt(self) -> str:
        base = SYSTEM_PROMPT
        try:
            from .tools.memory import _work_memory, _long_memory

            wm = _work_memory(self.config).list()
            lt = _long_memory(self.config).all()
            if wm:
                base += "\n\n# Work memory\n" + json.dumps(wm, ensure_ascii=False, indent=2)
            if lt:
                base += "\n\n# Long-term memory\n" + json.dumps(lt, ensure_ascii=False, indent=2)
        except Exception:  # pragma: no cover
            pass
        # inject matched skill prompt
        try:
            matched = self.skills.match(self._history[-1]["content"]) if self._history else None
            if matched:
                base += f"\n\n# Active skill: {matched.name}\n{matched.prompt}"
        except Exception:  # pragma: no cover
            pass
        return base

    def _tool_message(self, tool_call: Dict[str, Any], result) -> Dict[str, Any]:
        content = result.to_message() if hasattr(result, "to_message") else str(result)
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": (tool_call.get("function") or {}).get("name", ""),
            "content": content,
        }

    def _confirm_message(self, name: str, args: Dict[str, Any]) -> str:
        try:
            pretty = json.dumps(args, ensure_ascii=False)
        except Exception:
            pretty = str(args)
        return f"Tool `{name}` is about to run with arguments:\n{pretty}\n\nProceed?"

    def _log(self, line: str) -> None:
        self.callbacks.on_log(line)
