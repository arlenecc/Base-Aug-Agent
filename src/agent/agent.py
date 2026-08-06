"""The agent: reasoning loop, tool dispatch, streaming, confirmation, skills."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
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
    def on_error(self, message: str) -> None: ...
    def on_context_shrunk(self, summary: str, reason: str) -> None: ...

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

Knowledge base: if a local knowledge base is configured and indexed, you MUST proactively use `rag_search` to retrieve relevant context BEFORE answering any user question that could be answered from the user's own documents. Use `rag_status` to check what's available, and `rag_ingest` to re-index after new files are added to the knowledge base directory. Workflow:
1. At the start of a conversation turn, if the user's question could relate to indexed documents, call `rag_status` first to check whether the knowledge base is available and non-empty.
2. If the knowledge base has data, call `rag_search` with a relevant query derived from the user's question.
3. Use the retrieved context to ground your answer. If no relevant context is found, say so and answer from your general knowledge.
4. After the user adds new files to the knowledge base directory, call `rag_ingest` to index them (this is incremental — only new/modified files are processed).
Always prefer knowledge base context over general knowledge when answering questions about the user's documents or specialized domain."""


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
    # Fast path: if the text is pure ASCII (common for English/code/JSON),
    # skip the per-character CJK check entirely — str.isascii() is O(1)-ish
    # (implemented as a single memcmp on CPython).
    if text.isascii():
        return max(1, len(text) // 4)

    # Mixed/CJK text: count CJK characters via str.translate for speed.
    # Building the translation table is O(1) and translate is O(n) in C,
    # much faster than a Python-level per-character loop for long text.
    cjk = 0
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or  # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or  # CJK Extension A
                0x3000 <= cp <= 0x30FF or  # CJK symbols + Japanese kana
                0xFF00 <= cp <= 0xFFEF):  # Fullwidth forms
            cjk += 1
    other = len(text) - cjk
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
        self.max_history = int(getattr(config, "max_history", 50))
        # Context shrink config: shrink proactively when estimated prompt size
        # exceeds context_shrink_ratio * max_context_tokens, or reactively when
        # the LLM API returns a context_length_exceeded-style error.
        self.max_context_tokens = int(getattr(config, "max_context_tokens", 32000))
        self.context_shrink_ratio = float(getattr(config, "context_shrink_ratio", 0.9))
        # Cap shrink attempts per run() so a misconfigured LLM can't loop
        # forever on summarize-then-retry.
        self._max_shrinks_per_run = 3
        # Number of recent messages to retain verbatim during a shrink; older
        # messages are summarized. Tuned to keep the immediate task context
        # (current user request + latest tool result + assistant reply) intact.
        self._shrink_keep_recent = 6
        self._history: List[Dict[str, Any]] = []
        self._total_tokens = 0
        self._cached_prompt_turn = -1
        self._cached_prompt = ""
        self._turn_idx = 0  # incremented each run() call; used for prompt cache
        self._tool_call_counter = 0  # stable id generator for missing tool_call ids

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._history = []
        self._total_tokens = 0
        self._tool_call_counter = 0
        # Invalidate cached prompt
        self._cached_prompt_turn = -1
        self._cached_prompt = ""

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def _trim_history(self) -> None:
        """Evict oldest messages when history exceeds max_history.

        Never leaves a dangling ``tool`` result message at the start —
        it must follow an assistant message with matching ``tool_calls``.
        """
        limit = self.max_history
        if limit <= 0 or len(self._history) <= limit:
            return
        # Start with a simple slice, then skip any leading tool messages.
        cut = len(self._history) - limit
        while cut < len(self._history) and self._history[cut].get("role") == "tool":
            cut += 1
        del self._history[:cut]

    # ------------------------------------------------------------------
    # Context-size estimation and shrink strategy
    # ------------------------------------------------------------------

    def _msg_token_count(self, msg: Dict[str, Any]) -> int:
        """Token count for a single message (role + content + tool_calls)."""
        count = 4  # role + structural overhead per message
        content = msg.get("content")
        if isinstance(content, str):
            count += _estimate_tokens(content)
        elif content:
            count += _estimate_tokens(json.dumps(content, ensure_ascii=False))
        tcs = msg.get("tool_calls") or []
        for tc in tcs:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            count += _estimate_tokens(fn.get("name", ""))
            count += _estimate_tokens(fn.get("arguments", "") or "")
        return count

    def _estimate_history_tokens(self) -> int:
        """Estimate the token count of the full prompt (system + history).

        Always does a full O(n) scan of _history. The history list is
        typically short (<= max_history messages), so the cost is negligible
        compared to the complexity of maintaining an incremental counter
        that can get out of sync when _history is modified directly.
        """
        system_tokens = _estimate_tokens(self._system_prompt())
        history_tokens = sum(self._msg_token_count(msg) for msg in self._history)
        return system_tokens + history_tokens

    def _should_shrink_context(self) -> bool:
        """Return True if the estimated context size exceeds the shrink
        threshold (context_shrink_ratio * max_context_tokens)."""
        if self.max_context_tokens <= 0:
            return False
        threshold = int(self.max_context_tokens * self.context_shrink_ratio)
        return self._estimate_history_tokens() > threshold

    @staticmethod
    def _is_context_too_long_error(err_str: str) -> bool:
        """Heuristically detect LLM API errors caused by exceeding the
        context window. Covers OpenAI, Anthropic, and common local server
        (vLLM/llama.cpp/Ollama) error message variants."""
        e = err_str.lower()
        if "context_length" in e or "context length" in e:
            return True
        if "context window" in e or "maximum context" in e:
            return True
        if "prompt is too long" in e or "prompt too long" in e:
            return True
        if "too many tokens" in e or "token limit" in e or "tokens limit" in e:
            return True
        return False

    def _shrink_context(self, reason: str) -> bool:
        """Summarize older messages, replace them with a summary marker, and
        append the summary (with timestamp) to workspace/memory.md for
        long-term/time-based recall.

        Keeps the most recent ``_shrink_keep_recent`` messages verbatim so the
        agent still has the immediate task context (latest user request, tool
        results, and assistant reply). Older messages are summarized via the
        LLM and replaced with a single user-role summary message.

        Returns True if a shrink actually happened, False if there wasn't
        enough history to shrink (in which case the caller should not retry).
        """
        # Need at least keep_recent + a few older messages to make a summary
        # worthwhile; otherwise we'd just keep what we have.
        if len(self._history) <= self._shrink_keep_recent + 2:
            self._log(
                f"[context] shrink requested ({reason}) but history too small "
                f"({len(self._history)} msgs) — skipping"
            )
            return False

        old_messages = self._history[:-self._shrink_keep_recent]
        recent_messages = self._history[-self._shrink_keep_recent:]

        # Ensure recent_messages doesn't start with a dangling tool result.
        # A tool message must follow an assistant message with matching
        # tool_calls. If the slice boundary falls right after such an assistant
        # message, the tool result at the start of recent_messages would have
        # no preceding assistant — the API would reject the request. Pull the
        # preceding assistant (and any earlier tool messages in the same turn)
        # back from old_messages into recent_messages.
        while recent_messages and recent_messages[0].get("role") == "tool":
            if not old_messages:
                break
            recent_messages.insert(0, old_messages.pop())

        transcript = self._format_messages_for_summary(old_messages)
        summary_prompt = (
            "请把下面的对话历史压缩成一份精炼的摘要，只保留关键信息：\n"
            "1. 用户的核心需求和当前任务目标\n"
            "2. 已经做出的重要决定与原因\n"
            "3. 已经完成的步骤及其结果（含工具调用的关键输出）\n"
            "4. 尚未完成的子任务和下一步计划\n"
            "5. 关键的文件路径、配置值、用户偏好、URL 等具体信息\n"
            "6. 任何错误、警告或需要后续注意的事项\n\n"
            "要求：用要点列出，不要复述整段对话；保留所有具体的路径、数值、"
            "标识符；不要编造未出现过的信息。若某条信息不重要可省略。\n\n"
            f"对话历史:\n{transcript}"
        )

        try:
            summary = self._summarize_via_llm(summary_prompt)
        except Exception as e:
            self._log(f"[context] summarize LLM call failed: {e}")
            summary = ""

        if not summary or not summary.strip():
            # Fallback: build a minimal structural summary from messages so we
            # don't lose tool results entirely when the LLM is unavailable.
            summary = self._fallback_summary(old_messages)
            if not summary:
                summary = "(上下文收缩：旧消息已清理，无可用摘要)"

        summary = summary.strip()

        # Persist to memory.md with timestamp for long-term recall.
        self._append_summary_to_memory(summary, reason)

        # Replace old messages with a single user-role summary marker. Using
        # role=user (not system) keeps it inside the conversation the model
        # treats as dialogue rather than instructions, and ensures the very
        # first history message is never a dangling tool result.
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        summary_msg = {
            "role": "user",
            "content": (
                f"[上下文摘要 - {ts} - 触发原因: {reason}]\n"
                f"{summary}\n"
                f"[摘要结束，以下是后续对话]"
            ),
        }
        self._history = [summary_msg] + list(recent_messages)
        # Recompute incremental counter after shrink (cheaper than full O(n) scan
        # on every turn, and only runs when context actually shrinks).
        self._log(
            f"[context] shrunk: removed {len(old_messages)} old msgs, "
            f"kept {len(recent_messages)} recent + 1 summary marker "
            f"(reason: {reason})"
        )
        try:
            self.callbacks.on_context_shrunk(summary, reason)
        except Exception:  # pragma: no cover - defensive
            pass
        return True

    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Render a list of history messages as a readable transcript for the
        summarizer LLM. Truncates very long tool outputs to keep the
        summarization prompt itself within budget."""
        lines = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif content is None:
                text = ""
            else:
                text = json.dumps(content, ensure_ascii=False)
            # Cap individual message text so the summarizer prompt stays bounded.
            if len(text) > 2000:
                text = text[:2000] + f"... [truncated, {len(text)} chars total]"
            if role == "tool":
                name = msg.get("name", "tool")
                lines.append(f"[{i}] 工具 {name} 结果: {text}")
            elif role == "assistant":
                tcs = msg.get("tool_calls") or []
                if tcs:
                    tc_descs = []
                    for tc in tcs:
                        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                        tc_descs.append(f"{fn.get('name','?')}({fn.get('arguments','')})")
                    lines.append(f"[{i}] 助手(调用工具: {', '.join(tc_descs)}): {text}")
                else:
                    lines.append(f"[{i}] 助手: {text}")
            elif role == "user":
                lines.append(f"[{i}] 用户: {text}")
            else:
                lines.append(f"[{i}] {role}: {text}")
        return "\n".join(lines)

    def _fallback_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Build a minimal structural summary without calling the LLM.
        Extracts user messages and tool names so the agent retains at least
        a coarse trace of what happened."""
        if not messages:
            return ""
        users = []
        tools = []
        for m in messages:
            r = m.get("role")
            if r == "user" and isinstance(m.get("content"), str):
                users.append(m["content"][:200])
            elif r == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                    if fn.get("name"):
                        tools.append(fn["name"])
        parts = []
        if users:
            parts.append("用户请求: " + " | ".join(users))
        if tools:
            parts.append("调用过的工具: " + ", ".join(tools))
        return "; ".join(parts) if parts else ""

    def _summarize_via_llm(self, prompt: str) -> str:
        """Call the LLM (non-interactive, no tools) to produce a summary.
        Uses a low temperature for determinism. Reuses chat_stream so the
        same client/transport is used."""
        messages = [
            {"role": "system", "content": "You are a precise conversation summarizer. 输出中文要点。"},
            {"role": "user", "content": prompt},
        ]
        out: List[str] = []
        for ev in self.llm.chat_stream(messages, tools=None, temperature=0.2):
            if ev.type == "content":
                out.append(ev.content)
            elif ev.type == "done":
                break
        return "".join(out).strip()

    def _append_summary_to_memory(self, summary: str, reason: str) -> None:
        """Append the shrink summary to workspace/memory.md with a timestamp
        header. Creates the file if absent. Uses atomic write (temp + rename)
        so concurrent shrinks or crashes can't corrupt the log."""
        try:
            memory_path = os.path.join(self.config.workspace, "memory.md")
            os.makedirs(self.config.workspace, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = (
                f"\n## 上下文摘要 - {now}\n"
                f"- 触发原因: {reason}\n\n"
                f"{summary}\n"
            )
            # Read existing content (if any) and append. Atomic write to avoid
            # corruption if the process is killed mid-write.
            existing = ""
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, "r", encoding="utf-8") as f:
                        existing = f.read()
                except OSError:
                    existing = ""
            tmp = memory_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(existing + entry)
            os.replace(tmp, memory_path)
        except Exception as e:  # pragma: no cover - defensive
            self._log(f"[context] failed to append memory.md: {e}")

    # ------------------------------------------------------------------
    def run(self, user_input: str) -> str:
        """Process one user turn. Returns the final assistant content."""
        self._turn_idx += 1
        # Invalidate per-turn cached prompt so it's rebuilt with current memory/skills.
        self._cached_prompt_turn = -1
        self._cached_prompt = ""
        self._log(f"USER: {user_input}")

        # skill suggestion (固化) - non-blocking
        try:
            suggested = self.skills.record_request(user_input)
            if suggested is not None:
                self.callbacks.on_skill_suggested(suggested)
        except Exception as e:  # pragma: no cover
            self._log(f"[skills] record_request error: {e}")

        self._history.append({"role": "user", "content": user_input})

        self._trim_history()

        final_content = ""
        # Use a manual iteration counter instead of `for i in range(...)` so
        # context-shrink retries don't consume an iteration. A shrink is a
        # recovery step, not a reasoning step, and shouldn't eat into the
        # agent's reasoning budget.
        iteration = 0
        shrink_count = 0
        while iteration < self.max_iterations:
            # --- Proactive context shrink: if estimated prompt size exceeds
            # context_shrink_ratio * max_context_tokens, summarize older
            # messages before calling the LLM. Avoids the round-trip of
            # sending an over-large prompt and getting back an error.
            if shrink_count < self._max_shrinks_per_run and self._should_shrink_context():
                pct = int(self.context_shrink_ratio * 100)
                shrunk = self._shrink_context(
                    reason=f"主动收缩: 估计上下文已达 {pct}% 阈值"
                )
                if shrunk:
                    shrink_count += 1
                    continue  # re-evaluate; don't burn an iteration on shrink

            iteration += 1
            messages = self._build_messages()
            self._log(f"[iter {iteration}] -> LLM ({self.config.model})")

            content_buf = []
            reasoning_buf = []
            tool_calls: List[Dict[str, Any]] = []
            usage: Dict[str, int] = {}
            t0 = time.time()
            ttfb = None  # time to first byte — set on first content/reasoning chunk
            completion_tokens = 0
            # Live token-speed every 0.3s during streaming.
            # Start at None (not 0.0) so the first chunk doesn't immediately
            # emit a misleading speed based on TTFB (which includes network
            # latency + model prefill, not generation speed).
            last_speed_emit: Optional[float] = None
            seen_done = False
            # Incremental token estimate: instead of joining the full
            # content_buf + reasoning_buf every 0.3s (O(N) join + O(N)
            # char scan, where N grows linearly per turn — 10K chars means
            # 10K iterations every 0.3s), we accumulate the token count
            # chunk-by-chunk. Each _estimate_tokens call is O(chunk_len)
            # which is tiny. The total is O(Σ chunk_len) = O(N) across the
            # whole turn, versus O(N²/0.3) for the naive approach.
            live_est_tokens = 0

            try:
                for ev in self.llm.chat_stream(
                    messages, tools=self.tools.schemas(), temperature=self.config.temperature
                ):
                    if ev.type == "content":
                        content_buf.append(ev.content)
                        self.callbacks.on_content(ev.content)
                        live_est_tokens += _estimate_tokens(ev.content)
                    elif ev.type == "reasoning":
                        reasoning_buf.append(ev.content)
                        self.callbacks.on_reasoning(ev.content)
                        live_est_tokens += _estimate_tokens(ev.content)
                    elif ev.type == "done":
                        seen_done = True
                        tool_calls = ev.tool_calls
                        usage = ev.usage
                        completion_tokens = usage.get("completion_tokens", 0)
                        continue  # final speed is emitted below; skip live emit
                    # Live token-speed every 0.3s during streaming.
                    # First chunk: record TTFB but don't emit yet (we need at
                    # least 0.3s of generation to compute a meaningful speed).
                    now = time.time()
                    if ttfb is None:
                        ttfb = now
                    if last_speed_emit is None:
                        last_speed_emit = now
                    elif now - last_speed_emit >= 0.3:
                        # Use generation time (now - ttfb), not total time
                        # (now - t0), to exclude TTFB from the speed calc.
                        gen_elapsed = max(now - ttfb, 1e-6)
                        if live_est_tokens > 0:
                            live_speed = live_est_tokens / gen_elapsed
                            self.callbacks.on_token_speed(
                                self._total_tokens + live_est_tokens, live_speed
                            )
                        last_speed_emit = now
            except Exception as e:
                err_str = str(e)
                # --- Reactive context shrink: if the LLM rejected the prompt
                # for being too long, shrink and retry the same iteration
                # instead of failing the whole turn.
                if (
                    self._is_context_too_long_error(err_str)
                    and shrink_count < self._max_shrinks_per_run
                ):
                    self._log(
                        f"[context] LLM rejected prompt as too long: {err_str[:200]}; "
                        f"shrinking and retrying (attempt {shrink_count + 1}/{self._max_shrinks_per_run})"
                    )
                    shrunk = self._shrink_context(reason="被动收缩: LLM context_length_exceeded")
                    if shrunk:
                        shrink_count += 1
                        # Don't increment iteration — retry the same turn.
                        iteration -= 1
                        continue
                self._log(f"[agent] LLM stream error: {err_str}")
                self.callbacks.on_error(err_str)
                self.callbacks.on_finished()
                return final_content or ""

            if not seen_done and (content_buf or reasoning_buf or tool_calls):
                # The stream ended without an explicit "done" event — likely a
                # network cut or server-side truncation. Log so it's visible.
                self._log("[agent] warning: LLM stream ended without 'done' event")

            # Total elapsed time (includes TTFB, used for total turn timing)
            elapsed = max(time.time() - t0, 1e-6)
            # Generation-only elapsed (excludes TTFB, used for speed calc).
            # If ttfb is None (no chunks arrived), fall back to elapsed.
            gen_elapsed = max((time.time() - ttfb), 1e-6) if ttfb is not None else elapsed

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
            # divided by generation-only time (excludes TTFB).
            # Fall back to completion_tokens if we somehow have no estimate.
            speed_tokens = turn_tokens if turn_tokens else completion_tokens
            speed = (speed_tokens / gen_elapsed) if speed_tokens else 0.0
            self.callbacks.on_token_speed(self._total_tokens, speed)
            self._log(
                f"[iter {iteration}] <- {len(''.join(content_buf))} content chars, "
                f"{len(reasoning_text)} reasoning chars (~{reasoning_tokens} tok), "
                f"{len(tool_calls)} tool_calls, usage={usage}, "
                f"{speed_tokens} tok / {gen_elapsed:.2f}s = {speed:.1f} tok/s"
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
                        self._tool_call_counter += 1
                        tc["id"] = f"call_{self._tool_call_counter}_{idx}"
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
                    # Notify UI of start (with raw args for visibility) before
                    # reporting the failure, so the UI shows a complete cycle.
                    self.callbacks.on_tool_start(name, {"_raw_arguments": raw_args})
                    res = ToolResult(False, error=str(e))
                    self.callbacks.on_tool_end(name, res)
                    msg = self._tool_message(tc, res)
                    self._history.append(msg)
                    continue

                tool = self.tools.get(name)

                needs_confirm = bool(tool and tool.should_confirm(args)) if tool else False
                self._log(f"[tool] {name}({raw_args}) needs_confirm={needs_confirm}")

                if tool is None:
                    # Unknown tool: still notify UI of start for a complete cycle.
                    self.callbacks.on_tool_start(name, args)
                    res = self.tools.execute(name, args)  # returns unknown error
                    self.callbacks.on_tool_end(name, res)
                    msg = self._tool_message(tc, res)
                    self._history.append(msg)
                    continue

                if needs_confirm:
                    confirm_msg = self._confirm_message(name, args)
                    if not self.callbacks.confirm(confirm_msg):
                        self._log(f"[tool] {name} DENIED by user")
                        from .tools import ToolResult
                        # Tool never actually started (user declined before
                        # invocation); only report the denial result. The UI
                        # intentionally does not receive on_tool_start here so
                        # that "never started" semantics are preserved.
                        res = ToolResult(False, error=f"User denied {name} ({confirm_msg})")
                        self.callbacks.on_tool_end(name, res)
                        msg = self._tool_message(tc, res)
                        self._history.append(msg)
                        continue

                self.callbacks.on_tool_start(name, args)
                res = self.tools.execute(name, args)
                self.callbacks.on_tool_end(name, res)
                self._log(f"[tool] {name} -> success={res.success} "
                          f"out_len={len(res.output)} err={(res.error or '')[:120]}")
                msg = self._tool_message(tc, res)
                self._history.append(msg)

        self._log("[agent] max iterations reached, stopping.")
        self.callbacks.on_finished()
        return final_content

    # ------------------------------------------------------------------
    def _build_messages(self) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        msgs.extend(self._history)
        return msgs

    def _system_prompt(self) -> str:
        """Build the full system prompt: base + memory + matched skill.

        Caches the prompt within a single agent turn (self._turn_idx) so that
        _estimate_history_tokens() and _build_messages() don't recompute it
        twice for the same turn.  Memory and skill match results are stable
        within one LLM call cycle.
        """
        if (hasattr(self, "_cached_prompt_turn")
                and self._cached_prompt_turn == self._turn_idx
                and hasattr(self, "_cached_prompt")):
            return self._cached_prompt

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
            # Find the latest user message — skills match user intents, not
            # assistant/tool messages. self._history[-1] after the first turn
            # is an assistant message whose content may be None.
            user_text = ""
            for msg in reversed(self._history):
                if msg.get("role") == "user" and msg.get("content"):
                    user_text = msg["content"]
                    break
            matched = self.skills.match(user_text) if user_text else None
            if matched:
                base += f"\n\n# Active skill: {matched.name}\n{matched.prompt}"
        except Exception:  # pragma: no cover
            pass

        # Dynamically inject knowledge base status so the agent knows RAG
        # tools are available and whether the KB has indexed content.  Without
        # this the agent has no way to discover that a local KB exists and
        # will never proactively call rag_search / rag_status.
        try:
            rag_engine = self.tools.get_rag_engine() if self.tools else None
            if rag_engine is not None:
                status = rag_engine.status()
                chunks = status.get("chunks_stored", 0)
                sources = status.get("sources", [])
                if chunks > 0:
                    source_list = ", ".join(
                        os.path.basename(s) for s in sources[:10]
                    ) + ("..." if len(sources) > 10 else "")
                    base += (
                        f"\n\n# Local knowledge base (ACTIVE)\n"
                        f"A local knowledge base is available with {chunks} indexed "
                        f"text chunks from {len(sources)} file(s): {source_list}\n"
                        f"You have the tools `rag_search`, `rag_status`, and "
                        f"`rag_ingest` to interact with it.  When the user asks a "
                        f"question that could be answered from these documents, "
                        f"MUST call `rag_search` first before answering.\n"
                        f"知识库已就绪，包含 {chunks} 个文本切片，来自 {len(sources)} 个文件。"
                        f"回答与用户文档相关的问题时，必须先调用 `rag_search` 检索相关内容。"
                    )
                else:
                    base += (
                        f"\n\n# Local knowledge base (EMPTY)\n"
                        f"A knowledge base is configured but has no indexed content yet. "
                        f"Use `rag_ingest` to index documents from the knowledge base "
                        f"directory, then use `rag_search` to query them.\n"
                        f"知识库已配置但尚未索引任何文档。可使用 `rag_ingest` 索引文档后，"
                        f"再用 `rag_search` 进行检索。"
                    )
        except Exception:  # pragma: no cover
            pass

        self._cached_prompt_turn = self._turn_idx
        self._cached_prompt = base
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
