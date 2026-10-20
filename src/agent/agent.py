"""The agent: reasoning loop, tool dispatch, streaming, confirmation, skills."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol

from .config import AgentConfig
from .llm_client import LLMClient, StreamEvent
from .skills import SkillManager
from .tools import ToolRegistry, parse_args

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callbacks protocol
# ---------------------------------------------------------------------------


class AgentCallbacks:
    """Override these methods to drive the UI. Default impls are no-ops."""

    def is_cancelled(self) -> bool:
        """协作式取消：run() 在循环边界/流式事件/工具执行前轮询。

        默认 False（本地单测无取消概念）。UI 的 BridgeCallbacks 返回
        「终止对话」按钮设置的标志——run() 必须能及时看到它，否则取消只在
        下一个恰好需要确认的工具处生效，其余场景会跑满 max_iterations。
        """
        return False

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
        """默认**拒绝**而非放行。

        ``confirm`` 用于 code_run / shell_run / MCP 这类可影响系统的操作。
        未显式实现回调的场景（脚本、无头测试、自动化调用）下返回 True 等于
        把「确认」这一步整个架空 —— 模型生成的任意命令都会被执行。安全侧是
        拒绝：需要放行时由调用方显式实现。
        """
        return False

    def ask_user(self, prompt: str) -> Optional[str]:
        return None

    def on_finished(self) -> None: ...


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are base-agent, a local autonomous agent. Complete the user's task by reasoning, planning, and calling tools.

Rules:
1. Non-trivial task → first call `task_plan` (create: goal + 3-8 concrete steps). After each tool result, mark progress (`task_plan` update) and decide the next step; add steps as you learn more. When every step is done, call `task_plan` with op='clear' to remove the finished plan; recreate it for a new task.
2. After each tool result, reflect: did it move you toward the goal? Decide the next step before acting.
3. Prefer the least destructive path. Risky/irreversible actions need confirmation.
4. Missing info → `ask_user`, never guess.
5. Recall prior context with `memory_search` before answering about the user or past work. Use `work_memory` only for multi-step notes (intermediate results, decisions, preferences).
6. Complex task → `skill_semantic_search` first; if it finds nothing, try `skill_search` (they index different directories). Then `skill_load` and follow it. No match → proceed.
7. User names a book/document → `rag_outline` for structure, then `rag_search` (with source) for details.
8. Semantic search queries (`memory_search`/`skill_semantic_search`/`rag_search`): use the user's own natural-language phrasing. Never translate, transliterate to pinyin, or reduce to bare keywords — these are vector searches and mangled queries silently return nothing.
9. On tool ERROR: fix the arguments and retry. Two identical failures in a row → change approach or `ask_user`. Never loop on the same failing call.
10. Reply concisely in plain text when done.

Paths are relative to the workspace. `code_run`/`shell_run` run without confirmation unless the command/code looks destructive (system files, privilege escalation, system stability); external MCP tools always ask first."""

# ---------------------------------------------------------------------------
# Token estimation: approximate token count from text length.
# Used for real-time speed display during reasoning streaming (where we don't
# yet have the API's usage figure) and as a fallback when the API's usage
# omits reasoning tokens entirely.
# ---------------------------------------------------------------------------

# CJK 码点 → None 的删除表，供 str.translate 一次性统计 CJK 字符数（C 层实现）。
# 覆盖统一表意文字、扩展 A、CJK 符号/日文假名、全角字符。
_CJK_DELETE_TABLE = {
    cp: None
    for cp in list(range(0x4E00, 0xA000))      # CJK 统一表意文字
    + list(range(0x3400, 0x4DC0))              # CJK 扩展 A
    + list(range(0x3000, 0x3100))              # CJK 符号 + 日文假名
    + list(range(0xFF00, 0xFFF0))              # 全角字符
}


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
    # much faster than a Python-level per-character loop for long text
    # (旧实现是纯 Python 逐字符循环，长文本下明显偏慢，与上面的注释不符)。
    # translate() 删除的是 CJK 字符，所以剩下的是非 CJK —— 用总长度相减
    # 得到 CJK 数量（比逐字符 Python 循环快一个数量级）。
    cjk = len(text) - len(text.translate(_CJK_DELETE_TABLE))
    other = len(text) - cjk
    # CJK: ~1.5 chars/token; ASCII/other: ~4 chars/token
    return max(1, int(cjk / 1.5 + other / 4.0))


# 系统提示动态注入段的上限。
# 这些段不在上下文收缩（_shrink_context 只压 _history）的作用范围内，
# 若不加限制会随会话时长单调膨胀，最终挤爆上下文窗口。
_MAX_INJECTED_FACTS = 20        # 当天事实最多注入条数
_MAX_FACTS_CHARS = 2000         # 事实段总字符上限
_MAX_FACT_CHARS = 120           # 单条事实字符上限
_MAX_SKILL_PROMPT_CHARS = 1500  # 命中技能 prompt 的字符上限
_MAX_DAILY_FACTS = 200          # 当天事实桶的持久化条数上限

# ---------------------------------------------------------------------------
# 反思/防停滞机制（run() 内的看门狗）。
# 目标：不让任务「半路跑丢或直接就停」。四道防线：
#   1) 空响应（无内容无工具调用）→ 撤掉空消息、注入反思提醒后重试；
#   2) 同一 (工具, 参数) 连续失败 → 在工具结果里附加重试指引；
#   3) 连续多次工具失败（不管参数是否相同）→ 注入整段反思提醒；
#   4) 迭代预算耗尽 → 强制做一次「无工具」收尾总结，绝不静默终止。
# 每类提醒每个 run() 有次数上限，防止提醒本身变成新的循环。
# ---------------------------------------------------------------------------

_MAX_EMPTY_NUDGES = 2                 # 空响应重试次数
_MAX_REFLECTION_NUDGES = 2            # 连续失败反思注入次数
_IDENTICAL_FAIL_GUIDANCE_AT = 2       # 相同调用失败几次后附加重试指引
_CONSECUTIVE_FAIL_NUDGE_AT = 4        # 连续失败几次后注入反思提醒

_EMPTY_RESPONSE_NUDGE = (
    "[系统提示] 你上一条回复是空的。请反思当前任务目标：如果还没完成，"
    "调用合适的工具继续下一步；如果已完成，直接给出最终答复。不要输出空回复。"
)

_FAILURE_REFLECTION_NUDGE = (
    "[系统提示] 已连续多次工具调用失败。请先停下反思：当前目标是什么？"
    "已经尝试了什么？失败原因可能是什么？然后换一种方法或修正参数继续；"
    "缺少必要信息时用 ask_user 向用户确认，不要继续盲目重试。"
)

_WRAPUP_NUDGE = (
    "[系统提示] 你已达到本轮工具调用次数上限，不能再调用工具。"
    "请基于当前进度做最终答复：总结已完成的工作与关键结果，说明还剩哪些"
    "未完成的步骤、下一步该做什么，方便用户后续继续。"
)


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
        aux_llm: Optional["Any"] = None,
    ):
        self.llm = llm  # object with chat_stream(messages, tools, temperature)
        # 辅助模型（#140）：轻量子任务（上下文摘要）专用，与主模型分离，
        # 避免一次摘要把主模型的推理队列占住。可为 None（退化为用主模型）。
        # 同样由 UI 持有并负责 close()，agent 不关它。
        self.aux_llm = aux_llm
        self.config = config
        self.callbacks = callbacks
        self.tools = tool_registry or ToolRegistry(
            config=config, callbacks=callbacks,
            # 用 getter 而非快照：辅助模型可能在 agent 构造后才加载/切换，必须
            # 在每次 RAG 检索时读「当前」实例（见 RagSearchTool._condense_with_aux）。
            aux_llm_getter=lambda: self.aux_llm,
        )
        self.skills = skills or SkillManager(
            path=os.path.join(config.workspace, ".agent", "skills.json"),
            skills_dir=os.path.join(config.workspace, ".agent", "skills"),
        )
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
        # tools schema 的 token 估算缓存（见 _tool_schema_tokens）。
        self._tool_schema_tokens_cache: Optional[int] = None
        self._tool_schema_version: int = -2  # -2 = 从未计算过
        # _system_prompt 缓存附带的工具状态版本号（见 _system_prompt）。
        self._cached_prompt_state = -2
        # UI 回调异常计数（见 _safe_callback）：只记录前几条，避免刷屏。
        self._callback_errors = 0

        # 事实抽取的防堆叠锁。必须用 RLock：_extract_facts_sync 持锁期间会调
        # _get_extract_llm()，后者也要 acquire 同一把锁（普通 Lock 会自锁死，
        # 与 vector_store 的 _documents_lock 是同一类问题）。另外必须在
        # __init__ 里创建——懒初始化是 check-then-act 竞态，两个抽取线程可能
        # 各自建一把锁并各自成功 acquire，防堆叠形同虚设。
        self._fact_extract_lock = threading.RLock()
        # close() 后置位：抽取守护线程不被跟踪/join，看到它就不再新建
        # 任何资源（专用 LLM client、LongTermMemory），否则无人回收。
        self._closed = False

    # ------------------------------------------------------------------
    def _invalidate_local_kv_cache(self) -> None:
        """让本地后端丢弃跨轮 KV 缓存（历史被改写/清空时调用）。

        本地后端（MLX）只在「新 prompt 完整包含上一轮序列」时复用 KV 缓存。
        历史一旦被改写（上下文收缩、reset、编辑历史），前缀必然对不齐——
        后端自己也会检测到并重建，这里主动通知更直接，也省一次无效比对。
        对没有该方法的后端（HTTP 客户端）是 no-op。
        """
        invalidate = getattr(self.llm, "invalidate_kv_cache", None)
        if callable(invalidate):
            try:
                invalidate()
            except Exception:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._history = []
        self._total_tokens = 0
        self._tool_call_counter = 0
        # Invalidate cached prompt
        self._cached_prompt_turn = -1
        self._cached_prompt = ""
        # 「清空对话」= 会话状态归零：任务计划属于当前会话，一并清除，
        # 否则旧计划会在新会话的系统提示里残留、误导模型。
        try:
            from .planner import clear_plan
            clear_plan(self.config.workspace)
        except Exception:  # pragma: no cover - defensive
            pass
        # 历史清空 → KV 前缀必然对不齐，主动失效本地缓存。
        self._invalidate_local_kv_cache()

    def close(self) -> None:
        """Release the dedicated fact-extraction LLM client's connection pool.

        ``self.llm`` is owned by the UI (which closes it separately); we only
        own ``_extract_llm`` (a lazily-built independent LLMClient created to
        avoid racing the agent's shared httpx.Client from the extraction thread).

        Also latches ``_closed``: the extraction daemon thread is not tracked
        or joined, so it can otherwise create ``_extract_llm`` (and a fresh
        LongTermMemory) *after* this point — and nothing would ever close them.
        """
        # 先置标志再关：抽取线程看到 _closed 就不再新建任何资源。
        self._closed = True
        ext = getattr(self, "_extract_llm", None)
        if ext is not None and ext is not self.llm:
            try:
                ext.close()
            except Exception:
                pass
            self._extract_llm = None

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
        # 边界：尾部若连续若干条都是 tool 消息，cut 会一路递增把历史**清空**
        # （max_history 被设得很小时可达）。至少保留最后一条，否则模型拿不到
        # 任何上下文，部分服务端还会因没有 user 消息直接报错。
        while (cut < len(self._history) - 1
               and self._history[cut].get("role") == "tool"):
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
        """Estimate the token count of the full prompt (system + history + tools).

        Always does a full O(n) scan of _history. The history list is
        typically short (<= max_history messages), so the cost is negligible
        compared to the complexity of maintaining an incremental counter
        that can get out of sync when _history is modified directly.

        Tools must be counted: every request actually sends
        ``tools=self.tools.schemas()``, and with many MCP tools the schema
        alone can be several thousand tokens. Excluding it systematically
        underestimates the prompt, so proactive shrink never triggers and the
        only protection left is the (capped) reactive shrink after the API
        already rejected the request.
        """
        system_tokens = _estimate_tokens(self._system_prompt())
        history_tokens = sum(self._msg_token_count(msg) for msg in self._history)
        tools_tokens = self._tool_schema_tokens()
        return system_tokens + history_tokens + tools_tokens

    def _tool_schema_tokens(self) -> int:
        """Cached token estimate of the tools schema (JSON sent every request).

        缓存键是 **schema 版本号**（工具表变动即递增），而不是历史长度：

        * 旧实现以 ``len(self._history)`` 为键 —— 每追加一条消息键就失配，
          于是每轮迭代都要把整套 schema 重新 ``json.dumps`` 一遍再全量
          扫描字符（MCP 工具多时是数千 token 的纯浪费）。
        * 中间版本用 schema 内容的 md5 指纹做键 —— 指纹本身也要每轮
          ``json.dumps`` 一遍（缓存命中时同样跑），省了估算却没省序列化。
          版本号由 ``ToolRegistry._invalidate_schema_cache()`` 维护，
          工具表不变就完全跳过序列化。
        * 反之，历史长度不变而工具集变化（动态注册/注销 MCP 工具）时
          旧键又永不失效，估算失真。版本号两种场景都覆盖。
        """
        try:
            version = getattr(self.tools, "schema_version", -1) if self.tools else -1
        except Exception:
            version = -1
        cached = self._tool_schema_tokens_cache
        if cached is not None and self._tool_schema_version == version:
            return cached
        try:
            schemas = self.tools.schemas() if self.tools else []
        except Exception:
            schemas = []
        try:
            text = json.dumps(schemas, ensure_ascii=False, sort_keys=True) if schemas else ""
            est = _estimate_tokens(text) if text else 0
        except Exception:
            est = 0
        self._tool_schema_tokens_cache = est
        self._tool_schema_version = version
        return est

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
                # old_messages 已抽空却仍是 tool 消息 → 前面不可能再有对应的
                # assistant(tool_calls)。此时必须**丢弃**这条孤儿 tool 消息，
                # 否则收缩后的历史以裸 tool 开头，API 会以 400 拒绝整轮请求。
                recent_messages.pop(0)
                continue
            recent_messages.insert(0, old_messages.pop())

        transcript = self._format_messages_for_summary(old_messages)
        summary_prompt = (
            "把下面的对话压缩成要点摘要，只保留：\n"
            "1. 用户的核心需求与当前目标\n"
            "2. 已做的决定及原因\n"
            "3. 已完成的步骤与结果（含工具关键输出）\n"
            "4. 未完成的子任务与下一步\n"
            "5. 具体路径、数值、标识符、配置值、偏好、URL\n"
            "6. 错误/警告等需注意的事项\n\n"
            "要求：只输出要点列表，不要前言或解释；不要复述对话；不要编造。"
            "不重要的信息直接省略。\n\n"
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
        # 历史被整体改写，本地后端的 KV 前缀必然对不齐。虽然后端会自己检测到
        # 并重建，这里主动失效更直接（也顺便释放显存里的缓存）。
        self._invalidate_local_kv_cache()
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
        same client/transport is used.

        优先走辅助模型（#140）：摘要是轻量子任务，用小模型更快，也不会占住
        主模型的推理队列。辅助模型缺失或不可用时退回主模型——摘要失败最坏
        只是上下文不收缩，不能让整轮对话崩掉。
        """
        messages = [
            {"role": "system", "content": "你是精确的对话摘要器，用中文输出要点列表，不加前言和解释。"},
            {"role": "user", "content": prompt},
        ]
        target = self.aux_llm if self.aux_llm is not None else self.llm
        out: List[str] = []
        try:
            for ev in target.chat_stream(messages, tools=None, temperature=0.2):
                if ev.type == "content":
                    out.append(ev.content)
                elif ev.type == "done":
                    break
        except Exception as e:
            if target is self.llm:
                raise
            self._log(f"[context] 辅助模型摘要失败，回退主模型: {e}")
            target = self.llm
            out = []
            for ev in target.chat_stream(messages, tools=None, temperature=0.2):
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

    def _maybe_extract_facts_async(self, assistant_reply: str) -> None:
        """Launch fact extraction in a daemon thread.

        The extraction calls the LLM (200ms-2s) and we don't want to delay
        on_finished() — the user should get their reply immediately.  The
        thread is a daemon so it won't block process exit if the user
        closes the app right after a reply.

        We capture the *recent conversation window* on the main thread before
        spawning the worker (the worker must not read self._history — by the
        time it runs, the main thread may have already started the next turn
        and mutated/shrunk the list).  A multi-turn window (instead of just
        the last exchange) lets the extractor resolve pronouns and project
        references ("这个项目用 X 吗") that a single isolated turn cannot.
        """
        # Collect the last few user/assistant exchanges for context.
        recent: List[str] = []
        for msg in reversed(self._history):
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                recent.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
                if len(recent) >= 6:
                    break
        recent.reverse()
        if not recent:
            return
        # 跳过无实质内容的对话：最新一条 user 消息过短（如「read data.txt」、
        # 「write and read back」这类单句指令）说明没有可抽取的持久事实，
        # 抽取只会浪费一次 LLM 调用。这与旧行为保持一致（user/assistant 均
        # 需 >= 20 字符才抽取）。
        last_user = ""
        for msg in reversed(self._history):
            if msg.get("role") == "user" and msg.get("content"):
                last_user = msg["content"].strip()
                break
        if len(last_user) < 20:
            return
        if len((assistant_reply or "").strip()) < 20:
            return
        # The final reply is already in _history as the last assistant msg;
        # if not (edge case), append it explicitly.
        conversation = "\n\n".join(recent)

        import threading
        thread = threading.Thread(
            target=self._extract_facts_sync,
            args=(conversation,),
            daemon=True,
            name="fact-extraction",
        )
        thread.start()

    def _extract_facts_sync(self, conversation: str) -> None:
        """Best-effort: use the LLM to extract entities/relations/observations
        from a conversation fragment and persist them to the knowledge
        graph.  Runs on a daemon thread — any failure is logged and swallowed.

        Concurrency: extraction uses a *dedicated* LLM client (its own
        ``httpx.Client``), never the agent's ``self.llm``.  The agent's
        ``httpx.Client`` is NOT thread-safe — if the daemon thread reused it
        while the main agent thread streamed the next turn, the shared
        connection pool would race and fail the request.  A ``_fact_extract_lock``
        further prevents multiple extractions from piling up.
        """
        lock = self._fact_extract_lock
        if not lock.acquire(blocking=False):
            # A previous extraction is still running; skip to avoid piling up.
            return
        try:
            # agent 已被关闭（应用配置/关窗）→ 直接放弃。否则下面的
            # _get_long_memory 会在 tools.shutdown() **之后**重建
            # LongTermMemory（GraphMemoryStore 懒加载 ~137MB ONNX +
            # lancedb.connect），而这个新实例再也没有人负责回收。
            if getattr(self, "_closed", False):
                return
            from .graph_memory import extract_facts_via_llm
            extract_llm = self._get_extract_llm()
            if extract_llm is None:
                # 拿不到专用 client（锁被占 / 构造失败 / 已关闭）：绝不降级到
                # self.llm —— 那会让守护线程与主线程的流式请求共用连接池。
                self._log("[memory] 本轮跳过事实抽取（专用 LLM 客户端不可用）")
                return
            data = extract_facts_via_llm(extract_llm, conversation)

            entities = data.get("entities", [])
            relations = data.get("relations", [])
            if not entities and not relations:
                self._log("[memory] 本轮无可持久化事实（抽取为空或 LLM 返回空图）")
                return

            from .tools.memory import _get_long_memory
            lm = _get_long_memory(self.tools, self.config)
            if lm is None:
                self._log("[memory] 本轮跳过事实持久化（长期记忆已释放）")
                return
            g = lm.graph
            added = 0
            # 收集本次抽取到的所有 observation（实体观察），同时用于
            # 长期记忆（graph）与短期记忆（当天事实）两份写入。
            all_observations: List[str] = []
            # 批量写入：逐个调 add_observations/create_relation 会让每个元素
            # 各触发一次整图 JSON 序列化 + 原子写盘（≤8 次）。batch_apply 在
            # 一把锁内改完内存、只落盘一次。
            added = g.batch_apply(entities, relations)
            for ent in entities:
                obs = [o.strip() for o in (ent or {}).get("observations", []) if o and o.strip()]
                all_observations.extend(obs)
            if added:
                self._log(f"[memory] 抽取 {added} 条 observation + {len(relations)} 条 relation 存入长期记忆")
            # 同时追加到短期记忆（当天桶）：一次抽取，两份写入，避免重复抽取。
            if all_observations:
                self._append_daily_facts(all_observations)
        except Exception as e:  # pragma: no cover — best-effort
            self._log(f"[memory] 事实抽取失败: {e}")
        finally:
            lock.release()

    def _get_extract_llm(self):
        """Return a dedicated LLM client for fact extraction.

        Fact extraction runs on a daemon thread concurrently with the agent's
        next turn.  ``LLMClient`` wraps a single ``httpx.Client`` that is NOT
        thread-safe, so extraction must not reuse ``self.llm``.  We lazily
        build one independent ``LLMClient`` (its own connection pool) and
        cache it on the instance.

        For non-``LLMClient`` llm objects (test mocks, scripted LLMs) we return
        ``self.llm`` unchanged — those have no shared socket state.
        """
        cached = getattr(self, "_extract_llm", None)
        if cached is not None:
            return cached

        from .llm_client import LLMClient
        if not isinstance(self.llm, LLMClient):
            # Non-LLMClient (test mock / scripted): reuse self.llm (no shared
            # socket state to race).
            return self.llm

        # 懒初始化会被两个线程并发触发：confirm 弹窗的
        # _describe_code_purpose（worker 线程）与抽取守护线程。不加锁会各自
        # 建一个 LLMClient，后写者覆盖 _extract_llm，被覆盖一方的 httpx 连接
        # 池永不关闭（close() 只关属性里最后一个）。
        #
        # 但这里绝不能阻塞等锁：抽取线程会**在整个 LLM 调用期间**持有这把锁
        # （见 _extract_facts_sync），阻塞等待会让 agent 线程卡住一个完整
        # 请求（最长 read timeout）才发出确认弹窗——与「不延迟弹窗」的意图
        # 正好相反。
        #
        # 取不到锁 / 构造失败时**绝不能退回 self.llm**：那正是本函数存在的
        # 理由所要避免的情形（守护线程与 agent 主线程的流式请求共用同一个
        # httpx.Client —— 连接池竞争会让两边的流交错损坏）。此时返回 None，
        # 由调用方放弃本次抽取（事实抽取是 best-effort，丢一次远好于污染
        # 正在进行的对话）。
        if getattr(self, "_closed", False):
            return None
        if not self._fact_extract_lock.acquire(blocking=False):
            return None
        try:
            cached = getattr(self, "_extract_llm", None)
            if cached is not None:
                return cached
            try:
                self._extract_llm = LLMClient(
                    base_url=self.llm.base_url,
                    api_key=self.llm.api_key,
                    model=self.llm.model,
                    timeout=self.llm.timeout,
                    max_tokens=self.llm.max_tokens,
                    top_p=self.llm.top_p,
                    min_p=self.llm.min_p,
                    top_k=self.llm.top_k,
                    repetition_penalty=self.llm.repetition_penalty,
                )
                return self._extract_llm
            except Exception as e:  # pragma: no cover — best-effort
                logger.warning("[memory] 抽取专用 LLM 客户端创建失败: %s", e)
        finally:
            self._fact_extract_lock.release()
        return None

    def _append_daily_facts(self, facts: List[str]) -> None:
        """Append auto-extracted facts to work memory under today's bucket.

        短期记忆只保留当天的自动抽取事实；历史事实已在长期记忆 graph 中
        持久化（供 memory_search 检索），无需在短期记忆里重复保留。对话时
        只把当天事实注入上下文（见 _system_prompt），长期记忆则靠检索。

        Key 约定：``__auto_facts__`` → {"date": "YYYY-MM-DD", "facts": [...]}
        使用保留前缀，避免与模型手动 work_memory set 的 key 冲突。
        """
        import datetime
        from .tools.memory import _work_memory
        try:
            today = datetime.date.today().isoformat()
            wm = _work_memory(self.config)
            existing = wm.get("__auto_facts__") or ""
            # 解析现有当天事实（若跨天则视为过期，重置为空）。
            bucket_date, bucket_facts = self._parse_daily_facts(existing)
            if bucket_date != today:
                bucket_facts = []
            # 去重追加（保持原有顺序，新事实排后）。
            seen = set(bucket_facts)
            for f in facts:
                if f and f not in seen:
                    bucket_facts.append(f)
                    seen.add(f)
            # 桶上限：事实每轮累积，不设上限会同时撑爆 work_memory.json
            # 与注入的系统提示（注入侧另有 _MAX_INJECTED_FACTS 二次兜底）。
            # 超限时丢弃最旧的，保留最近的。
            if len(bucket_facts) > _MAX_DAILY_FACTS:
                bucket_facts = bucket_facts[-_MAX_DAILY_FACTS:]
            wm.set(
                "__auto_facts__",
                json.dumps({"date": today, "facts": bucket_facts}, ensure_ascii=False),
            )
        except Exception as e:  # pragma: no cover — best-effort
            self._log(f"[memory] 短期记忆追加失败: {e}")

    @staticmethod
    def _parse_daily_facts(raw: str):
        """Parse the __auto_facts__ work-memory value → (date, [facts]).

        Returns ("", []) on malformed/empty input so callers treat it as
        "no valid today-bucket" and reset.
        """
        if not raw:
            return "", []
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return "", []
            date = data.get("date", "")
            facts = data.get("facts", [])
            if not isinstance(date, str) or not isinstance(facts, list):
                return "", []
            return date, [f for f in facts if isinstance(f, str) and f.strip()]
        except (json.JSONDecodeError, TypeError):
            return "", []

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
        # --- 反思/防停滞看门狗状态（每个 run() 重置）---
        empty_nudges = 0            # 空响应已重试次数
        reflection_nudges = 0       # 连续失败反思已注入次数
        identical_fail_streak = 0   # 相同 (工具, 参数) 连续失败次数
        consecutive_failures = 0    # 连续工具失败次数（不管参数）
        last_tool_sig = None        # 上一次工具调用签名 (name, raw_args)

        while iteration < self.max_iterations:
            # 协作式取消：每轮迭代开始时检查。旧实现只在 confirm/ask_user 里
            # 检查 _cancelled——若模型接下来调用的都是无需确认的工具，取消
            # 根本不生效，会一直跑满 max_iterations。
            if self._safe_is_cancelled():
                self._log("[agent] 收到取消请求，终止本轮对话")
                self._safe_callback("on_finished")
                return final_content

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
            # 服务端的结束原因（"stop" / "length" / "tool_calls"）。
            finish_reason = ""
            # 流式过程中用户点了「终止对话」→ 中断读取并直接结束本轮。
            cancelled_mid_stream = False
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
                    # 流式期间也轮询取消（bool 检查，代价可忽略），否则长回复
                    # 期间点「终止对话」要等整个回复生成完才生效。
                    if self._safe_is_cancelled():
                        self._log("[agent] 取消：中断流式读取")
                        cancelled_mid_stream = True
                        # 生成被用户掐断：后端记录的是不完整的 token 序列，
                        # 下一轮前缀必然对不齐（而且我们也不会把这次残输出
                        # 存进历史）。主动失效，省一次无效比对。
                        self._invalidate_local_kv_cache()
                        break
                    if ev.type == "content":
                        content_buf.append(ev.content)
                        self._safe_callback("on_content", ev.content)
                        live_est_tokens += _estimate_tokens(ev.content)
                    elif ev.type == "reasoning":
                        reasoning_buf.append(ev.content)
                        self._safe_callback("on_reasoning", ev.content)
                        live_est_tokens += _estimate_tokens(ev.content)
                    elif ev.type == "done":
                        seen_done = True
                        tool_calls = ev.tool_calls
                        usage = ev.usage
                        finish_reason = getattr(ev, "finish_reason", "") or ""
                        # 网关可能返回 null → `or 0` 归一，避免后续算术抛 TypeError
                        # 并被误报成「LLM 流错误」。
                        completion_tokens = usage.get("completion_tokens") or 0
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
                            # 必须隔离：这个裸调用位于外层 try 内，一次 UI 异常
                            # 会被当成「LLM 流错误」上报，并把本轮已产出的全部
                            # 内容丢掉（assistant 消息还没写入历史）。
                            self._safe_callback(
                                "on_token_speed",
                                self._total_tokens + live_est_tokens, live_speed,
                            )
                        last_speed_emit = now
            except Exception as e:
                err_str = str(e)
                # Friendly message for streaming read timeouts (the most common
                # cause of "reply stops halfway and never resumes").
                if not err_str:
                    import httpx
                    if isinstance(e, httpx.ReadTimeout):
                        err_str = "模型响应超时（读取中断），请重试"
                    elif isinstance(e, httpx.TimeoutException):
                        err_str = "模型请求超时，请重试"
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
                # on_error 也要隔离：它一旦抛出，下面的 on_finished 就永远
                # 不会执行，UI 会一直停在忙碌状态（没有任何报错可见）。
                self._safe_callback("on_error", err_str)
                self._safe_callback("on_finished")
                return final_content or ""

            if cancelled_mid_stream:
                # 流式期间被取消：不把半截回复当完整回答记入历史（UI 已展示了
                # 内容）。直接结束本轮；用户下一条消息会以干净的状态开始。
                self._safe_callback("on_finished")
                return final_content

            if not seen_done and (content_buf or reasoning_buf):
                # The stream ended without an explicit "done" event — likely a
                # network cut or server-side truncation. Log so it's visible.
                # （tool_calls 只在 "done" 分支赋值，此处恒为 []，不应参与判断。）
                self._log(
                    "[agent] warning: LLM stream ended without 'done' event — "
                    "回复可能被截断"
                )
            if finish_reason == "length":
                # 被 max_tokens 截断：回复是半截的。必须让用户可见，否则
                # 一个被截断的答案会被当成完整结论使用。
                self._log(
                    "[agent] warning: 本轮回复被 max_tokens 截断 "
                    "(finish_reason=length)，内容可能不完整"
                )
                self._safe_callback(
                    "on_log", "⚠ 本轮回复达到 max_tokens 上限被截断，可能不完整"
                )

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

            # Total tokens: use the API's total if it's non-zero and includes
            # reasoning (i.e. api_reasoning_tokens > 0 or no reasoning text was
            # streamed). Otherwise add our estimate on top.
            # 部分网关/代理会返回 null 字段；用 `or 0` 归一，否则 `None > 0`
            # 抛 TypeError，而这行在 try 保护之外会让整轮 run() 直接崩溃。
            api_total = usage.get("total_tokens") or 0
            if api_total > 0 and (api_reasoning_tokens or not reasoning_text):
                # API total is trustworthy
                self._total_tokens += api_total
            else:
                # API total is missing or doesn't include reasoning
                self._total_tokens += api_total + extra_reasoning_tokens

            if usage:
                self._safe_callback("on_usage", usage)
            # Final speed: use ONLY the completion token count (output tokens)
            # divided by generation-only time (excludes TTFB).
            #
            # BUG FIX: previously used `turn_tokens` which includes
            # prompt_tokens (input).  Dividing total_tokens by generation
            # time produced wildly inflated speeds (e.g. 8000 prompt + 300
            # completion tokens / 20s = 415 t/s instead of 15 t/s).
            #
            # If the API provides reasoning_tokens separately and they're
            # NOT included in completion_tokens, add them.  If they ARE
            # included (api_reasoning_tokens > 0), completion_tokens already
            # covers everything.
            if api_reasoning_tokens:
                speed_tokens = completion_tokens  # already includes reasoning
            else:
                speed_tokens = completion_tokens + extra_reasoning_tokens
            speed = (speed_tokens / gen_elapsed) if speed_tokens else 0.0
            self._safe_callback("on_token_speed", self._total_tokens, speed)
            self._log(
                f"[iter {iteration}] <- {len(''.join(content_buf))} content chars, "
                f"{len(reasoning_text)} reasoning chars (~{reasoning_tokens} tok), "
                f"{len(tool_calls)} tool_calls, usage={usage}, "
                f"{speed_tokens} completion_tok / {gen_elapsed:.2f}s = {speed:.1f} tok/s"
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
            # 原始输出镜像（仅本地 MLX 后端提供，含工具调用 XML）。
            #
            # 为什么需要：本地后端的 KV 前缀复用要求历史渲染与上一轮生成序列
            # 逐 token 一致。但结构化历史是「解析后」的——think 标签被剥掉、
            # 工具调用被转成 JSON，重新渲染必然和模型当初生成的不一样，前缀
            # 对不齐，每轮仍要重算整段工具 schema 的 prefill（M1 上 ~15s）。
            # 存一份原文给本地后端重建忠实序列用。
            #
            # 这是「双视图」：content 保持解析后的形状供 HTTP 后端、记忆抽取、
            # 上下文收缩与 UI 使用；raw_content 只有本地后端的 _template_messages
            # 会读。HTTP 后端拿不到该属性，历史格式对它完全不变。
            raw_output = getattr(self.llm, "last_raw_output", "")
            if raw_output:
                assistant_msg["raw_content"] = raw_output
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
            # 只在有实际文本时更新 final_content：带 tool_calls 的轮次
            # content 为 None，无条件覆盖会把上一轮的最终答复抹成 ""，
            # 触及 max_iterations 时用户会看到一个空回复。
            if assistant_msg.get("content"):
                final_content = assistant_msg["content"]

            assistant_has_text = bool((assistant_msg.get("content") or "").strip())
            if not tool_calls and not assistant_has_text:
                # 防线 1：空响应（无文本、无工具调用）。模型可能卡住或"不知道
                # 做什么就什么都不做"。处理：
                #   a) 从历史撤掉这条空 assistant 消息——content=None 且无
                #      tool_calls 的消息部分服务端会以 400 拒绝，留着还会污染
                #      下一轮请求；
                #   b) 注入反思提醒后重试（有次数上限），而不是静默结束；
                #   c) 重试用尽 → 带着已有内容正常收尾，不让用户干等。
                self._history.pop()
                if empty_nudges < _MAX_EMPTY_NUDGES and not self._safe_is_cancelled():
                    empty_nudges += 1
                    self._log(
                        f"[agent] 模型返回空响应，注入反思提醒重试 "
                        f"({empty_nudges}/{_MAX_EMPTY_NUDGES})"
                    )
                    self._history.append({"role": "user", "content": _EMPTY_RESPONSE_NUDGE})
                    continue
                self._log("[agent] 空响应重试用尽，结束本轮")
                self._safe_callback("on_finished")
                return final_content

            if not tool_calls:
                # no further action -> done
                # Best-effort: extract facts from this conversation turn and
                # persist them to the long-term knowledge graph.  Runs in a
                # daemon thread so the user gets their reply immediately
                # without waiting for the extraction LLM call.
                self._maybe_extract_facts_async(final_content)
                self._safe_callback("on_finished")
                return final_content

            # execute each tool call
            for tc in tool_calls:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "")

                # 协作式取消：每个工具执行前检查。终止后必须为剩余的每个
                # tool_call 补一条合成 tool 消息——assistant_msg 里的
                # tool_calls 已经写入历史，缺少对应的 tool 结果会导致下一轮
                # 请求被 OpenAI 兼容 API 以 HTTP 400 拒绝（悬挂 tool_calls）。
                if self._safe_is_cancelled():
                    self._log(f"[tool] {name} 已因用户取消而跳过")
                    from .tools import ToolResult
                    try:
                        skip_args = parse_args(raw_args)
                    except ValueError:
                        skip_args = {"_raw_arguments": raw_args}
                    self._safe_callback("on_tool_start", name, skip_args)
                    res = ToolResult(False, error="用户取消了对话，工具未执行")
                    self._safe_callback("on_tool_end", name, res)
                    self._history.append(self._tool_message(tc, res))
                    continue

                # 任何路径都必须产出一条 tool 消息：assistant_msg 的 tool_calls
                # 已写入历史，缺一条结果就会变成悬挂 tool_calls（下一轮 400）。
                # 故整段执行包在 _run_one_tool_call 里，异常一律转成失败结果。
                res = self._run_one_tool_call(tc)
                tool_msg = self._tool_message(tc, res)

                # 防线 2/3：失败模式追踪。同一 (工具, 参数) 连续失败 → 在工具
                # 结果末尾附加重试指引；连续多次失败（不管参数）→ 注入整段
                # 反思提醒。让模型在下一步决策前被迫先复盘，而不是盲目重试。
                sig = (name, raw_args)
                if res.success:
                    identical_fail_streak = 0
                    consecutive_failures = 0
                else:
                    identical_fail_streak = (
                        identical_fail_streak + 1 if sig == last_tool_sig else 1
                    )
                    consecutive_failures += 1
                    if identical_fail_streak >= _IDENTICAL_FAIL_GUIDANCE_AT:
                        tool_msg["content"] += (
                            f"\n\n[系统提示] 相同的调用已连续失败 "
                            f"{identical_fail_streak} 次。请修改参数、更换方法，"
                            "或用 ask_user 向用户求助；不要原样重试。"
                        )
                last_tool_sig = sig
                self._history.append(tool_msg)

            # 防线 3：连续多次失败后注入反思提醒（次数有上限，防止提醒本身
            # 变成新的循环）。重置计数器，让下一次提醒需要重新积累失败。
            if (consecutive_failures >= _CONSECUTIVE_FAIL_NUDGE_AT
                    and reflection_nudges < _MAX_REFLECTION_NUDGES):
                reflection_nudges += 1
                consecutive_failures = 0
                self._log(
                    f"[agent] 连续工具失败，注入反思提醒 "
                    f"({reflection_nudges}/{_MAX_REFLECTION_NUDGES})"
                )
                self._history.append({"role": "user", "content": _FAILURE_REFLECTION_NUDGE})

            # 单轮内也可能突破 max_history（一轮最多追加 1+N 条 × 迭代次数），
            # 只在 run() 开头裁剪是不够的。
            self._trim_history()

        # --- 防线 4：迭代预算耗尽，绝不静默终止 ---
        # 走到这里说明最后一轮仍以工具调用收尾（正常完成会在循环内 return）。
        # 旧行为只打一行日志就返回，用户看到的是「任务跑一半突然停了」。
        # 现在注入收尾提醒，并做一次不带工具的最终调用，强制模型基于当前
        # 进度总结成果与剩余步骤；失败/取消时退回最近一条非空回复。
        self._log("[agent] max iterations reached，执行收尾总结")
        if not self._safe_is_cancelled():
            self._history.append({"role": "user", "content": _WRAPUP_NUDGE})
            try:
                messages = self._build_messages()
                wrapup_parts: List[str] = []
                for ev in self.llm.chat_stream(
                    messages, tools=None, temperature=self.config.temperature
                ):
                    if self._safe_is_cancelled():
                        self._log("[agent] 收尾总结被取消")
                        break
                    if ev.type == "content":
                        wrapup_parts.append(ev.content)
                        self._safe_callback("on_content", ev.content)
                wrapup_text = "".join(wrapup_parts).strip()
                if wrapup_text:
                    self._history.append({"role": "assistant", "content": wrapup_text})
                    final_content = wrapup_text
                    self._maybe_extract_facts_async(wrapup_text)
            except Exception as e:
                # 收尾调用本身失败不能掩盖已有进度：退回回退逻辑。
                self._log(f"[agent] 收尾总结失败: {e}")

        if not final_content and self._history:
            # 触及 max_iterations 时最后一轮通常是「只有 tool_calls、没有文本」
            # 的 assistant 消息，final_content 被覆盖成 ""。回退到最近一条非空
            # assistant 文本，避免用户看到一次「什么都没回答」的空回复。
            for msg in reversed(self._history):
                if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
                    final_content = msg["content"]
                    break

        self._safe_callback("on_finished")
        return final_content

    def _run_one_tool_call(self, tc: Dict[str, Any]):
        """执行单个 tool_call，返回 ToolResult——**保证不抛异常**。

        参数解析失败、工具不存在、用户拒绝、UI 回调抛错等全部转换成失败的
        ToolResult，让调用方始终能补上配对的 tool 消息。
        """
        from .tools import ToolResult

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
            # Notify UI of start (with raw args for visibility) before
            # reporting the failure, so the UI shows a complete cycle.
            self._safe_callback("on_tool_start", name, {"_raw_arguments": raw_args})
            res = ToolResult(False, error=str(e))
            self._safe_callback("on_tool_end", name, res)
            return res

        try:
            tool = self.tools.get(name)
        except Exception as e:
            tool = None
            self._log(f"[tool] {name} lookup error: {e}")

        needs_confirm = bool(tool and tool.should_confirm(args)) if tool else False
        self._log(f"[tool] {name}({raw_args}) needs_confirm={needs_confirm}")

        if tool is None:
            # Unknown tool: still notify UI of start for a complete cycle.
            self._safe_callback("on_tool_start", name, args)
            res = self.tools.execute(name, args)  # returns unknown error
            self._safe_callback("on_tool_end", name, res)
            return res

        if needs_confirm:
            confirm_msg = self._confirm_message(name, args)
            try:
                allowed = self.callbacks.confirm(confirm_msg)
            except Exception as e:
                # 确认环节本身出错 → 按「未获授权」处理（安全侧）。
                self._log(f"[tool] {name} confirm error: {e}")
                allowed = False
            if not allowed:
                self._log(f"[tool] {name} DENIED by user")
                # Tool never actually started (user declined before
                # invocation); only report the denial result. The UI
                # intentionally does not receive on_tool_start here so
                # that "never started" semantics are preserved.
                res = ToolResult(False, error=f"User denied {name} ({confirm_msg})")
                self._safe_callback("on_tool_end", name, res)
                return res

        self._safe_callback("on_tool_start", name, args)
        try:
            res = self.tools.execute(name, args)
        except Exception as e:
            # registry.execute 已兜底，这里防的是 UI 回调与未知工具实现。
            self._log(f"[tool] {name} raised {type(e).__name__}: {e}")
            res = ToolResult(False, error=f"{type(e).__name__}: {e}")
        self._safe_callback("on_tool_end", name, res)
        self._log(f"[tool] {name} -> success={res.success} "
                  f"out_len={len(res.output)} err={(res.error or '')[:120]}")
        return res

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

        缓存键还包含 registry 的 state_version：task_plan 这类状态型工具
        会在同一次 run() 内改写注入系统提示的数据（任务计划），不把版本
        计入键的话，后续迭代拿到的仍是旧提示——模型看不到自己刚写的计划。
        """
        try:
            state_ver = getattr(self.tools, "state_version", 0) if self.tools else 0
        except Exception:
            state_ver = 0
        if (hasattr(self, "_cached_prompt_turn")
                and self._cached_prompt_turn == self._turn_idx
                and self._cached_prompt_state == state_ver
                and hasattr(self, "_cached_prompt")):
            return self._cached_prompt

        base = SYSTEM_PROMPT

        try:
            from .tools.memory import _work_memory

            # Work memory: inject only the keys the model explicitly set
            # (its scratchpad), PLUS today's auto-extracted facts.  Long-term
            # memory is NOT injected here — it is recalled on demand via the
            # memory_search tool.
            wm = _work_memory(self.config).list()
            compact_wm = {}
            today_facts: List[str] = []
            for k, v in wm.items():
                if k == "__auto_facts__":
                    # 短期记忆里的自动抽取事实：只注入当天桶（历史已在长期
                    # 记忆 graph 中，靠 memory_search 检索）。
                    _, facts = self._parse_daily_facts(str(v))
                    today_facts = facts
                    continue
                s = str(v)
                compact_wm[k] = s[:200] + "…" if len(s) > 200 else s
            if compact_wm or today_facts:
                parts = []
                if today_facts:
                    # 当天事实用列表形式注入，更易读（而非嵌套 JSON）。
                    # 必须加条数/长度上限：事实每轮累积且系统提示不在上下文
                    # 收缩范围内，放任不管会随会话时长单调膨胀直至挤爆窗口。
                    capped = []
                    used = 0
                    for f in today_facts:
                        if len(capped) >= _MAX_INJECTED_FACTS or used >= _MAX_FACTS_CHARS:
                            break
                        item = f if len(f) <= _MAX_FACT_CHARS else f[:_MAX_FACT_CHARS] + "…"
                        capped.append(item)
                        used += len(item)
                    dropped = len(today_facts) - len(capped)
                    fact_block = "\n".join(f"- {f}" for f in capped)
                    if dropped > 0:
                        fact_block += f"\n- (…{dropped} more omitted)"
                    parts.append("# Today's facts (auto-extracted)\n" + fact_block)
                if compact_wm:
                    parts.append("# Work memory (scratchpad)\n" +
                                 json.dumps(compact_wm, ensure_ascii=False, separators=(",", ":")))
                base += "\n\n" + "\n\n".join(parts)
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
                # 技能 prompt 是用户自写的原文，长度不可控；截断并提示用
                # skill_load 读全文，避免长技能把系统提示撑大。
                sp = matched.prompt or ""
                if len(sp) > _MAX_SKILL_PROMPT_CHARS:
                    sp = (sp[:_MAX_SKILL_PROMPT_CHARS] +
                          "\n…（已截断，用 skill_load 读取完整技能内容）")
                base += f"\n\n# Active skill: {matched.name}\n{sp}"
        except Exception:  # pragma: no cover
            pass

        # Inject the current task plan so every iteration (and every post-shrink
        # request) re-anchors the model to the goal and its progress. The plan
        # lives in the system prompt, which context shrinking never touches.
        try:
            from .planner import load_plan, render_plan
            plan = load_plan(self.config.workspace)
            if plan is not None:
                base += "\n\n" + render_plan(plan)
        except Exception:  # pragma: no cover - defensive
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
                    # 仅注入文档名（不含路径），最多 8 个，超出省略。
                    source_list = ", ".join(
                        os.path.basename(s) for s in sources[:8]
                    )
                    if len(sources) > 8:
                        source_list += f" (and {len(sources) - 8} more)"
                    base += (
                        f"\n\n# Knowledge base\n"
                        f"{chunks} chunks in {len(sources)} file(s): {source_list}. "
                        f"Answer from these via `rag_search`."
                    )
                else:
                    base += (
                        f"\n\n# Knowledge base\n"
                        f"Configured but empty. Use `rag_ingest` to index."
                    )
        except Exception:  # pragma: no cover
            pass

        self._cached_prompt_turn = self._turn_idx
        self._cached_prompt_state = state_ver
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

    @staticmethod
    def _summarize_code(code: str) -> str:
        """Heuristic fallback: infer the purpose of a code snippet.

        Tries comments/docstrings first (models usually emit a purpose comment),
        then the first non-import statement.  Returns a short "做什么" phrase
        WITHOUT dumping the code.  Used only when LLM summarization is
        unavailable.
        """
        code = code.strip()
        if not code:
            return "（空代码）"
        lines = [ln.rstrip() for ln in code.splitlines()]

        # 1) Extract a purpose comment / docstring near the top.
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            # Module docstring or inline comment describing intent.
            if s.startswith('"""') or s.startswith("'''"):
                doc = s.strip('"\'').strip()
                if doc and not doc.startswith(("import", "from", "#!")):
                    return doc[:80]
            if s.startswith("#"):
                comment = s.lstrip("#").strip()
                # Skip shebang and import markers.
                if comment and not comment.startswith(("!", "encoding", "coding")):
                    return comment[:80]
            # Stop at the first real statement (imports come before purpose comments
            # in many generated snippets, so scan past leading imports).
            if s.startswith(("import ", "from ")):
                continue
            break

        # 2) Infer from the first executable statement.
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith(("#", "import ", "from ")):
                continue
            if s.startswith("def "):
                m = re.match(r"def\s+(\w+)", s)
                return f"定义函数 {m.group(1)}" if m else "定义函数"
            if s.startswith("class "):
                m = re.match(r"class\s+(\w+)", s)
                return f"定义类 {m.group(1)}" if m else "定义类"
            if s.startswith("pip install"):
                return "安装 Python 包：" + s[len("pip install"):].strip()[:60]
            if s.startswith("npm install"):
                return "安装 Node 包：" + s[len("npm install"):].strip()[:60]
            return "执行代码片段：" + s[:60]
        return "执行一段代码"

    @staticmethod
    def _summarize_command(cmd: str) -> str:
        """Infer the purpose of a shell command (no full command dump)."""
        cmd = cmd.strip()
        if not cmd:
            return "（空命令）"
        # Map common commands to a human-readable purpose.
        parts = cmd.split()
        prog = parts[0] if parts else ""
        known = {
            "pip": "安装/管理 Python 包",
            "npm": "安装/管理 Node 包",
            "brew": "安装/管理 macOS 软件",
            "apt": "安装/管理系统软件包",
            "apt-get": "安装/管理系统软件包",
            "yum": "安装/管理系统软件包",
            "git": "Git 版本控制操作",
            "cd": "切换目录",
            "ls": "列出目录内容",
            "cat": "查看文件内容",
            "mkdir": "创建目录",
            "touch": "创建文件",
            "rm": "删除文件/目录",
            "mv": "移动/重命名文件",
            "cp": "复制文件",
            "curl": "发起网络请求",
            "wget": "下载文件",
            "python": "运行 Python 脚本",
            "python3": "运行 Python 脚本",
            "node": "运行 Node 脚本",
            "grep": "搜索文本内容",
            "find": "查找文件",
            "chmod": "修改文件权限",
            "echo": "输出文本",
            "tar": "打包/解压归档",
            "unzip": "解压 zip 文件",
            "ssh": "SSH 远程连接",
            "docker": "Docker 容器操作",
            "kubectl": "Kubernetes 集群操作",
        }
        if prog in known:
            return known[prog]
        # Generic: show the program name only.
        return f"执行命令 {prog}"

    def _describe_code_purpose(self, code: str) -> str:
        """Return a one-line purpose description for code/command.

        Strategy: heuristic FIRST (zero latency — most generated code carries
        a purpose comment/docstring, and known commands map to a fixed phrase).
        Only when the heuristic yields a generic/empty description do we fall
        back to the LLM (a slow 1-3s call that would otherwise delay the
        confirmation dialog on every run).
        """
        is_cmd = self._looks_like_command(code)
        if is_cmd:
            heuristic = self._summarize_command(code)
            # Command mapping is authoritative for known programs; only fall
            # back to LLM for unknown programs.
            if not heuristic.startswith("执行命令 "):
                return heuristic
        else:
            heuristic = self._summarize_code(code)
            # A useful heuristic is a real purpose phrase (comment/docstring/
            # function name), not the generic "执行代码片段：<first line>" or
            # "执行一段代码" fallback.
            if not heuristic.startswith(("执行代码片段", "执行一段代码")):
                return heuristic

        # Fallback: ask the LLM (dedicated client, with a short timeout).
        try:
            llm = self._get_extract_llm()
            if llm is not None and llm is not self.llm:
                # 持防堆叠锁使用：后台抽取线程在整个 LLM 调用期间也持有这把
                # 锁，而快路径 _get_extract_llm 本身不持锁——不加这层的话
                # confirm 的流式请求会与后台抽取并发共用同一个 httpx.Client
                # （模块明确声明非线程安全），两边流交错损坏。拿不到锁说明
                # 抽取正在进行：LLM 兜底只是锦上添花，退回启发式描述，
                # 弹窗不延迟。
                if not self._fact_extract_lock.acquire(blocking=False):
                    return heuristic
                try:
                    prompt = (
                        "用 ≤20 字中文说明以下代码/命令做什么，只输出这一句，"
                        "不要贴代码、不要以「这段代码」开头：\n\n"
                        + code[:2000]
                    )
                    # chat_stream 产出的是 StreamEvent 数据类，不是 dict。旧代码写
                    # `ev.get(...)` 会抛 AttributeError 并被下面的 except 吞掉，
                    # 导致 LLM 兜底永远是死代码、只能退回通用描述。这里按属性访问。
                    answer_parts = []
                    for ev in llm.chat_stream([{"role": "user", "content": prompt}]):
                        if ev.type == "content":
                            answer_parts.append(ev.content)
                        elif ev.type == "done":
                            break
                    answer = "".join(answer_parts).strip()
                    if answer:
                        return answer[:80]
                finally:
                    self._fact_extract_lock.release()
        except Exception:
            pass
        return heuristic

    @staticmethod
    def _looks_like_command(text: str) -> bool:
        """True if text looks like a shell command rather than code."""
        s = text.strip()
        if not s:
            return False
        # Commands are usually a single line starting with a known program,
        # without Python keywords.
        if "\n" in s:
            return False
        first = s.split()[0] if s.split() else ""
        return first in {
            "pip", "npm", "brew", "apt", "apt-get", "yum", "git", "cd", "ls",
            "cat", "mkdir", "touch", "rm", "mv", "cp", "curl", "wget", "python",
            "python3", "node", "grep", "find", "chmod", "echo", "tar", "unzip",
            "ssh", "docker", "kubectl",
        }

    def _confirm_message(self, name: str, args: Dict[str, Any]) -> str:
        """Build a concise confirmation message (purpose description, no code).

        ``code_run`` / ``shell_run`` carry full code/commands — dumping them
        into the dialog makes it unreadably long and pushes the Yes/No buttons
        off-screen.  We show a short *purpose* description instead.
        """
        if name == "code_run":
            code = str(args.get("code", "")).strip()
            timeout = args.get("timeout")
            purpose = self._describe_code_purpose(code)
            msg = f"即将执行代码：{purpose}"
            if timeout is not None:
                msg += f"\n超时：{timeout}s"
            return msg + "\n\n是否继续？"
        if name == "shell_run":
            cmd = str(args.get("command", "")).strip()
            timeout = args.get("timeout")
            purpose = self._describe_code_purpose(cmd)
            msg = f"即将执行命令：{purpose}"
            if timeout is not None:
                msg += f"\n超时：{timeout}s"
            return msg + "\n\n是否继续？"

        # Other tools: show a compact argument summary (truncated).
        try:
            pretty = json.dumps(args, ensure_ascii=False)
        except Exception:
            pretty = str(args)
        if len(pretty) > 500:
            pretty = pretty[:500] + "…"
        return f"即将调用工具 `{name}`：\n{pretty}\n\n是否继续？"

    def _log(self, line: str) -> None:
        self._safe_callback("on_log", line)

    def _safe_is_cancelled(self) -> bool:
        """取消标志查询的隔离版本。

        取消状态由 Qt 侧的回调提供（控件可能已销毁 → RuntimeError）。裸调用
        ``is_cancelled()`` 的异常会穿出 ``run()``：已写入历史的
        assistant(tool_calls) 拿不到配对的 tool 消息，下一轮请求被 API 400
        拒绝。查询失败时保守地返回 True（已取消）会让正常对话被误终止，因此
        这里返回 False —— 继续执行，由下一个检查点再确认。
        """
        fn = getattr(self.callbacks, "is_cancelled", None)
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception as e:
            if self._callback_errors < 3:
                self._callback_errors += 1
                logger.warning("[agent] is_cancelled() 回调异常（按未取消处理）: %s", e)
            return False

    def _safe_callback(self, method: str, *args) -> None:
        """调用 UI 回调并隔离其异常。

        UI 回调最终跑在 Qt 信号槽里，可能因控件已销毁等原因抛异常
        （RuntimeError / TypeError）。若不隔离：
        1. 流式循环里的异常会被外层 ``except Exception`` 当成「LLM 流错误」
           上报，并在 except 块里二次抛出，导致 ``on_finished()`` 永不执行
           —— UI 一直停在忙碌状态。
        2. 工具循环里的异常会让已写入历史的 assistant(tool_calls) 得不到
           对应的 tool 消息，下一轮请求被 API 以 400 拒绝。

        这里统一吞掉并记日志：一次 UI 渲染失败不该毁掉整轮对话。
        """
        fn = getattr(self.callbacks, method, None)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as e:  # pragma: no cover - 取决于 UI 状态
            self._callback_errors += 1
            # 只记录前几条，避免 UI 持续抛错时日志被刷屏。
            if self._callback_errors <= 5:
                try:
                    self.callbacks.on_log(
                        f"[agent] callback {method} raised: {type(e).__name__}: {e}"
                    )
                except Exception:
                    pass
