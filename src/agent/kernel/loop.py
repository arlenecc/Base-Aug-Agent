"""极简三态自动机：内核只做控制流。

    THINK ──has_calls──► ACT ──continue──► THINK
      │                   │
      │ no_calls          ├─ consolidate ─► CONSOLIDATE ──► THINK
      ▼                   └─ interrupt/done ─► EXIT
     EXIT

它不拥有工具实现、不拥有记忆、不拥有 RAG —— 那些都是注入的能力：
``dispatcher``（能力）、``session``（历史）、``hooks``（经验沉淀）。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .budget import ContextBudget, estimate_tokens
from .compressor import l0_capture, l1_trim, l2_summary
from .outcome import (
    ExitReason, KernelState, LLMResponse, StepOutcome, ToolCall, next_state,
)
from .session import Session


# 空响应 / 流中断的重试上限
_MAX_EMPTY_RETRY = 3
_LLM_RETRY_BASE_DELAY = 1.5
_MAX_LLM_RETRIES = 2

# L0 硬上限：工具结果进上下文前先兜住病态长度（工具自身上限更高，例如
# file_read 允许 200K 字符）。L1 之后还会按 RAW 区余额再裁一刀。
_L0_HARD_CAP = 20_000

# 流中断 / 截断标记（对齐 GenericAgent 的 do_no_tool）
_TRUNCATION_MARKS = ("[!!! 流异常中断", "!!!Error:", "max_tokens !!!]")
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_]*\n[\s\S]{50,}?```")
_TAG_STRIP_RE = re.compile(r"<(thinking|think|summary)>[\s\S]*?</\1>", re.I)

# ---- 规划门 / 反思门（防止「跑丢」与「半路停」）---------------------------
# 任务开始后前几轮还没有计划时，每轮都提醒模型先规划。给 3 次机会：
# 一直不规划就放行（可能是「帮我读个文件」这类一句话任务，强制规划反而烦人）。
_PLAN_GATE_TURNS = 3
# 模型想收尾但计划里还有未完成步骤时的最大提醒次数。超过后尊重模型的决定，
# 但在回复末尾附上未完成清单——用户必须知道任务其实没做完。
_FINISH_NUDGE_MAX = 2
# 发生过失败时放宽反思门预算：用户明确要求「步骤失败后自动恢复并继续」，
# 此时模型停下来问「是否继续」正是要防的行为——多提醒几次换来自动恢复，
# 比早早收尾、让用户手动打「继续」更符合预期。
_FINISH_NUDGE_FAIL_MAX = 4
# 失败调用签名保留数（识别「原样重试」：同名工具 + 相同参数再次失败）
_MAX_FAIL_SIGS = 8
# 把步骤标记 blocked 前至少要有多少次失败尝试（步骤边界 = 上次 update_plan）
_MIN_ATTEMPTS_BEFORE_BLOCK = 2
# 「未充分尝试就标记 blocked」的提醒上限（防无限循环）
_BLOCK_NUDGE_MAX = 2

_PLAN_GATE_HINT = (
    "\n[System] 开始执行前，先用 update_plan(op=\"create\", goal=..., steps=[...]) "
    "把目标和路径规划成 2~7 个可验证的步骤；然后从第一步开始，每完成一步就用 "
    "update_plan(op=\"patch\", step_id=..., status=\"done\", note=\"结论\") 更新进度。"
)


class KernelHooks:
    """经验沉淀回调。默认 no-op —— 内核本身不做任何沉淀。"""

    def consolidated(self, session: Session, reason: str) -> bool:
        """折叠/沉淀回调。返回 True 表示上下文真的变小了。"""
    def finished(self, session: Session, text: str) -> None: ...
    def failed(self, session: Session, err: str) -> None: ...


class KernelResult:
    """一次 run() 的产物。"""

    def __init__(self, text: str = "", reason: ExitReason = ExitReason.DONE,
                 turns: int = 0, distilled: bool = False):
        self.text = text
        self.reason = reason
        self.turns = turns
        self.distilled = distilled

    def __repr__(self) -> str:
        return f"<KernelResult {self.reason.value} turns={self.turns} len={len(self.text)}>"


class KernelLoop:
    """三态自动机主循环。"""

    def __init__(
        self,
        llm: Any,
        dispatcher: Any,                      # dispatch(List[ToolCall]) -> List[ToolResult]
        callbacks: Any,
        config: Any,
        system_prompt: Callable[[], str],
        session: Optional[Session] = None,
        hooks: Optional[KernelHooks] = None,
        log: Optional[Callable[[str], None]] = None,
        max_turns: int = 0,
    ):
        # llm 允许是「对象」或「返回对象的 callable」。UI 会整体替换
        # agent.llm（切换模型/配置），用 callable 才能拿到最新的客户端。
        self.llm = llm
        self.dispatcher = dispatcher
        self.callbacks = callbacks
        self.config = config
        self.system_prompt = system_prompt
        self.hooks = hooks or KernelHooks()
        self.log = log or (lambda *_a, **_k: None)
        self.session = session or Session(
            budget=ContextBudget(
                int(getattr(config, "kernel_max_context_tokens", 30_000) or 30_000)
            ),
            raw_turns=int(getattr(config, "kernel_raw_turns", 2) or 2),
            max_digest_lines=int(getattr(config, "kernel_max_digest_lines", 40) or 40),
        )
        # 必须与 Session 共用同一个预算对象，否则内核记的账和 Session 折叠时
        # 读的余额是两本账（L1/L3 会各按自己的数裁决）。
        self.budget = self.session.budget
        self.consolidate_every = max(1, int(getattr(config, "kernel_consolidate_every", 4) or 4))
        self.max_tool_chars = int(getattr(config, "kernel_max_tool_chars", 6_000) or 6_000)
        self.max_turns = max(1, int(max_turns or getattr(config, "max_iterations", 15) or 15))

        self.state: KernelState = KernelState.THINK
        self.distill_requested = False
        self._schemas: Optional[List[Dict[str, Any]]] = None
        self._schema_tokens = 0
        self._empty_retry = 0
        self._shrink_guard = 0
        self._reask_guard = 0
        self._turns_used = 0
        self._last_consolidated_turn = -1
        self._total_tokens = 0
        self._finish_nudge = 0
        # ---- 失败恢复记账（每次 run() 重置）----
        self._fail_streak = 0        # 连续失败的工具调用数（任一批全成功即清零）
        self._failed_total = 0       # 本次 run 失败总数；>0 即放宽反思门预算
        self._last_fail: Optional[Tuple[str, str]] = None   # 最近一次 (工具名, 错误)
        self._fail_sigs: List[str] = []          # 近期失败调用签名（原样重试检测）
        self._step_fails = 0         # 上次 update_plan 之后的失败数 ≈ 当前步骤的尝试次数
        self._block_nudge = 0        # 「未充分尝试就 blocked」已提醒次数

    # ------------------------------------------------------------------
    # 注入的能力：工具 schema
    # ------------------------------------------------------------------
    def set_schemas(self, schemas: List[Dict[str, Any]]) -> None:
        self._schemas = list(schemas or [])
        self._schema_tokens = estimate_tokens(json.dumps(self._schemas, ensure_ascii=False)) if self._schemas else 0
        self.budget.set_used("tools", self._schema_tokens)

    def schemas(self) -> List[Dict[str, Any]]:
        if self._schemas is None:
            try:
                self.set_schemas(self.dispatcher.schemas())
            except Exception:
                self.set_schemas([])
        self.budget.set_used("tools", self._schema_tokens)
        return self._schemas or []

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> KernelResult:
        session = self.session
        self.state = KernelState.THINK
        self._empty_retry = 0
        self._shrink_guard = 0
        self._reask_guard = 0
        self._finish_nudge = 0
        self._fail_streak = 0
        self._failed_total = 0
        self._last_fail = None
        self._fail_sigs = []
        self._step_fails = 0
        self._block_nudge = 0
        # max_turns 是「本次 run() 的轮次上限」，不是「整个会话的累计轮次」。
        # Session.turn_index 跨 run() 单调递增（它就是历史编号），拿它当循环
        # 上界会让 Agent 在累计 N 轮之后彻底不再调用 LLM。
        self._turns_used = 0
        self.distill_requested = False

        pending = user_input
        final_text = ""
        exit_reason = ExitReason.MAX_TURNS

        while self._turns_used < self.max_turns:
            if self._cancelled():
                self.log("[kernel] 收到取消请求，终止")
                exit_reason = ExitReason.CANCELLED
                break

            # ---- CONSOLIDATE ----
            if self._needs_consolidate():
                self.state = KernelState.CONSOLIDATE
                self._last_consolidated_turn = session.current_turn
                over = self.budget.raw_over
                reason = "预算告罄" if over else f"每 {self.consolidate_every} 轮"
                # 周期性沉淀也必须真的折叠（旧代码这里传 0，force_fold 变成空操作），
                # 否则 turns 只会增长、extra_digest 永远为空。
                folded = session.force_fold(3 if over else 1)
                self.log(f"[kernel] CONSOLIDATE ({reason}) folded={folded} {session.stats()}")
                try:
                    self.hooks.consolidated(session, reason)
                except Exception as e:
                    self.log(f"[kernel] consolidate hook error: {e}")
                # 只折一轮就回 THINK：`raw` 的账在 build_messages 里刷新，这里
                # 旧实现用 continue 循环折叠——但 _used["raw"] 是上一轮的旧值，
                # 会连折 3×3=9 轮才罢休（其中 6 轮是白折的）。
                # 真的没东西可折了才交上层兜底收缩。
                if over and folded == 0 and self._shrink_guard < 3:
                    self._shrink_guard += 1
                    try:
                        self.hooks.consolidated(session, "fold-exhausted")
                    except Exception:
                        pass
                self.state = next_state(KernelState.CONSOLIDATE, "done")

            # ---- THINK ----
            self.state = KernelState.THINK
            # 规划门：任务已开始但还没有计划时，提醒先规划再动手。
            # 只在前几轮施加，之后放行——一句话任务（「读一下 a.txt」）强制
            # 规划只会添乱。
            if (session.get_plan() is None
                    and self._turns_used < _PLAN_GATE_TURNS):
                pending = (pending or "") + _PLAN_GATE_HINT
            user_text = session.anchor_prompt(extra=pending)
            rec = session.start_turn(user_text)
            # rollback 路径不减计数：重试由各自的 guard 兜底，这里只保证终止性。
            self._turns_used += 1
            messages = session.build_messages(self.system_prompt(), user_text)
            self.schemas()

            try:
                resp = self._ask(messages)
            except Exception as e:
                msg = str(e)
                # 上下文超长是可恢复的：交给上层兜底收缩后重试同一轮。
                # 不重试 = 用户直接看到硬错误（旧实现的被动收缩正是为此存在）。
                if self._is_context_error(msg) and self._shrink_guard < 3:
                    self._shrink_guard += 1
                    self._rollback_turn(session)
                    shrunk = False
                    try:
                        shrunk = bool(self.hooks.consolidated(session, "fold-exhausted"))
                    except Exception:
                        pass
                    if shrunk:
                        self.log(f"[kernel] 上下文超长，兜底收缩后重试 ({self._shrink_guard})")
                        continue
                    # 没东西可压了 —— 再发一次同样的超长 prompt 没有意义。
                self.log(f"[kernel] LLM error: {e}")
                try:
                    self.hooks.failed(session, str(e))
                except Exception:
                    pass
                exit_reason = ExitReason.ERROR
                break

            if resp is None:                      # 流式期间被取消
                # 这一轮只有 user 没有 assistant，留着会让历史出现两条相邻
                # user 消息；取消即丢弃，回退掉。
                self._rollback_turn(session)
                exit_reason = ExitReason.CANCELLED
                break

            if not resp.has_calls:
                # 反思门：模型想收尾，但计划里还有未完成步骤 —— 不能就这么放它
                # 走。这正是「半路任务跑丢 / 直接就停了」的主因：模型自认为做完
                # 了，实际步骤没走完。提醒有限次，超过后尊重决定但在回复里附上
                # 未完成清单（用户必须知道任务并没完成）。
                open_steps = session.plan_open_steps()
                # 失败发生过时放宽预算：此时「停下来问用户是否继续」正是要防的
                # 行为，模型应当依据失败信息自动恢复并继续后续步骤。
                nudge_budget = (_FINISH_NUDGE_FAIL_MAX if self._failed_total
                                else _FINISH_NUDGE_MAX)
                if open_steps and self._finish_nudge < nudge_budget:
                    self._finish_nudge += 1
                    todo = "; ".join(
                        f"{s.get('id')}.{s.get('title')}" for s in open_steps[:5]
                    )
                    self._rollback_turn(session)
                    if self._failed_total and self._last_fail:
                        # 失败感知的提醒：把最近一次失败摆到模型眼前，并要求
                        # 「依据失败信息自动解决」而不是停下来等用户发话。
                        fname, ferr = self._last_fail
                        pending = (
                            f"[System] 计划中还有 {len(open_steps)} 个未完成步骤：{todo}。"
                            f"且此前有工具调用失败（最近：{fname} → {ferr[:160]}）。"
                            "不要停下来、不要问用户是否继续——请依据失败信息自动解决："
                            "1) 读完整错误定位原因 2) 换方法重试（不同工具/不同参数，"
                            "或用 code_run 写诊断脚本）3) 多种方法都不可行时才把该步标记 "
                            "blocked 并写明原因，再继续后续步骤。请现在调用工具继续执行。"
                        )
                    else:
                        pending = (
                            f"[System] 计划中还有 {len(open_steps)} 个未完成步骤：{todo}。"
                            "请继续执行下一步（调用相应工具）；若某步确实无法完成，先用 "
                            "update_plan 把它标为 blocked/skipped 并写明原因，再给出总结。"
                        )
                    self.log(f"[kernel] 反思门：还有 {len(open_steps)} 步未完成，提醒继续")
                    continue
                event = self._no_tool_event(resp)
                if event == "retry":
                    self._rollback_turn(session)
                    pending = "[ERROR] 上一轮回复为空或流被截断，请重新生成；如需操作必须显式调用工具。"
                    continue
                if event == "reask":
                    self._reask_guard += 1
                    if self._reask_guard <= 1:
                        # 第一次只是提醒：整轮作废重来（这一轮没有 assistant
                        # 输出，留着只会污染历史）。
                        self._rollback_turn(session)
                        pending = (
                            "[System] 检测到你上轮主要内容是大段代码却未调用工具。"
                            "若这些代码需要执行/写入/分析，请显式调用 code_run / "
                            "file_write / file_patch；"
                            "若只是展示给用户的示例，请在回复中补充自然语言说明。"
                        )
                        continue
                    # 第二次不再纠缠：把这轮的正文当作最终答复收下。
                    # 注意必须在 finish_turn 之后再 break —— 回滚会把 rec 从
                    # session 摘掉，之后再 finish 写的是游离对象，等于丢轮。
                    final_text = resp.content
                    session.finish_turn(rec, resp.content, [], [],
                                        l2_summary(resp.content))
                    exit_reason = ExitReason.DONE
                    break
                final_text = resp.content
                session.finish_turn(rec, resp.content, [], [],
                                    l2_summary(resp.content))
                exit_reason = ExitReason.DONE
                break

            # ---- ACT ----
            self.state = next_state(KernelState.THINK, "has_calls")
            tool_calls = resp.tool_calls
            try:
                raw_results = self.dispatcher.dispatch(tool_calls)
            except Exception as e:
                self.log(f"[kernel] dispatcher error: {e}")
                raw_results = [None] * len(tool_calls)

            # 失败记账必须在拼装 pending 之前：步骤内失败数要在 update_plan
            # 把它重置之前，被「未充分尝试就 blocked」的守卫读走。
            step_fails_before = self._step_fails
            batch = self._track_failures(tool_calls, raw_results)

            outcomes = self._to_outcomes(tool_calls, raw_results)
            tool_results = [
                {"id": tc.id, "name": tc.name, "content": str(o.data or "")}
                for tc, o in zip(tool_calls, outcomes)
            ]
            summary = resp.summary or l2_summary(resp.content, [tc.name for tc in tool_calls])
            session.finish_turn(rec, resp.content, tool_calls, tool_results, summary)
            has_summary = bool(resp.summary)

            if any(o.should_exit for o in outcomes):
                final_text = resp.content
                exit_reason = ExitReason.INTERRUPT
                break

            next_prompts = sorted({o.next_prompt for o in outcomes if o.next_prompt})
            if not next_prompts:
                final_text = resp.content
                exit_reason = ExitReason.DONE
                break

            parts = list(next_prompts)
            if not has_summary:
                parts.append("[TIPS] 回复正文必须包含 <summary>…</summary>（≤30 字物理快照）。")
            # 失败恢复提示：计划步骤出错时不许停下等用户说「继续」——依据失败
            # 信息自动换方案，并防住原样重试与「未充分尝试就 blocked」。
            parts.extend(self._recovery_hints(tool_calls, raw_results, batch,
                                              step_fails_before))
            pending = "\n".join(parts)
            self.state = next_state(KernelState.ACT, "continue")
        else:
            exit_reason = ExitReason.MAX_TURNS

        if exit_reason == ExitReason.MAX_TURNS and not final_text:
            final_text = (
                f"（已达到本轮最大轮次上限 {self.max_turns}，任务可能未完成。"
                f"请先总结已完成的进展，再询问用户是否继续。）"
            )
        # 提醒次数用尽后模型仍要收尾：尊重决定，但必须把「其实没做完」告诉用户，
        # 否则又是一个静默的半途而废。
        if exit_reason is ExitReason.DONE and self._finish_nudge:
            left = session.plan_open_steps()
            if left:
                todo = "; ".join(f"{s.get('id')}.{s.get('title')}" for s in left[:8])
                final_text = (final_text or "") + (
                    f"\n\n---\n⚠ 计划中仍有 {len(left)} 个未完成步骤：{todo}"
                    "（可回复「继续」让我接着做，或用 update_plan 调整计划。）"
                )

        try:
            self.hooks.finished(session, final_text)
        except Exception as e:
            self.log(f"[kernel] finished hook error: {e}")

        return KernelResult(final_text, exit_reason, self._turns_used,
                            self.distill_requested)

    # ------------------------------------------------------------------
    # 转移判定
    # ------------------------------------------------------------------
    def _needs_consolidate(self) -> bool:
        t = self.session.current_turn
        if self.budget.raw_over:
            return True
        if t > 0 and t % self.consolidate_every == 0 and t != self._last_consolidated_turn:
            return True
        return False

    @staticmethod
    def _rollback_turn(session: Session) -> None:
        """空轮/无效轮不占历史，直接回退。"""
        if session.turns:
            session.turns.pop()
        session.turn_index = max(0, session.turn_index - 1)

    # ------------------------------------------------------------------
    # THINK：一次 LLM 调用
    # ------------------------------------------------------------------
    def _llm(self) -> Any:
        """解析 LLM 客户端（允许运行时被整体替换）。"""
        return self.llm() if callable(self.llm) else self.llm

    def _ask(self, messages: List[Dict[str, Any]]) -> Optional[LLMResponse]:
        content_buf: List[str] = []
        reasoning_buf: List[str] = []
        calls: List[Dict[str, Any]] = []
        usage: Dict[str, Any] = {}
        ttfb: Optional[float] = None
        last_emit: Optional[float] = None
        live_est = 0
        attempt = 0
        t0 = time.time()
        cancelled = False
        seen_done = False

        while True:
            content_buf, reasoning_buf, calls, usage = [], [], [], {}
            ttfb = last_emit = None
            live_est = 0
            got_any = False
            seen_done = False
            try:
                for ev in self._llm().chat_stream(
                    messages,
                    tools=self.schemas(),
                    temperature=getattr(self.config, "temperature", 0.7),
                ):
                    got_any = True
                    if self._cancelled():
                        cancelled = True
                        break
                    now = time.time()
                    if ttfb is None:
                        ttfb = now
                    t = getattr(ev, "type", "")
                    if t == "content":
                        content_buf.append(ev.content)
                        self.callbacks.on_content(ev.content)
                        live_est += estimate_tokens(ev.content)
                    elif t == "reasoning":
                        reasoning_buf.append(ev.content)
                        self.callbacks.on_reasoning(ev.content)
                        live_est += estimate_tokens(ev.content)
                    elif t == "done":
                        seen_done = True
                        calls = getattr(ev, "tool_calls", None) or []
                        usage = getattr(ev, "usage", None) or {}
                        continue
                    if last_emit is None:
                        last_emit = now
                    elif live_est and now - last_emit >= 0.3:
                        gen = max(now - (ttfb or t0), 1e-6)
                        self.callbacks.on_token_speed(self._total_tokens + live_est, live_est / gen)
                        last_emit = now
                break
            except Exception as e:
                # 只有「一个 chunk 都没收到」才重试：已吐过内容的流重试会拼接两段输出。
                if got_any or attempt >= _MAX_LLM_RETRIES:
                    raise
                if self._is_context_error(str(e)) or re.search(r"HTTP 4\d\d", str(e)):
                    raise
                attempt += 1
                delay = _LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.log(f"[kernel] LLM 首字节前失败，{delay:.1f}s 后重试 ({attempt}): {e}")
                if self._cancelled():
                    raise
                time.sleep(delay)

        if cancelled:
            return None

        reasoning_text = "".join(reasoning_buf)
        if not seen_done and ("".join(content_buf) or reasoning_text):
            # 流被截断（网络断开 / 服务端提前结束）——必须留下可见痕迹。
            self.log("[agent] warning: LLM stream ended without 'done' event")

        if usage:
            try:
                self.callbacks.on_usage(usage)
            except Exception:
                pass
            details = usage.get("completion_tokens_details") or {}
            api_reasoning = int(details.get("reasoning_tokens") or 0) if isinstance(details, dict) else 0
            est_reasoning = estimate_tokens(reasoning_text)
            api_total = int(usage.get("total_tokens") or 0)
            # API 的 total 可信（含 reasoning）时才直接用；否则补上估算值，
            # 否则推理密集型模型会显示低得离谱的 tok/s。
            if api_reasoning or not reasoning_text:
                self._total_tokens = getattr(self, "_total_tokens", 0) + api_total
            else:
                self._total_tokens = getattr(self, "_total_tokens", 0) + api_total + est_reasoning
        completion = int(usage.get("completion_tokens") or 0)
        gen_elapsed = max(time.time() - (ttfb or t0), 1e-6)
        try:
            self.callbacks.on_token_speed(self._total_tokens,
                                          completion / gen_elapsed if completion else 0.0)
        except Exception:
            pass

        text = "".join(content_buf)
        resp = LLMResponse(
            content=text,
            reasoning=reasoning_text,
            tool_calls=[self._to_tool_call(c) for c in calls],
            usage=usage,
        )
        resp.summary = l2_summary(text, [tc.name for tc in resp.tool_calls]) if text else ""
        return resp

    def _to_tool_call(self, raw: Dict[str, Any]) -> ToolCall:
        fn = (raw.get("function") or {}) if isinstance(raw, dict) else {}
        args_raw = fn.get("arguments", "")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else {}
        except (json.JSONDecodeError, ValueError) as e:
            # 解析失败必须报回模型，静默当 {} 会变成莫名的「缺少必填参数」。
            args = {"__parse_error__": f"Invalid JSON in tool arguments: {e}",
                    "__raw__": args_raw}
        if not isinstance(args, dict):
            args = {}
        tid = str(raw.get("id") or "")
        if not tid:
            # 部分本地模型不发 id；没有 id 就无法把 tool 结果关联回去。
            # 前缀必须区别于模型常见的 "call_1"，否则同一轮内可能撞 id
            # （两条 tool 消息同 tool_call_id → API 4xx）。
            self._tc_counter = getattr(self, "_tc_counter", 0) + 1
            tid = f"kcall_{self._tc_counter}"
        return ToolCall(
            id=tid,
            name=str(fn.get("name") or ""),
            args=args,
            raw_arguments=args_raw if isinstance(args_raw, str) else "",
        )

    # ------------------------------------------------------------------
    # ACT：工具结果 → StepOutcome（L1 裁剪在这里发生）
    # ------------------------------------------------------------------
    def _to_outcomes(self, tool_calls: List[ToolCall],
                     raw_results: List[Any]) -> List[StepOutcome]:
        n = max(1, len(tool_calls))
        max_chars = self.budget.remaining_for_tool(n, self.max_tool_chars)
        out: List[StepOutcome] = []
        for tc, res in zip(tool_calls, raw_results):
            # ask_user 与普通工具同路径：用户回答作为 tool 结果落进历史，
            # 模型带着答案继续任务。之前把它特判成 INTERRUPT 会让整轮终止、
            # 用户答案被直接丢弃（回答进不了任何后续 LLM 调用）。
            if tc.name == "start_long_term_update":
                self.distill_requested = True
            # L0：先兜住病态长度（工具本身没上限时保护进程内存），
            # L1：再按 RAW 区余额裁到真正要注入的长度。
            out.append(StepOutcome(
                data=l1_trim(l0_capture(self._result_text(res), hard_cap=_L0_HARD_CAP),
                             max_chars),
                next_prompt="\n"))
        return out

    @staticmethod
    def _result_text(res: Any) -> str:
        if res is None:
            return "ERROR: 工具未返回结果"
        to_msg = getattr(res, "to_message", None)
        if callable(to_msg):
            try:
                return to_msg()
            except Exception:
                pass
        return str(res)

    # ------------------------------------------------------------------
    # ACT：失败记账与自动恢复提示
    #
    # 计划步骤失败时，内核不允许「自动停下等用户打继续」：失败信息必须转化为
    # 下一轮的恢复指令（换方案 / 先诊断），同时防住两种浪费——原样重试同一个
    # 失败调用、没试几次就把步骤标记 blocked。
    # ------------------------------------------------------------------
    @staticmethod
    def _result_ok(res: Any) -> Tuple[bool, str]:
        """ToolResult → (是否成功, 错误摘要)。None/异常形态一律算失败。"""
        if res is None:
            return False, "工具未返回结果"
        if getattr(res, "success", True):
            return True, ""
        err = str(getattr(res, "error", "") or "") or "未知错误"
        return False, err[:300]

    @staticmethod
    def _call_sig(tc: ToolCall) -> str:
        """失败调用签名：工具名 + 参数。用于识别「原样重试」。"""
        try:
            args = json.dumps(tc.args, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            args = str(tc.args)
        return f"{tc.name}::{args[:500]}"

    def _track_failures(self, tool_calls: List[ToolCall],
                        raw_results: List[Any]) -> Dict[str, Any]:
        """ACT 态失败记账：连续失败、失败签名、当前步骤的失败次数。

        ``update_plan`` 的失败不计入步骤失败数（它是对计划的登记，不是对步骤
        本身的尝试）；但它**成功**时视为步骤边界，重置计数。返回本批摘要
        （含「原样重试」检测结果），供 ``_recovery_hints`` 使用。
        """
        failed: List[Tuple[ToolCall, str]] = []
        repeats: List[Tuple[str, str]] = []
        plan_mutated = False
        for tc, res in zip(tool_calls, raw_results):
            ok, err = self._result_ok(res)
            if tc.name == "update_plan":
                if ok:
                    plan_mutated = True
                continue
            if ok:
                continue
            failed.append((tc, err))
            self._fail_streak += 1
            self._failed_total += 1
            self._last_fail = (tc.name, err)
            self._step_fails += 1
            sig = self._call_sig(tc)
            if sig in self._fail_sigs:
                repeats.append((tc.name, err))
            self._fail_sigs.append(sig)
            del self._fail_sigs[:-_MAX_FAIL_SIGS]
        if not failed:
            self._fail_streak = 0
        if plan_mutated:
            self._step_fails = 0
        return {"failed": failed, "repeats": repeats, "plan_mutated": plan_mutated}

    def _recovery_hints(self, tool_calls: List[ToolCall], raw_results: List[Any],
                        batch: Dict[str, Any], step_fails_before: int) -> List[str]:
        """依据本批失败信息生成「自动恢复」提示，拼进下一轮锚点。

        三个层次：
          1. 原样重试 → 禁止重复，必须换方法
          2. 连续失败 ≥2 → 停止同思路，先诊断再换方案
          3. 未充分尝试就把步骤标记 blocked → 恢复 doing 再试
        """
        hints: List[str] = []
        if batch["repeats"]:
            name, err = batch["repeats"][0]
            hints.append(
                f"[System] 检测到与之前完全相同的失败调用：`{name}`"
                f"（错误：{err[:160]}）。原样重试不会改变结果，禁止重复。"
                "请换参数/换工具/换思路；或用 code_run 写诊断脚本定位根因；"
                "确认此路不通后，把对应计划步骤标记 blocked 并继续后续步骤。"
            )
        elif self._fail_streak >= 2:
            name, err = self._last_fail or ("", "")
            hints.append(
                f"[System] 已连续失败 {self._fail_streak} 次"
                f"（最近：{name} → {err[:160]}）。不要停下来问用户是否继续——"
                "请依据错误信息自动修复：先读完整错误定位原因，再换一种方法重试"
                "（不同工具/不同参数/先用 code_run 诊断）。同一思路只试一次。"
            )
        if self._premature_block(tool_calls, raw_results, step_fails_before):
            hints.append(
                f"[System] 步骤刚被标记 blocked，但本步骤至今只失败了 "
                f"{step_fails_before} 次——尝试不够。请先用 update_plan 把它恢复 "
                "doing，再依据上面的错误信息换方案重试（至少 2 种不同方法）；"
                "只有多种方法都确认不可行后，才标记 blocked 并写明原因，然后继续"
                "后续步骤。不要向用户确认「是否继续」。"
            )
        return hints

    def _premature_block(self, tool_calls: List[ToolCall], raw_results: List[Any],
                         step_fails_before: int) -> bool:
        """本批把某步骤标记为 blocked，且该步骤的失败尝试次数不足。

        命中即消耗一次提醒额度（``_block_nudge``）——守卫只提醒有限次，
        模型坚持放弃时最终尊重它的决定。
        """
        if self._block_nudge >= _BLOCK_NUDGE_MAX:
            return False
        if step_fails_before >= _MIN_ATTEMPTS_BEFORE_BLOCK:
            return False
        for tc, res in zip(tool_calls, raw_results):
            if tc.name != "update_plan":
                continue
            if str(tc.args.get("status", "")).strip().lower() != "blocked":
                continue
            ok, _ = self._result_ok(res)
            if ok:
                self._block_nudge += 1
                return True
        return False

    # ------------------------------------------------------------------
    # 无工具调用时的判定（对齐 GenericAgent 的 do_no_tool）
    # ------------------------------------------------------------------
    def _no_tool_event(self, resp: LLMResponse) -> str:
        content = (resp.content or "").strip()
        if not content and not (resp.reasoning or "").strip():
            self._empty_retry += 1
            return "retry" if self._empty_retry <= _MAX_EMPTY_RETRY else "done"
        tail = content[-120:]
        if any(m in tail for m in _TRUNCATION_MARKS) or (
            content.endswith("</summary>") and len(content) < 100
        ):
            self._empty_retry += 1
            return "retry" if self._empty_retry <= _MAX_EMPTY_RETRY else "done"
        if self._looks_like_unexecuted_code(content):
            return "reask"
        return "done"

    @staticmethod
    def _looks_like_unexecuted_code(content: str) -> bool:
        """恰好一个大代码块 + 结尾 + 正文几乎没有自然语言 → 疑似该执行却没执行。"""
        blocks = _CODE_BLOCK_RE.findall(content)
        if len(blocks) != 1:
            return False
        m = _CODE_BLOCK_RE.search(content)
        if not m or m.end() < len(content) - 2:
            return False
        residual = content.replace(m.group(0), "")
        residual = _TAG_STRIP_RE.sub("", residual)
        return len(re.sub(r"\s+", "", residual)) <= 30

    # ------------------------------------------------------------------
    def _cancelled(self) -> bool:
        try:
            return bool(self.callbacks.is_cancelled())
        except Exception:
            return False

    @staticmethod
    def _is_context_error(err: str) -> bool:
        e = (err or "").lower()
        return any(k in e for k in (
            "context_length", "context length", "context window",
            "maximum context", "prompt is too long", "prompt too long",
            "too many tokens", "token limit",
        ))
