"""闲时后台自动补全每本书的章节目录摘要。

设计要点（对应需求）：
- 无论辅助引擎是否「启用」，都进行本任务。
- 若辅助引擎「启用且当前可用（已加载且未关闭）」→ 直接复用，节奏 **15 分钟/章**，
  用完不卸载（保持常驻，避免反复加载）。
- 若「未启用且不可用」→ 临时 create + load 一个辅助引擎实例，用完 ``close()`` 释放，
  节奏 **5 分钟/章**。
- 已生成过摘要的章节不重复生成（状态以 sidecar JSON 落盘，与 ``rag/documents/*.md``
  缓存同目录，避免动 Lance 表结构 / 旧表全表重写的 OOM 风险）。
- 逐章基于「全章全文」调辅助模型生成 **50-100 字**摘要；≤4000 字的短章节优先
  整章直摘（质量最高），超过 4000 字的长章节一律按 **每 3000 字一个 map** 走
  map-reduce（分块摘 → 逐级归并），以保证长章摘要不被单段直摘压掉细节。
- 全部章节完成后，结合目录结构与各章摘要，生成「真正的全书摘要」。
- 摘要过程中若发现某章正文内含未被捕获的小节标题，就把这些小节补进目录结构
  （顺便完善章节目录）。
- 闲时才干活：由调用方传入 ``is_idle()``；仅在空闲且到达节奏间隔时才处理一章。

调度与模型调用都做了防御：任何单步异常都不终止整个后台线程，只跳过本轮。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .document_outline import Chapter, extract_chapters
from ..model_backends import create_aux_llm_backend

logger = logging.getLogger(__name__)

# 估算 token 时的「字符→token」系数（中文为主，高估以留余量、避免溢出）。
_CHARS_PER_TOKEN = 1.5
# 整章直摘时预留的指令+输出 token 预算。
_WHOLE_OVERHEAD = 800
# 节奏下限（秒）。
_INTERVAL_SHARED = 900   # 启用/可用：15 分钟/章
_INTERVAL_TEMP = 300     # 临时加载：5 分钟/章
# 调度器轮询间隔（秒）。
_POLL_INTERVAL = 30
# 单章内最多补多少个被遗漏的小节（防止极端情况下结构爆炸）。
_MAX_SUBS_PER_CHAPTER = 12

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def acquire_aux(config: Any, aux_getter) -> Tuple[Optional[Any], bool, int]:
    """取得一个可用的辅助模型后端。

    返回 ``(backend, owned, interval_seconds)``：
      * ``owned=True`` 表示本函数临时创建并加载，调用方用完必须 ``release_aux()``。
      * ``interval_seconds`` 是这一轮的节奏下限（shared=900 / temp=300）。

    规则（严格对应需求的三条）：
      * 当前已有「可用」的辅助引擎（``aux_getter`` 返回未关闭的实例）→ 直接复用，
        不卸载（``owned=False``），节奏 15 分钟。
      * 否则若配置为「启用」→ 创建并加载，加载后保持常驻（后续轮次会变成上一种
        情况），节奏 15 分钟。
      * 否则（**未启用且不可用**）→ 临时 create + load，用完卸载（``owned=True``），
        节奏 5 分钟。
    """
    try:
        current = aux_getter() if aux_getter is not None else None
    except Exception:
        current = None
    if current is not None and not getattr(current, "_closed", False):
        return current, False, _INTERVAL_SHARED

    enabled = bool(getattr(config, "aux_llm_enabled", False))
    try:
        backend = create_aux_llm_backend(config)
        backend._ensure_loaded()
    except Exception as e:  # 模型路径无效 / 加载失败 → 本轮跳过
        logger.warning("[summary] 临时加载辅助引擎失败，跳过本轮: %s", e)
        return None, False, _INTERVAL_SHARED
    if enabled:
        # 启用状态下保持常驻，后续轮次直接复用，不卸载。
        return backend, False, _INTERVAL_SHARED
    return backend, True, _INTERVAL_TEMP


def release_aux(backend: Optional[Any], owned: bool) -> None:
    """用完释放辅助引擎；仅 ``owned=True``（临时加载）时才 close()。"""
    if owned and backend is not None:
        try:
            backend.close()
        except Exception as e:  # pragma: no cover - 释放失败不致命
            logger.warning("[summary] 释放临时辅助引擎失败（忽略）: %s", e)


# 摘要/归纳类调用的生成上限：章节摘要(~200 token)与全书摘要(300-500字≈700 token)
# 都远小于此。强制封顶可防止「模型不吐 EOS」时一路生成到 config.max_tokens(默认
# 32768) —— 那会既慢又吃显存，是闲时调度器后台跑摘要时的真实失控风险。
_SUMMARY_MAX_TOKENS = 1024
# 整章直摘的字数上限：超过该字数的章节一律走 map-reduce（即便全文能塞进上下文
# 窗口），以保证长章摘要质量、不被单段直摘压掉细节；短章节才优先整章直摘。
_WHOLE_MAX_CHARS = 4000
# map-reduce 的切块粒度：每 ~3000 字切一个 map（按段落边界尽量不截断句子）；以及
# map 阶段每块摘要的输出 token 上限（2-3 句中文约 60-150 token，留足余量）。
_MAP_CHUNK_CHARS = 3000
_MAP_CHUNK_MAX_TOKENS = 400
# 并行批大小——多个分片一次性喂进辅助模型做批量前向，显著提升长章节摘要速度
# （MLX 后端走真·批量 [B,T] 并行解码，其余后端顺序兜底），上限 5 以控显存。
_MAP_CHUNK_BATCH = 5


def _safe_chat(
    aux, messages: List[Dict[str, str]], temperature: float, max_tokens: int = _SUMMARY_MAX_TOKENS
) -> str:
    """用辅助模型跑一段非流式摘要/归纳；异常返回空串。

    调用期间临时调整后端配置，结束还原：
      * 生成长度封顶到 ``max_tokens`` —— 防止个别模型不吐 EOS 时狂生成到默认上限
        （默认 32768）既慢又吃显存；
      * 关闭「思考模式」（``llm_enable_thinking``）—— 摘要是单轮直答任务，开启
        思考会把 token 预算全耗在推理链上、正文还没输出就到上限，导致整章直摘
        返回空（Qwen3.5 系实测）。关闭后直接作答、更快更稳。
    """
    cfg = getattr(aux, "config", None)
    saved: Dict[str, Any] = {}
    if cfg is not None:
        for attr, val in (("max_tokens", max_tokens), ("llm_enable_thinking", False)):
            try:
                saved[attr] = getattr(cfg, attr, None)
                setattr(cfg, attr, val)
            except Exception:
                saved.pop(attr, None)
    try:
        buf: List[str] = []
        for ev in aux.chat_stream(messages, tools=None, temperature=temperature):
            if ev.type == "content":
                buf.append(ev.content)
        return "".join(buf).strip()
    except Exception as e:
        logger.warning("[summary] 辅助模型调用失败: %s", e)
        return ""
    finally:
        if cfg is not None:
            for attr, old in saved.items():
                try:
                    setattr(cfg, attr, old)
                except Exception:
                    pass


def _is_error_text(text: str) -> bool:
    """辅助模型返回里是否像「上下文溢出」之类的报错（而非正常摘要）。"""
    low = (text or "").lower()
    if "context length" in low or "context_length" in low or "maximum context" in low:
        return True
    if "exceed" in low and "context" in low:
        return True
    return False


def _clean_summary(text: str) -> str:
    """清理辅助模型输出的摘要：去代码围栏 / 常见前缀 / 首尾引号。"""
    t = (text or "").strip()
    if not t:
        return ""
    if t.startswith("```"):
        t = t.strip("`")
        t = re.sub(r"^[A-Za-z0-9_-]+\n", "", t)
    t = re.sub(r"^(摘要[:：]|【摘要】|以下.*?摘要[：:])", "", t, flags=re.S).strip()
    if len(t) >= 2 and t[0] in "「\"'\"“" and t[-1] in "」\"'\"”":
        t = t[1:-1].strip()
    return t


def _context_window(aux, config) -> int:
    """辅助引擎有效上下文窗口（token）。优先取后端属性，回落到配置默认值。"""
    cw = getattr(aux, "context_window", None)
    if isinstance(cw, int) and cw >= 512:
        return cw
    return int(getattr(config, "aux_max_context_tokens", 16384) or 16384)


def _estimate_tokens(text: str) -> int:
    """中文为主，按 _CHARS_PER_TOKEN 估算 token（高估以留余量、避免溢出）。"""
    return max(1, int(len(text) * _CHARS_PER_TOKEN))


def summarize_chapter(
    aux, book_title: str, chapter_title: str, chapter_text: str, config=None
) -> str:
    """生成单章 50-100 字摘要，务必基于「全章全文」并保证质量。

    - 整章能放进辅助引擎上下文（含指令与输出预算）→ 直接「整章直摘」，质量最高。
    - 放不下（窗口小 / 章节过长）→ map-reduce：按段落切块，每块先摘（map），
      再逐级归并（reduce）成最终章摘要，既保证质量又不溢出上下文。
    """
    body = (chapter_text or "").strip()
    if not body:
        return ""
    ctx = _context_window(aux, config)
    # 长章节（>4000 字）一律走 map-reduce：单段直摘会压掉细节、损害摘要质量，即便
    # 全文能塞进上下文窗口也不整章直摘。短章节才优先整章直摘（质量最高）。
    if len(body) <= _WHOLE_MAX_CHARS and \
            _estimate_tokens(body) + _WHOLE_OVERHEAD <= ctx:
        s = _summarize_whole(aux, book_title, chapter_title, body)
        if s:
            return s
        logger.warning(
            "[summary] 整章直摘失败，降级 map-reduce: %s / %s", book_title, chapter_title
        )
    return _summarize_mapreduce(aux, book_title, chapter_title, body, ctx)


def _summarize_whole(aux, book_title, chapter_title, body) -> str:
    system = (
        "你是严谨的图书摘要助手。请阅读给出的【完整章节正文】，只输出一段中文摘要，"
        "字数严格控制在 50-100 字之间，概括该章节的核心论点、关键事实或情节发展；"
        "不要列要点、不要任何前言与解释、不要标题。"
    )
    user = f"《{book_title}》章节：【{chapter_title}】\n\n完整章节正文：\n{body}"
    text = _safe_chat(
        aux,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    if text and not _is_error_text(text):
        return _clean_summary(text)
    return ""


def _summarize_mapreduce(aux, book_title, chapter_title, body, ctx) -> str:
    chunks = _split_by_paragraph(body, _MAP_CHUNK_CHARS)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return _map_chunk(aux, book_title, chapter_title, chunks[0])
    # map 阶段：分片并行（批量前向，batch=5）一次性处理，显著快于逐条。
    parts = _map_chunks_batch(aux, book_title, chapter_title, chunks)
    # 个别分片若批量失败（多为溢出），降级走单条 map（含对半递归兜底）。
    for i, p in enumerate(parts):
        if not p:
            parts[i] = _map_chunk(aux, book_title, chapter_title, chunks[i])
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return _reduce_summaries(aux, book_title, chapter_title, parts, ctx)


def _map_chunk_messages(book_title: str, chapter_title: str, chunk: str) -> List[Dict[str, str]]:
    """构造单块正文的 map 提示（与 _map_chunk 共用，便于批量拼接）。"""
    system = (
        "你是图书摘要助手。阅读下面这一段【章节正文片段】，用 2-3 句中文概括这部分"
        "的关键内容、事实或论点；只输出摘要，不要前言与标题。"
    )
    user = f"《{book_title}》【{chapter_title}】片段：\n{chunk}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _split_by_paragraph(text: str, size: int) -> List[str]:
    """按段落切分，使每块接近 size 字符；单段超长则按句子/硬切。"""
    if len(text) <= size:
        return [text]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return [text[i : i + size] for i in range(0, len(text), size)]
    out: List[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = (cur + "\n\n" + p) if cur else p
            continue
        if cur:
            out.append(cur)
        if len(p) > size:
            for i in range(0, len(p), size):
                out.append(p[i : i + size])
            cur = ""
        else:
            cur = p
    if cur:
        out.append(cur)
    return [o for o in out if o.strip()]


def _map_chunk(aux, book_title, chapter_title, chunk, depth: int = 0) -> str:
    """map 阶段：对一块正文生成 2-3 句短摘要；若溢出则对半递归再合并。"""
    text = _safe_chat(
        aux, _map_chunk_messages(book_title, chapter_title, chunk), temperature=0.3
    )
    if text and not _is_error_text(text):
        return _clean_summary(text)
    # 失败（多半溢出）→ 对半切后再摘并合并
    if depth < 4 and len(chunk) > 800:
        a, b = _half_split(chunk)
        sa = _map_chunk(aux, book_title, chapter_title, a, depth + 1)
        sb = _map_chunk(aux, book_title, chapter_title, b, depth + 1)
        if sa or sb:
            return _merge_two(aux, book_title, chapter_title, sa, sb)
    return ""


def _map_chunks_batch(
    aux, book_title: str, chapter_title: str, chunks: List[str]
) -> List[str]:
    """map 阶段批量并行：把分片按 batch=5 分组，一次性喂辅助模型批量生成。

    依赖后端的 ``generate_batch``（MLX 走真·批量 [B,T] 并行前向；其余后端顺序兜底）。
    调用期间临时关闭思考模式、封顶输出 token，避免长章节 map 被推理链拖慢/失控。
    """
    if not chunks:
        return []
    cfg = getattr(aux, "config", None)
    saved: Dict[str, Any] = {}
    if cfg is not None:
        for attr, val in (("llm_enable_thinking", False), ("max_tokens", _MAP_CHUNK_MAX_TOKENS)):
            try:
                saved[attr] = getattr(cfg, attr, None)
                setattr(cfg, attr, val)
            except Exception:
                saved.pop(attr, None)
    try:
        msgs_list = [_map_chunk_messages(book_title, chapter_title, ch) for ch in chunks]
        out = aux.generate_batch(
            msgs_list,
            batch_size=_MAP_CHUNK_BATCH,
            temperature=0.3,
            max_tokens=_MAP_CHUNK_MAX_TOKENS,
        )
        return [_clean_summary(x) if x else "" for x in out]
    except Exception as e:
        logger.warning("[summary] 批量 map 失败，降级逐条: %s", e)
        return [_map_chunk(aux, book_title, chapter_title, ch) for ch in chunks]
    finally:
        if cfg is not None:
            for attr, old in saved.items():
                try:
                    setattr(cfg, attr, old)
                except Exception:
                    pass


def _half_split(text: str) -> Tuple[str, str]:
    """在文本中点附近找段落/句末边界对半切，尽量不截断句子。"""
    mid = len(text) // 2
    for pat in (r"\n\s*\n", r"[。！？!?；;\n]"):
        best = None
        for m in re.finditer(pat, text):
            pos = m.end()
            if best is None or abs(pos - mid) < abs(best - mid):
                best = pos
        if best is not None:
            return text[:best], text[best:]
    return text[:mid], text[mid:]


def _merge_two(aux, book_title, chapter_title, sa: str, sb: str) -> str:
    parts = [x for x in (sa, sb) if x]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return _call_reduce(aux, book_title, chapter_title, parts, 2)


def _reduce_summaries(aux, book_title, chapter_title, parts, ctx) -> str:
    """reduce 阶段：把若干短摘要逐级归并成最终 50-100 字章摘要。"""
    if len(parts) == 1:
        return _clean_summary(parts[0])
    combined = "\n".join(f"· {p}" for p in parts)
    if _estimate_tokens(combined) + 600 <= ctx:
        return _clean_summary(_call_reduce(aux, book_title, chapter_title, parts, 3))
    mid = len(parts) // 2
    left = _reduce_summaries(aux, book_title, chapter_title, parts[:mid], ctx)
    right = _reduce_summaries(aux, book_title, chapter_title, parts[mid:], ctx)
    return _reduce_summaries(aux, book_title, chapter_title, [left, right], ctx)


def _call_reduce(aux, book_title, chapter_title, parts, cap_sentences: int) -> str:
    text = "\n".join(f"· {p}" for p in parts)
    n = max(2, min(cap_sentences, 4))
    system = (
        f"你是图书摘要助手。把下面这些【同章节不同片段的摘要】合并成一段 {n} 句以内"
        f"的中文摘要，保留关键内容；只输出合并后的摘要，不要前言与标题。"
    )
    user = f"《{book_title}》【{chapter_title}】片段摘要：\n{text}"
    return _safe_chat(
        aux,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )


def compose_full_summary(
    aux,
    book_title: str,
    chapters: List[Dict[str, Any]],
    entries: List[Dict[str, Any]],
) -> str:
    """结合目录结构与各章摘要，在「所有章节完成后做一次」连贯的「全书摘要」（300-500 字）。"""
    lines: List[str] = []
    for c, e in zip(chapters, entries):
        indent = "  " * max(0, int(c.get("level", 1)) - 1)
        s = (e.get("summary") or "").strip()
        if s:
            lines.append(f"{indent}- 【{c.get('title', '')}】{s}")
    if not lines:
        return ""
    toc = "\n".join(lines)
    system = (
        "你是图书内容主编。根据下面的目录结构与各章摘要，写一段 300-500 字的中文"
        "「全书摘要」，连贯地概括全书主旨、脉络与各章贡献；不要逐章罗列、不要前言。"
    )
    user = f"《{book_title}》\n目录与章节摘要：\n{toc}"
    full = _safe_chat(
        aux,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    full = _clean_summary(full)
    # 控制篇幅：过长则再压缩一次到 500 字内。
    if len(full) > 520:
        system2 = (
            "你是图书内容主编。把下面这段「全书摘要」压缩成 300-500 字，保留核心主旨"
            "与脉络，连贯通顺；只输出压缩后的摘要，不要前言。"
        )
        full = _clean_summary(
            _safe_chat(
                aux,
                [
                    {"role": "system", "content": system2},
                    {"role": "user", "content": full},
                ],
                temperature=0.3,
            )
        )
    return full


def extract_subchapters(text: str) -> List[Chapter]:
    """从章节正文里抓出未被顶层目录捕获的小节标题（用于顺便完善目录结构）。"""
    subs: List[Chapter] = []
    seen: set = set()
    for m in _HEADING_RE.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        if not title or title in seen:
            continue
        seen.add(title)
        subs.append(Chapter(level, title, ""))
        if len(subs) >= _MAX_SUBS_PER_CHAPTER:
            break
    return subs


def _slice_sub(parent_body: str, title: str, level: int) -> str:
    """从 parent 正文里切出标题为 ``title``（level 级）的小节正文。

    用于「目录结构完善」场景：被补进目录的小节并不在 ``extract_chapters`` 的顶层
    输出里，需要从父章正文里按标题定位切片得到它的正文。
    """
    try:
        pat = re.compile(
            r"^#{" + str(level) + r"}\s+" + re.escape(title) + r"\s*$", re.MULTILINE
        )
        m = pat.search(parent_body)
        if not m:
            return ""
        start = m.end()
        nxt = re.compile(r"^#{1," + str(level) + r"}\s+", re.MULTILINE).search(
            parent_body, m.start() + 1
        )
        end = nxt.start() if nxt else len(parent_body)
        return parent_body[start:end].strip()
    except re.error:
        return ""


def render_enriched_digest(
    chapters: List[Dict[str, Any]], state: Dict[str, Any]
) -> Optional[str]:
    """把「目录结构 + 逐章摘要 + 全书摘要」渲染成缩略版文本。

    用于 ``rag_outline`` 查找文档缩略版时优先采用「已逐章摘要并优化」的版本。
    若该书还没有任何逐章摘要（``state`` 为空 / 全为空串），返回 ``None``，
    调用方应回退到普通 TOC。
    """
    if not chapters:
        return None
    entries = state.get("entries") or []
    has_summary = any(
        (e.get("summary") or "").strip() for e in entries[: len(chapters)]
    )
    if not has_summary:
        return None

    full = (state.get("full") or "").strip()
    out: List[str] = ["# 目录与章节摘要"]
    if full:
        out.append("")
        out.append("## 全书摘要")
        out.append(full)
    out.append("")
    out.append("## 章节摘要")
    for i, ch in enumerate(chapters):
        level = int(ch.get("level", 1))
        title = ch.get("title", "")
        indent = "  " * max(0, level - 1)
        summary = ""
        if i < len(entries):
            summary = (entries[i].get("summary") or "").strip()
        if summary:
            out.append(f"{indent}- 【{title}】{summary}")
        else:
            out.append(f"{indent}- {title}")
    return "\n".join(out)


class BookSummaryScheduler(threading.Thread):
    """闲时后台调度器：周期性地在空闲时处理一本书的一章摘要。"""

    def __init__(
        self,
        engine_getter,
        config: Any,
        aux_getter,
        is_idle,
        idle_threshold: int = 120,
    ):
        super().__init__(daemon=True, name="book-summary-scheduler")
        self._engine_getter = engine_getter
        self._config = config
        self._aux_getter = aux_getter
        self._is_idle = is_idle
        self._idle_threshold = idle_threshold
        self._stop = False
        self._last_step_ts = 0.0
        self._last_interval = _INTERVAL_SHARED

    def stop(self) -> None:
        self._stop = True

    @property
    def _engine(self):
        try:
            return self._engine_getter() if self._engine_getter is not None else None
        except Exception:
            return None

    def _next_job(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """挑一本还有未摘要章节的书，返回 ``(doc, idx)``。

        ``idx == -1`` 表示该书章节已全部摘要完、但还没有全书摘要（该生成全书摘要）。
        """
        engine = self._engine
        if engine is None:
            return None
        try:
            docs = engine.get_documents_for_summary()
        except Exception:
            return None
        for doc in docs:
            chapters = doc.get("chapters") or []
            if not chapters:
                continue
            state = doc.get("summary_state") or {
                "entries": [], "full": "", "done": False, "skipped": False,
            }
            # 跳过已标记「无法抽取章节正文」的书，避免空转重试。
            if state.get("skipped"):
                continue
            entries = state.get("entries") or []
            # 与 chapters 对齐：长度不够则补空条目（标题取当前结构，便于搬运旧摘要）。
            if len(entries) < len(chapters):
                entries = entries + [
                    {"level": c.get("level"), "title": c.get("title"), "summary": ""}
                    for c in chapters[len(entries):]
                ]
            idx = next(
                (i for i, e in enumerate(entries) if not (e.get("summary") or "").strip()),
                None,
            )
            if idx is None:
                # 章节全摘要完 → 还没有全书摘要就交给 idx==-1 分支。
                if not (state.get("full") or "").strip():
                    return doc, -1
                continue
            return doc, idx
        return None

    def _chapter_body(self, fresh, stored_chapters, idx: int) -> str:
        """取第 ``idx`` 章的正文。

        顶层章节直接从 ``extract_chapters`` 的结果按 标题+level 匹配拿到正文；
        「目录完善」补进的小节不在 fresh 顶层，则递归从最近的前驱顶层章正文里
        按标题切片得到。
        """
        ch = stored_chapters[idx]
        title = ch.get("title", "")
        level = int(ch.get("level", 1))
        for f in fresh:
            if f.title == title and int(getattr(f, "level", 1)) == level:
                return f.text
        parent_idx = None
        for j in range(idx - 1, -1, -1):
            if int(stored_chapters[j].get("level", 1)) < level:
                parent_idx = j
                break
        if parent_idx is None:
            return ""
        parent_body = self._chapter_body(fresh, stored_chapters, parent_idx)
        return _slice_sub(parent_body, title, level)

    def _process_chapter(
        self, doc: Dict[str, Any], state: Dict[str, Any], aux
    ) -> None:
        """处理某本书的下一章（或补全书摘要）。"""
        md_path = doc.get("md_path", "")
        markdown = self._engine.read_markdown(md_path)
        if not markdown:
            return
        fresh = extract_chapters(markdown)
        stored_chapters = doc.get("chapters") or []
        if not stored_chapters:
            stored_chapters = [{"level": c.level, "title": c.title} for c in fresh]
        if not fresh and not stored_chapters:
            # 抽不出任何章节结构（极少数纯文本/扫描件）→ 标记跳过，避免空转重试。
            state.update({"entries": [], "full": "", "done": True, "skipped": True})
            self._engine.save_chapter_summary_state(
                doc["doc_name"], state, revised_chapters=stored_chapters
            )
            logger.info("[summary] 无法抽取章节正文，跳过: %s", doc["doc_name"])
            return

        # 以「表内 chapters」为权威结构（含此前完善补入的小节），逐章取正文：
        # 顶层从 fresh 取，完善出的小节从父章正文切片；并按标题搬回已有摘要，
        # 这样结构被修订/补全后跨轮不丢、不重复摘要。
        stored_entries = state.get("entries") or []
        merged: List[Dict[str, Any]] = []
        for i, ch in enumerate(stored_chapters):
            title = ch.get("title", "")
            level = int(ch.get("level", 1))
            summary = ""
            if i < len(stored_entries) and stored_entries[i].get("title") == title:
                summary = stored_entries[i].get("summary", "")
            body = self._chapter_body(fresh, stored_chapters, i)
            merged.append(
                {"level": level, "title": title, "summary": summary, "body": body}
            )

        target = next(
            (i for i, m in enumerate(merged) if not (m["summary"] or "").strip()), None
        )
        if target is None:
            # 都摘要完了，留给 idx==-1 分支生成全书摘要；这里不写 full，避免反复。
            return

        ch = merged[target]
        summary = summarize_chapter(
            aux, doc["doc_name"], ch["title"], ch["body"] or "", config=self._config
        )
        if not summary:
            # 生成失败：不写空摘要，保留未摘要状态，下轮重试。
            return
        merged[target]["summary"] = summary

        # 结构完善：本章正文里未被目录捕获的小节标题，补进目录（去重已存在的标题，
        # 避免把 extract_chapters 已捕获的嵌套标题重复插入）。
        existing_titles = {c.get("title", "") for c in stored_chapters}
        subs = [s for s in extract_subchapters(ch["body"] or "")
                if s.title not in existing_titles]
        insert_at = target + 1
        for s in subs:
            merged.insert(
                insert_at,
                {"level": s.level, "title": s.title, "summary": "", "body": ""},
            )
            insert_at += 1

        revised_chapters = [{"level": m["level"], "title": m["title"]} for m in merged]
        new_state = {
            "entries": [
                {"level": m["level"], "title": m["title"], "summary": m["summary"]}
                for m in merged
            ],
            "full": state.get("full", "") or "",
            "done": all((m["summary"] or "").strip() for m in merged),
            "skipped": False,
        }
        self._engine.save_chapter_summary_state(
            doc["doc_name"], new_state, revised_chapters=revised_chapters
        )
        logger.info(
            "[summary] 章节摘要完成(%d/%d): %s / %s",
            target + 1, len(merged), doc["doc_name"], ch["title"],
        )

    def _process_full(self, doc: Dict[str, Any], state: Dict[str, Any], aux) -> None:
        chapters = doc.get("chapters") or []
        entries = state.get("entries") or []
        full = compose_full_summary(aux, doc["doc_name"], chapters, entries)
        if not full:
            return
        state["full"] = full
        state["done"] = True
        self._engine.save_chapter_summary_state(doc["doc_name"], state, revised_chapters=chapters)
        logger.info("[summary] 全书摘要完成: %s", doc["doc_name"])

    def run(self) -> None:
        logger.info("[summary] 闲时章节摘要调度器已启动")
        while not self._stop:
            try:
                if self._stop:
                    break
                # 闲时判定 + 节奏间隔。
                if self._is_idle is not None and not self._is_idle():
                    time.sleep(_POLL_INTERVAL)
                    continue
                now = time.monotonic()
                if now - self._last_step_ts < self._last_interval:
                    time.sleep(_POLL_INTERVAL)
                    continue

                job = self._next_job()
                if job is None:
                    time.sleep(_POLL_INTERVAL)
                    continue
                doc, idx = job

                aux, owned, interval = acquire_aux(self._config, self._aux_getter)
                if aux is None:
                    # 加载失败：按当前节奏下限节流重试，避免每轮刷失败日志。
                    self._last_step_ts = time.monotonic()
                    self._last_interval = interval
                    time.sleep(_POLL_INTERVAL)
                    continue
                try:
                    state = doc.get("summary_state") or {
                        "entries": [], "full": "", "done": False, "skipped": False,
                    }
                    if idx == -1:
                        self._process_full(doc, state, aux)
                    else:
                        self._process_chapter(doc, state, aux)
                finally:
                    release_aux(aux, owned)
                self._last_step_ts = time.monotonic()
                self._last_interval = interval
            except Exception as e:  # 单轮异常不影响整个后台线程
                logger.warning("[summary] 调度器本轮异常（忽略）: %s", e)
                time.sleep(_POLL_INTERVAL)
        logger.info("[summary] 闲时章节摘要调度器已停止")
