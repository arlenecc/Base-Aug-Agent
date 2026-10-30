"""Local and remote language-model backends used by the Agent.

Optional runtimes are imported lazily so the default OpenAI-compatible mode
continues to work without MLX, ONNX Runtime, Transformers, or llama.cpp.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol

from .llm_client import LLMClient, LLMError, StreamEvent

logger = logging.getLogger(__name__)

# XML 工具调用的闭合标签，用字符拼接而非字面量，规避部分工具链对 XML 片段的
# 误判；同时让 _xml_tool_calls 的正参保持可读。
_LT = chr(60)
_GT = chr(62)
FE_RE = _LT + "/function" + _GT
PE_RE = _LT + "/parameter" + _GT
FN_RE = _LT + "function" + _GT
# 思考标签。qwen3.x 模板把 think 开标签放在 generation prompt 末尾，模型续写
# 思考正文、自己补 think 闭标签——所以生成内容里只有闭标签（见
# _prompt_opens_thinking / _TaggedOutputParser.begin_reasoning）。
THO = _LT + "think" + _GT
THE = _LT + "/think" + _GT

LLM_BACKENDS = ("openai", "mlx_lm", "mlx_vlm", "onnx", "llama_cpp")
# 辅助模型（#140）只支持进程内直载：没有网络端点可配。
AUX_LLM_BACKENDS = ("mlx_lm", "mlx_vlm", "onnx", "llama_cpp")


class LLMBackend(Protocol):
    def list_models(self) -> List[str]: ...

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Iterator[StreamEvent]: ...

    def close(self) -> None: ...


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_model_path(path: str, *, expect_file: bool = False) -> str:
    """Resolve a configured local model path without performing network access.

    相对路径按以下优先级查找（取首个真实存在的为准）：

    1. ``Path.cwd() / models``        —— 便携/开发：运行时目录含 models 时优先
    2. ``project_root() / models``    —— 开发期 ``Base-Aug-Agent/models``
    3. ``sys._MEIPASS / models``      —— PyInstaller 单文件(onefile)解压目录
    4. ``<exe>.parent / models``      —— 可执行文件同级（onedir / .app 的
       ``Contents/MacOS/models``，★打包默认落点）
    5. ``<exe>.parent.parent / Resources / models`` —— macOS .app 标准位
       ``Contents/Resources/models``
    6. ``<exe>.parents[2] / models``  —— ``.app`` 包根（``.app/models``）

    打包时只需把 ``models/`` 作为数据资源随可执行文件发布（如
    ``--add-data "models:models"`` 落到 ``Contents/MacOS/models``，命中 ④），
    即可被自动定位，业务代码无需改动。回退报错路径取「项目根」以保持开发期
    友好。
    """
    raw = os.path.expandvars(os.path.expanduser((path or "").strip()))
    if not raw:
        raise LLMError("未配置本地模型路径")
    candidate = Path(raw)
    if not candidate.is_absolute():
        executable = Path(sys.executable).resolve()
        meipass = getattr(sys, "_MEIPASS", None)
        root_cand = project_root() / candidate
        candidates = [
            Path.cwd() / candidate,                              # ① cwd
            root_cand,                                           # ② 项目根
            executable.parent / candidate,                      # ④ exe 同级（打包默认）
            executable.parent.parent / "Resources" / candidate, # ⑤ macOS .app Resources
        ]
        if meipass:                                             # ③ PyInstaller onefile
            candidates.insert(2, Path(meipass) / candidate)
        if len(executable.parents) >= 3:                        # ⑥ .app 包根
            candidates.append(executable.parents[2] / candidate)
        # 回退到项目根，给出开发期友好的报错路径。
        candidate = next((item for item in candidates if item.exists()), root_cand)
    candidate = candidate.resolve()
    if not candidate.exists():
        raise LLMError(f"本地模型不存在: {candidate}")
    if expect_file and not candidate.is_file():
        raise LLMError(f"需要模型文件，实际为目录: {candidate}")
    if not expect_file and not candidate.is_dir():
        raise LLMError(f"需要模型目录，实际为文件: {candidate}")
    return str(candidate)


def discover_local_models(models_dir: str = "models", backend: str = "") -> List[str]:
    """Return local model paths compatible with the selected backend."""
    try:
        root = Path(resolve_model_path(models_dir))
    except Exception:
        return []
    backend = (backend or "").lower()
    found: List[str] = []
    if backend == "llama_cpp":
        found.extend(str(p.resolve()) for p in root.rglob("*.gguf"))
    else:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            names = {p.name for p in child.iterdir() if p.is_file()}
            if backend in ("mlx_lm", "mlx_vlm") and not (
                "config.json" in names and any(n.endswith(".safetensors") for n in names)
            ):
                continue
            if backend == "onnx" and not any(n.endswith(".onnx") for n in names):
                continue
            found.append(str(child.resolve()))
    return sorted(found)


def _coerce_param(raw: str) -> Any:
    """把 XML ``<parameter>`` 的文本值转成合适的 Python 标量/容器。

    模型的工具参数在 XML 里全是纯文本，而工具侧期望的是 dict/list/number/bool
    （见 ``parse_args`` 之后的 JSON 反序列化）。所以：
    - ``true`` / ``false`` → bool；
    - 能 ``json.loads`` 的（``["a","b"]``、``{"k":1}``、``123``、``1.5``）→ 对应
      容器/数字；
    - 其余保持字符串（含首尾空白已剥离）。

    只认字面量，绝不 ``eval``——那会把模型输出当代码执行。
    """
    text = (raw or "").strip()
    if not text:
        return ""
    if text == "true":
        return True
    if text == "false":
        return False
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    # 标量与容器都要；字符串 "abc" 经 json.loads 会失败走到上面，所以能解析
    # 成功的都是非字符串类型。
    return obj


def _clone_prompt_cache(caches: Any) -> Optional[List[Any]]:
    """复制一份 prompt cache（用于「静态头缓存 + 每轮克隆」）。

    为什么需要：开启思考模式时，聊天模板会**剥掉**历史 assistant 消息里的
    ``<think>…</think>``（只要它后面还有消息），于是「重放历史」永远无法逐
    token 复现生成序列 → 跨轮 KV 前缀必然失配。而 ArraysCache（线性注意力）
    不可裁剪，失配就整份作废、全量 prefill。

    解法：缓存只保留**静态头**（system + 工具 schema，恒定不变），每轮生成时
    克隆一份出来用——克隆体被生成过程追加，母本始终停在头的长度，下一轮继续用。
    失败的代价从「prefill 整个 prompt」降到「只 prefill 头之后的对话部分」。

    复制失败返回 None（调用方退化为新建缓存）。
    """
    if caches is None:
        return None
    try:
        import copy
        import mlx.core as mx
    except Exception:  # pragma: no cover - 没有 mlx 就没有缓存可言
        return None
    def _copy_comp(v):
        """逐份复制：数组要真的复制（克隆体生成时会原地写入，共享会改坏母本），
        列表要新建容器（ArraysCache 用 list 存每层状态），None / 标量原样保留。"""
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return type(v)(_copy_comp(x) for x in v)
        if hasattr(v, "shape"):
            return mx.array(v)
        return v

    cloned: List[Any] = []
    try:
        for c in caches:
            new = copy.copy(c)          # 浅拷：拿到同类对象与属性壳
            state = getattr(c, "state", None)
            if state is not None:
                new.state = tuple(_copy_comp(s) for s in state)
            meta = getattr(c, "meta_state", None)
            if meta is not None:
                new.meta_state = tuple(_copy_comp(m) for m in meta)
            cloned.append(new)
    except Exception as e:  # pragma: no cover - 结构变了就放弃优化
        logger.debug("克隆 prompt cache 失败（退化为全量 prefill）: %s", e)
        return None
    return cloned or None


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    """两个 token 序列的最长公共前缀长度（KV 前缀复用的命中判定）。

    命中形态只有两种，都要求「新 prompt 与缓存序列逐 token 一致」：
      a) 缓存更长：缓存 = 上轮 prompt + 上轮生成，本轮 prompt 是它的前缀；
      b) 新 prompt 更长：恰好完全包含缓存（重试/追加场景）。
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _resolve_enable_thinking(config: Any) -> bool:
    """归一「思考模式」开关：未配置（None）视为**开启**。

    ``bool(None)`` 是 False，直接 ``bool(getattr(cfg, "llm_enable_thinking", True))``
    会把「用户从没设过」当成「用户关掉了」。后果是模板在渲染历史消息时剥掉
    ``<think>…</think>``，而生成提示是带 ``<think>`` 的 → 重放与生成序列错位 →
    KV 前缀缓存每轮整份作废。
    """
    value = getattr(config, "llm_enable_thinking", True)
    if value is None:
        return True
    if isinstance(value, str):
        # 手工编辑配置里常见 "false" —— bool("false") 是 True，必须按字面解析。
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


def _template_messages(messages: List[Dict[str, Any]],
                       think_opened: bool = True) -> List[Dict[str, Any]]:
    """把 agent 历史消息适配成 chat template 友好的形状。两件事：

    1. **tool_calls 的 arguments 还原成 dict。** agent 内部（以及 OpenAI 兼容
       API）里它是 JSON 字符串，而 qwen3.5-agentic-coder 的模板按
       ``tool_call.arguments is mapping`` 渲染——字符串会让整个参数段被静默
       丢掉，模型下一轮看到的就是「调了 task_plan 但没有任何参数」。
    2. **有 raw_content 时用它重建忠实序列（KV 前缀复用的关键）。**
       本地后端会把模型原始输出（含工具调用 XML）记在 ``raw_content`` 上；
       直接渲染它，模板输出就与上一轮生成序列**逐 token 一致**，下一轮
       prefill 才能命中缓存。做法（已用真实 tokenizer 验证对齐）：
         - ``content`` = raw_content（**原样**，不再手工补思考开标签）；
         - **不设置** ``reasoning_content``。
       旧实现补 ``THO + "\\n"`` 并把 ``reasoning_content`` 设成空字符串当「跳过
       自动拆分」的暗门；实测当前 tokenizer 会把空字符串渲染成
       ``<think>\\n\\n</think>\\n\\n``（多一个空思考块），序列从 assistant 块起
       就与生成时不一致 → 前缀缓存整份作废、每轮全量 prefill。
       raw 文本里已含工具调用的 XML，所以同时要**丢掉结构化的 tool_calls**，
       否则模板会再渲染一遍，变成重复内容。

    只有本地后端会设置 raw_content；HTTP 后端的历史原样透传，行为不变。
    """
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        copied = dict(msg)
        raw = copied.get("raw_content")
        if raw and copied.get("role") == "assistant":
            # raw 就是「生成提示之后模型真正吐出的那串文本」——把它原样作为
            # content，模板会按 enable_thinking 补上与生成提示相同的 `<think>\n`
            # 前缀，于是渲染结果与上一轮生成序列逐 token 一致（已实测）。
            #
            # 旧实现在这里再拼一个 THO + "\n"，并用 `reasoning_content = ""`
            # 当「跳过自动拆分」的暗门。实测当前 tokenizer 会把这个空字符串渲染
            # 成 `<think>\n\n</think>\n\n`（多出一个空思考块），序列从 assistant
            # 块起就对不齐 → 公共前缀掉到上轮 prompt 的最后一个 token，KV 缓存
            # 整份作废（每轮重新全量 prefill）。所以这里不再设置该字段。
            #
            # 第二个坑：raw 里若没有 `</think>`（模型直接吐工具调用、没写思考），
            # 模板同样会先插一个空思考块再拼原文。补一个闭合标签即可让整段按
            # 推理渲染、与生成时逐 token 对齐——多出来的 `</think>` 落在缓存区
            # **之后**，不影响前缀比对。
            copied["content"] = raw if ("</think>" in raw or not think_opened) else raw + "\n</think>"
            copied.pop("reasoning_content", None)
            # raw 里已含工具调用 XML，结构化字段再渲染一遍会重复。
            copied.pop("tool_calls", None)
            out.append(copied)
            continue
        calls = copied.get("tool_calls")
        if calls:
            normalized = []
            for tc in calls:
                if not isinstance(tc, dict):
                    normalized.append(tc)
                    continue
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn = dict(fn)
                    raw_args = fn.get("arguments")
                    if isinstance(raw_args, str) and raw_args.strip():
                        try:
                            fn["arguments"] = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            pass  # 保持原样，让模板/服务端自己处理
                    tc = {**tc, "function": fn}
                normalized.append(tc)
            copied["tool_calls"] = normalized
        out.append(copied)
    return out


def _prompt_opens_thinking(prompt: Any) -> bool:
    """判断 chat template 的 generation prompt 是否以思考开标签收尾。

    qwen3.x 系模板的 add_generation_prompt 输出 assistant 换行 +
    think 开标签 + 换行，模型续写的就是思考正文、自己补 think 闭标签。
    此时生成内容里没有 think 开标签，解析器必须由调用方预置 reasoning
    模式，否则整段思考会被当成普通 content 漏进对话区。

    只认末尾的开标签：历史消息里残留的 think 开标签不算。
    """
    text = prompt if isinstance(prompt, str) else ""
    if not text:
        return False
    tail = text.rstrip()
    return tail.endswith(THO)


class _TaggedOutputParser:
    """Turn local-model text tags into reasoning/content/tool StreamEvents.

    本地模型（MLX / ONNX / GGUF）没有 OpenAI 式的结构化 tool_calls 字段，
    一切以文本标签形式出现在生成内容里。两个必须知道的坑：

    1. **思考开标签在 prompt 里，不在生成内容里。** qwen3.x 系的模板以
       assistant 换行 + think 开标签 + 换行 结尾，模型接着写的就是思考正文，
       直到自己输出 think 闭标签——生成内容里**只有闭标签**。若按 content
       模式傻等 think 开标签，永远等不到，整段思考会被当成普通回复塞进对话区
       （「思考过程不显示」的直接原因）。所以 begin_reasoning() 让调用方
       按 prompt 实际情况预置模式。
    2. **工具调用 payload 有两种写法。** JSON（name/arguments）与
       XML（function=name 包裹 parameter=k）。旧实现只会 json.loads，
       XML 必然失败、tool_calls 恒为空——task_plan 永不被调用，任务计划面板
       一直空着。现在两种都认。
    """

    _OPENERS = (("<think>", "reasoning"), ("<tool_call>", "tool"))
    _CLOSERS = {"reasoning": "</think>", "tool": "</tool_call>"}

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "content"
        self._tool_buffer = ""
        self.tool_calls: List[Dict[str, Any]] = []

    def begin_reasoning(self) -> None:
        """告诉解析器：prompt 末尾已预置思考开标签，生成内容直接是思考正文。

        qwen3.x 模板的 generation prompt 以 think 开标签收尾，模型续写思考、
        自己补 think 闭标签。不调用本方法的话解析器停在 content 模式等一个
        永远不会出现的开标签，整段思考就漏进对话区了。
        """
        if self._mode == "content" and not self._buffer.strip():
            self._mode = "reasoning"

    @staticmethod
    def _xml_tool_calls(payload: str) -> List[Dict[str, Any]]:
        """解析 XML 工具调用 payload，返回 OpenAI 形状的 tool_call dict 列表。

        格式（qwen3.5-agentic-coder，见模型自带 chat_template.jinja）：一个块
        里可有多个 ``<function>``（并行调用），所以返回列表。任何一步不匹配都
        安静返回 ``[]``，由调用方决定是否再试 JSON 解析。
        """
        text = payload.strip().strip("`").strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
        if "<function" not in text:
            return []

        calls: List[Dict[str, Any]] = []
        # 两个正则都必须带闭合标签：漏掉闭合标签时 (.*?) 会匹配到空串，
        # 参数整段丢失——表现为「工具被调用了但没有任何参数」。
        # FE_RE / PE_RE 自身已含结尾的 '>'，不要再补一个。
        fn_re = r"<function\s*=\s*([^>\s]+)\s*>(.*?)\s*" + FE_RE
        param_re = r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)\s*" + PE_RE
        for fm in re.finditer(fn_re, text, re.DOTALL):
            name = fm.group(1).strip()
            body = fm.group(2)
            if not name:
                continue
            args: Dict[str, Any] = {}
            for pm in re.finditer(param_re, body, re.DOTALL):
                key = pm.group(1).strip()
                if not key:
                    continue
                args[key] = _coerce_param(pm.group(2))
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        return calls

    @staticmethod
    def _json_tool_calls(payload: str) -> List[Dict[str, Any]]:
        """解析 JSON 工具调用 payload（单对象或对象数组）。"""
        raw = payload.strip().strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return []
        objs = obj if isinstance(obj, list) else [obj]
        calls: List[Dict[str, Any]] = []
        for item in objs:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("function"), dict):
                function = item["function"]
                name = function.get("name", "")
                arguments = function.get("arguments", {})
            else:
                name = item.get("name", item.get("tool", ""))
                arguments = item.get("arguments", item.get("parameters", {}))
            if not name:
                continue
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                {
                    "id": str(item.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                    "type": "function",
                    "function": {"name": str(name), "arguments": arguments},
                }
            )
        return calls

    @classmethod
    def _parse_tool_calls(cls, payload: str) -> List[Dict[str, Any]]:
        """按「XML 优先、JSON 兜底」解析一个块的 payload。"""
        calls = cls._xml_tool_calls(payload)
        if calls:
            return calls
        calls = cls._json_tool_calls(payload)
        if calls:
            return calls
        logger.warning("本地模型返回了无法解析的 tool_call: %s", payload[:300])
        return []

    def _emit_tool_calls(self) -> None:
        """把 ``_tool_buffer`` 里攒下的 payload 解析成 tool_calls。"""
        payload = self._tool_buffer
        self._tool_buffer = ""
        if not payload.strip():
            return
        self.tool_calls.extend(self._parse_tool_calls(payload))

    def feed(self, text: str) -> Iterator[StreamEvent]:
        self._buffer += text or ""
        while self._buffer:
            if self._mode in self._CLOSERS:
                closer = self._CLOSERS[self._mode]
                idx = self._buffer.find(closer)
                if idx < 0:
                    if self._mode == "tool":
                        # 只把「不可能是闭合标签前缀」的部分搬进 _tool_buffer，
                        # 尾部按 closer 长度留悬念。旧实现整段倒空后清空 buffer，
                        # 于是跨 chunk 的闭合标签永远等不到——tool_calls 恒为空，
                        # 任务计划面板也就一直是空的。
                        keep = max(len(closer) - 1, 0)
                        move = max(0, len(self._buffer) - keep)
                        if move:
                            self._tool_buffer += self._buffer[:move]
                            self._buffer = self._buffer[move:]
                    else:
                        keep = max(len(closer) - 1, 0)
                        safe = self._buffer[:-keep] if keep and len(self._buffer) > keep else ""
                        if safe:
                            yield StreamEvent("reasoning", content=safe)
                            self._buffer = self._buffer[len(safe):]
                    break
                before = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(closer):]
                if self._mode == "reasoning":
                    if before:
                        yield StreamEvent("reasoning", content=before)
                else:
                    self._tool_buffer += before
                    self._emit_tool_calls()
                self._mode = "content"
                continue

            match = None
            for opener, mode in self._OPENERS:
                idx = self._buffer.find(opener)
                if idx >= 0 and (match is None or idx < match[0]):
                    match = (idx, opener, mode)
            if match is not None:
                idx, opener, mode = match
                if idx:
                    yield StreamEvent("content", content=self._buffer[:idx])
                self._buffer = self._buffer[idx + len(opener):]
                self._mode = mode
                continue

            keep = max(len(opener) for opener, _ in self._OPENERS) - 1
            safe_len = max(0, len(self._buffer) - keep)
            if safe_len:
                yield StreamEvent("content", content=self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
            break

    def finish(self) -> Iterator[StreamEvent]:
        if self._mode == "tool":
            self._tool_buffer += self._buffer
            self._emit_tool_calls()
        elif self._buffer:
            event_type = "reasoning" if self._mode == "reasoning" else "content"
            yield StreamEvent(event_type, content=self._buffer)
        self._buffer = ""
        self._tool_buffer = ""
        self._mode = "content"


class _LazyLocalBackend:
    def __init__(self, model_path: str, config: Any):
        self.model_path = model_path
        self.config = config
        self.model = os.path.basename(model_path.rstrip(os.sep)) or model_path
        self._load_lock = threading.RLock()
        self._generation_lock = threading.RLock()
        self._closed = False
        # 草稿模型（投机解码用）：由 UI 把已加载的辅助模型后端挂进来，见
        # set_draft_model()。默认 None → 退化为普通解码。
        self._draft_backend: Optional[Any] = None

    def generate_batch(
        self, message_lists, batch_size=5, temperature=0.7, max_tokens=1024
    ) -> List[str]:
        """批量生成的通用兜底：逐条走 ``chat_stream``，保证正确但非真并行。

        MLX 等支持真·批量的后端会重写本方法（一次性把多组消息 pad 成 ``[B, T]``
        做并行前向）。摘要任务调用前会在外层临时关闭思考模式，这里不再重复处理。
        """
        out: List[str] = []
        for msgs in message_lists:
            try:
                buf: List[str] = []
                for ev in self.chat_stream(msgs, tools=None, temperature=temperature):
                    if ev.type == "content":
                        buf.append(ev.content)
                out.append("".join(buf).strip())
            except Exception as e:  # pragma: no cover - 单条容错
                logger.warning("[batch] 单条生成失败: %s", e)
                out.append("")
        return out

    def list_models(self) -> List[str]:
        return [self.model]

    # ------------------------------------------------------------------
    def set_draft_model(self, backend: Optional[Any]) -> None:
        """挂上/摘掉投机解码用的草稿模型后端（通常是已加载的辅助模型）。

        草稿模型必须与主模型**同词表**（mlx_lm 的硬性要求）。这里只做保守校验：
        词表对不上就静默禁用投机解码，绝不影响正常生成。
        """
        self._draft_backend = backend

    def _draft_model_obj(self) -> Optional[Any]:
        """取出可用的草稿模型对象；不可用返回 None（退化为普通解码）。"""
        if not bool(getattr(self.config, "llm_speculative", True)):
            return None
        backend = getattr(self, "_draft_backend", None)
        if backend is None or backend is self:
            return None
        if getattr(backend, "_closed", False):
            return None
        model = getattr(backend, "_model_obj", None)
        if model is None:
            return None
        # 同词表校验：主模型与草稿模型的 vocab_size 必须一致，否则投机解码的
        # 拒绝采样无从比较 logits。
        try:
            main_vocab = getattr(self._model_obj, "vocab_size", None)
            draft_vocab = getattr(model, "vocab_size", None)
            if main_vocab and draft_vocab and int(main_vocab) != int(draft_vocab):
                logger.debug("投机解码禁用：草稿模型词表 %s != 主模型 %s",
                             draft_vocab, main_vocab)
                return None
        except Exception:  # pragma: no cover - 校验失败按禁用处理
            return None
        # 草稿后端正在被别处占用（例如 RAG 浓缩）时不抢：两边同时用一个模型
        # 会在 MLX 的线程局部 stream 上打架。
        lock = getattr(backend, "_generation_lock", None)
        if lock is not None and not lock.acquire(blocking=False):
            return None
        if lock is not None:
            lock.release()
        return model

    def _check_open(self) -> None:
        if self._closed:
            raise LLMError("本地模型后端已关闭")

    def clone(self):
        # Local runtimes keep multi-GB weights. Sharing one synchronized backend
        # is intentional; constructing an extraction-only copy would double RAM.
        return self


def _mlx_ram_bytes() -> int:
    """返回物理内存字节数；取不到时兜底 16GB。"""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 16 * 1024 ** 3


def apply_mlx_memory_policy() -> None:
    """统一 MLX 内存策略（可重复调用，每次按当时系统状态重新评估）。

    聊天后端与 RAG 各 MLX 入口（嵌入/重排/OCR）共用。2026-10-26 用户决策：

    memory_limit = **60% × 系统当前可用内存**（psutil available：总内存 −
        已用 − 皆可回收的缓存）。以「此刻还能安全吃多少」为准而非总内存
        固定比例——其它应用占用越多，MLX 能拿的越少。取不到 available
        （psutil 缺失）时回退总内存 60%。下限 1GB 保底（可用内存极低时
        至少让小模型能跑，宁可报错也不卡死系统）。
    cache_limit = **一律 1GB**。变长批次下每个新 padding 形状都缓存一套
        buffer，6GB 封顶实测仍会被嵌入期形状多样性吃满（#140 mem.log：
        每子批 +0.9GB 冲到 11.12GB）；1GB 足够跨层复用，代价是长上下文
        prefill 的 buffer 重分配开销（聊天场景可感知的抖动换取系统不卡死）。

    注意：memory_limit 直接绝对设置（不做 get+min「不抬高」逻辑——重复
    调用按当时可用内存刷新，同步卸载 LLM 后限制随之放宽，正是期望行为）。
    cache_limit 必须无条件设置：本机 MLX 版本没有 get_cache_limit（只有
    set），任何先 get 再 min 的写法都会抛 AttributeError。
    """
    try:
        import mlx.core as mx

        try:
            import psutil

            avail = psutil.virtual_memory().available
            mem_cap = int(avail * 0.60)
        except Exception:
            mem_cap = int(_mlx_ram_bytes() * 0.60)
        mem_cap = max(mem_cap, 1024 ** 3)  # 保底 1GB

        try:
            mem = min(int(mx.get_memory_limit()), mem_cap)
        except Exception:
            mem = mem_cap
        mx.set_memory_limit(mem)

        # cache 1GB 硬限：必须无条件设置——本机 MLX 没有 get_cache_limit
        # （只有 set），先 get 再 min 的写法会抛 AttributeError 且被吞掉，
        # 等于从未生效（#140：嵌入期 cache 无界冲到 11.12GB）。
        mx.set_cache_limit(1024 ** 3)
    except Exception as e:  # pragma: no cover - 失败不影响加载
        logger.debug("设置 MLX 内存策略失败（忽略）: %s", e)


def _mlx_batched_generate(
    model_obj,
    tokenizer,
    config,
    message_lists,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    prompt_fn,
) -> List[str]:
    """mlx_lm 真·批量生成引擎（主模型 / 辅助模型共用）。

    把多组消息按 ``batch_size`` 分组、pad 成 ``[B, T]`` 一次前向并行 prefill +
    自回归解码，显著快于逐条生成。``prompt_fn(messages)`` 负责应用 chat 模板
    （思考 / 工具等由各自后端决定），引擎本身只管并行生成与内存安全。

    内存安全三道闸：① 分块 prefill（按 2048 与 logits 预算自适应步长），避免
    长序列一次性 ``[B,T]`` 前向让全注意力层算出 ``[B,T,T]`` 级中间张量 OOM；
    ② ``B×T`` 预算 + 组内长度比约束，长分片自动退化为小批、同组行长度相近以压低
    右填充占比（填充 KV 会污染短行解码注意力）；③ 每块 / 每 256 步 ``clear_cache``
    释放中间张量，防显存累积泄漏。
    """
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_lm.models.cache import make_prompt_cache

    sampler = make_sampler(
        temp=max(0.0, float(temperature)),
        top_p=float(getattr(config, "top_p", 0.95)),
        min_p=float(getattr(config, "min_p", 0.0)),
        top_k=int(getattr(config, "top_k", 0)),
    )
    logits_processors = make_logits_processors(
        repetition_penalty=float(getattr(config, "repetition_penalty", 1.0)),
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_id = eos_id[0] if eos_id else None
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    n = len(message_lists)
    out_tokens: List[List[int]] = [[] for _ in range(n)]
    done = [False] * n
    if n == 0:
        return []
    # 内存预算（见模块 docstring / 上方函数说明）。
    _BATCH_TOKEN_BUDGET = 12_000
    _PREFILL_STEP = 2048  # 与 stream_generate 的 prefill_step_size 一致
    _LOGIT_BUDGET = 2_000_000_000
    _vocab_size = 0
    for _probe in (
        getattr(getattr(model_obj, "model", None), "embed_tokens", None),
        getattr(model_obj, "lm_head", None),
    ):
        _sh = getattr(getattr(_probe, "weight", _probe), "shape", None)
        if _sh:
            _vocab_size = int(_sh[0])
            break
    if _vocab_size <= 0:
        _vocab_size = 248_320
    import time as _t

    _t0 = _t.monotonic()  # GEN_BATCH_DEBUG 计时基准
    # 先全部模板化+编码，按长度降序排列：长序列先处理（受预算限制只能小批甚至
    # 单条），短序列随后可满额并行，同时减少批内填充浪费。
    prompts = [prompt_fn(m) for m in message_lists]
    encs = [list(tokenizer.encode(p)) for p in prompts]
    order = sorted(range(n), key=lambda i: -len(encs[i]))
    pos_i = 0
    while pos_i < n:
        t_max = max(1, len(encs[order[pos_i]]))
        b = min(batch_size, max(1, _BATCH_TOKEN_BUDGET // t_max), n - pos_i)
        # 组内长度比约束：只与首行长度相近（≥80%）的行同组，避免短行被长行垫高后
        # 大量右填充 KV 污染其解码注意力（实测产生回声式垃圾输出）。
        _min_len = (t_max * 4) // 5
        for _k in range(b):
            if len(encs[order[pos_i + _k]]) < _min_len:
                b = max(1, _k)
                break
        idxs = order[pos_i : pos_i + b]
        pos_i += b
        grp_encs = [encs[i] for i in idxs]
        g = len(grp_encs)
        max_len = max(len(e) for e in grp_encs)
        real_lens = [len(e) for e in grp_encs]
        ids = mx.array(
            [
                [(e[i] if i < len(e) else pad_id) for i in range(max_len)]
                for e in grp_encs
            ]
        )
        cache = make_prompt_cache(model_obj)
        step_size = max(
            128,
            min(_PREFILL_STEP, _LOGIT_BUDGET // max(1, g * _vocab_size * 2)),
        )
        row_last: List[Any] = [None] * g
        processed = 0
        while processed < max_len:
            s = min(step_size, max_len - processed)
            chunk = ids[:, processed : processed + s]
            logits = model_obj(chunk, cache=cache)
            mx.eval(logits)
            # 物化混合架构（qwen3_5 的 GatedDeltaNet 等）的递归缓存状态：其状态以
            # 副作用形式写入缓存对象，仅 mx.eval(logits) 不会持久化，会导致后续解码
            # 步复用旧状态、重复生成同一 token。
            mx.eval([c.state for c in cache])
            mx.clear_cache()  # 释放本块中间张量，防止显存随步累积泄漏
            if logits.ndim == 2:  # 个别返回展平形状，规整回 [B, S, V]
                logits = logits.reshape(g, s, -1)
            for j in range(g):
                rl = real_lens[j]
                if rl == 0:
                    row_last[j] = logits[j, 0]  # 占位，空编码行不参与解码
                elif processed < rl <= processed + s:
                    row_last[j] = logits[j, rl - 1 - processed]
            processed += s
        last_logits = mx.stack(row_last)  # [g, V]
        if os.environ.get("GEN_BATCH_DEBUG"):
            logger.warning(
                "[GB] 组=%s B=%d T=%d step_size=%d prefill完成 t=%.1fs",
                idxs, g, max_len, step_size, _t.monotonic() - _t0,
            )
        step = 0
        active = [not done[gi] for gi in idxs]
        while step < max_tokens and any(active):
            next_tok = []
            for j in range(g):
                if active[j]:
                    # 保持 [1,V] 二维：repetition 处理器内部用 logits[:, tokens]。
                    # 注意 mlx_lm 处理器签名是 (tokens, logits)，顺序别反。
                    lg = last_logits[j:j + 1]
                    toks = out_tokens[idxs[j]]
                    for proc in logits_processors:
                        lg = proc(toks, lg)
                    raw = sampler(lg[0])
                    if isinstance(raw, mx.array):
                        raw = raw.item() if raw.ndim == 0 else int(raw.reshape(-1)[0])
                    next_tok.append(int(raw))
                else:
                    next_tok.append(pad_id)
            nxt = mx.array(next_tok).reshape(g, 1)
            logits = model_obj(nxt, cache=cache)
            mx.eval(logits)
            mx.eval([c.state for c in cache])
            # 每 256 步清理一次中间张量（与 mlx_lm generate_step 一致）：每步都清
            # 会让 Metal 分配器逐缓冲重申请，实测解码慢一个量级；完全不清则多次
            # 调用后显存累积泄漏（实测 86s+OOM）。
            if step % 256 == 0:
                mx.clear_cache()
            last_logits = logits[:, 0, :]  # [g, V]
            step += 1
            if os.environ.get("GEN_BATCH_DEBUG") and step % 100 == 0:
                logger.warning(
                    "[GB] 组=%s 解码 step=%d/%d t=%.1fs",
                    idxs, step, max_tokens, _t.monotonic() - _t0,
                )
            for j in range(g):
                gi = idxs[j]
                if not active[j]:
                    continue
                nt = next_tok[j]
                if eos_id is not None and nt == eos_id:
                    done[gi] = True
                    active[j] = False
                else:
                    out_tokens[gi].append(nt)
    results = []
    for toks in out_tokens:
        try:
            results.append(tokenizer.decode(toks).strip())
        except Exception:
            results.append("")
    return results


class MLXLanguageBackend(_LazyLocalBackend):
    """Direct MLX-LM text model backend."""

    def __init__(self, model_path: str, config: Any):
        super().__init__(resolve_model_path(model_path), config)
        self._model_obj = None
        self._tokenizer = None
        # 跨轮复用的 KV 缓存与它对应的 token 序列（见 chat_stream 的前缀复用）。
        # 混合架构（线性层 ArraysCache）不可裁剪，所以只在「新 prompt 完整包含
        # 旧序列」时复用，否则整体重建——最坏退化为无缓存，不会出错。
        self._kv_cache = None
        self._kv_ids: List[int] = []
        # 静态头缓存（system + 工具 schema）的**母本**与它的 token 序列：只读，
        # 每轮生成都用它的克隆体（见 _clone_prompt_cache）。
        self._head_cache = None
        self._head_ids: List[int] = []
        # 创建该缓存的 **OS 线程**标识（threading.get_ident()）。
        # **MLX 的 prompt cache 与其 stream 是线程局部的**：缓存（及 stream）
        # 在 A 线程创建、却在 B 线程用于生成时，MLX 在 B 线程的 stream 表里
        # 找不到那个 stream，直接抛
        #   RuntimeError: There is no Stream(gpu, N) in current thread
        # （实测抛出点：mlx_lm/generate.py 的 mx.eval([c.state for c in
        # prompt_cache])）。表现为「第一条消息正常、第二条必崩」。
        #
        # 用 get_ident() 而不是「每条消息新建的 QThread」：UI 已改为**常驻单条
        # agent 线程**（main_window._ensure_agent_worker），OS 线程跨消息稳定，
        # 缓存可跨轮命中。get_ident() 在线程**终止后会被回收复用**，所以新建
        # worker 时 UI 会主动 invalidate_kv_cache()（见 _ensure_agent_worker），
        # 双保险堵住「旧线程已死、新线程拿到同一 ident」的极端情形。
        self._kv_thread_ident = None
        # 上一轮的模型原始输出（含工具调用 XML）。agent 把它存进历史的
        # raw_content，本地后端据此重建忠实的历史序列以命中前缀复用。
        # HTTP 后端不提供该属性，agent 会跳过——历史格式对 HTTP 保持不变。
        self.last_raw_output = ""

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._model_obj is not None:
            return
        with self._load_lock:
            if self._model_obj is not None:
                return
            try:
                from mlx_lm import load
            except ImportError as e:
                raise LLMError("MLX-LM 未安装，请安装: pip install mlx-lm") from e
            logger.info("正在从本地加载 MLX-LM: %s", self.model_path)
            # 统一 MLX 内存策略：总内存封顶 ~65% RAM（防 OS 换页卡死、给系统留
            # 余量），buffer cache 封顶 ~6GB（防 prefill 长上下文时大注意力矩阵
            # 每步反复释放重分配造成的生成抖动）。详见 apply_mlx_memory_policy。
            apply_mlx_memory_policy()
            self._model_obj, self._tokenizer = load(
                self.model_path,
                lazy=False,
                trust_remote_code=False,
                # qwen3.5 系列 tokenizer 自带 Mistral 式预切分正则缺陷
                # （transformers 5.x 加载时告警 "incorrect regex pattern"
                # 并建议传 fix_mistral_regex=True）。该标志只在检测到该
                # 正则时生效，其它模型不受影响——始终传，避免分词偏差。
                tokenizer_config={"fix_mistral_regex": True},
            )
            logger.info("MLX-LM 加载完成: %s", self.model_path)

    @staticmethod
    def _common_prefix_len(a: List[int], b: List[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _prompt(
        self,
        messages: List[Dict[str, Any]],
        tools,
        add_generation_prompt: bool = True,
    ) -> Any:
        # enable_thinking 由配置面板的「思考模式」开关控制
        # （config.llm_enable_thinking；辅助模型经 _AuxConfigView 映射到
        # aux_llm_enable_thinking）：
        #   True  → 模板在 generation prompt 末尾预置 <think>，模型先输出思考
        #           正文再作答（思考过程进「思考过程」面板）；
        #   False → 不预置开标签，模型直接作答，省掉整段 reasoning，更快。
        #
        # 不能缺省：qwen3.x 系模板里
        #   {%- if enable_thinking is defined and enable_thinking is false %}
        # 缺省时走 else 分支，照样输出 "<think>\n" 开头——即「未设置」会被当成
        # 开思考，开关就失效了。所以必须显式传值。
        # 老 tokenizer 不认识该参数时 TypeError 会被下面的兜底吃掉。
        #
        # 未配置（None）必须当成 **True**：bool(None) 是 False，会把「没设置过」
        # 当成「用户关掉了思考」，于是模板在渲染历史时剥掉 `<think>…</think>`，
        # 而生成时提示是带着 `<think>` 的——两边序列从 assistant 块起就错位，
        # KV 前缀缓存整份作废（实测：公共前缀掉到上轮 prompt 的最后一个 token）。
        thinking = _resolve_enable_thinking(self.config)
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": thinking,
        }
        if tools:
            kwargs["tools"] = tools
        # arguments 还原成 dict：模板按 mapping 渲染历史 tool_calls 的参数。
        # think_opened：生成提示是否预置了 <think>（思考模式开关），决定历史里
        # 的 raw_content 该怎么补闭合标签才能与生成序列对齐。
        messages = _template_messages(messages, think_opened=thinking)
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # 模板不接受 tools / enable_thinking：逐个退掉再试，避免一次
            # TypeError 把两个参数一起丢掉（工具定义一丢，模型就再也
            # 不会调 task_plan，任务计划面板永远空着）。
            for extra in ("enable_thinking", "tools"):
                if extra in kwargs:
                    pruned = dict(kwargs)
                    pruned.pop(extra)
                    try:
                        return self._tokenizer.apply_chat_template(messages, **pruned)
                    except TypeError:
                        continue
            kwargs.pop("tools", None)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _usable_head_cache(self):
        """返回可用的静态头缓存母本 (cache, ids)；不可用返回 (None, [])。

        线程亲和性：母本与其 stream 同样是线程局部的，换线程复用会让 MLX 抛
        ``No Stream in current thread``，所以必须校验建缓存的线程。
        """
        head = getattr(self, "_head_cache", None)
        head_ids = getattr(self, "_head_ids", None)
        if head is None or not head_ids:
            return None, []
        if getattr(self, "_head_thread_ident", None) != threading.get_ident():
            return None, []
        return head, head_ids

    def prime_prefix_cache(self, system_prompt: str, tools=None) -> None:
        """只 **prefill**（不生成）「system + tools」这段公共前缀，填好 KV 缓存。

        为什么这样做：真实生成会往缓存里追加输出 token，而混合架构
        （ArraysCache）不可裁剪，一旦缓存里多了几个 token，下一条 prompt 就
        不再是它的前缀 → 缓存整体作废 → 首条真实输入仍要全量 prefill（~20s）。
        这里只跑前向、不采样，缓存恰好等于「system+tools」前缀；首条真实 prompt
        以它开头，可直接命中复用（只 prefill 新增的 user 轮 token）。

        必须在**常驻对话线程**上调用（KV 缓存与 stream 是线程局部的），
        由 AgentWorker 的 warmup 任务保证。
        """
        self._ensure_loaded()
        with self._generation_lock:
            try:
                prompt = self._prompt(
                    [{"role": "system", "content": system_prompt}],
                    tools,
                    add_generation_prompt=False,
                )
                encode = getattr(self._tokenizer, "encode", None)
                if not callable(encode):
                    return
                ids = encode(prompt)
                if not isinstance(ids, list):
                    ids = list(ids)
                if not ids:
                    return
                import mlx.core as mx
                from mlx_lm.models.cache import make_prompt_cache

                logger.info("MLX 预热开始: 预填 %d token 的 system+tools 前缀", len(ids))
                cache = make_prompt_cache(self._model_obj)
                model = self._model_obj
                # 分块前向填 KV；与 stream_generate 的 prefill 同款（input[None]）。
                # 默认 4096 → 常见前缀（system+tools ≈ 2900 token）一次喂完，省掉
                # 第二次前向的 kernel 启动与同步开销。
                step = max(512, int(getattr(self.config, "local_llm_prompt_batch_size", 4096)))
                for i in range(0, len(ids), step):
                    chunk = ids[i:i + step]
                    model(mx.array([chunk]), cache=cache)
                    mx.eval([c.state for c in cache])
                self._kv_cache = cache
                self._kv_ids = list(ids)
                self._kv_thread_ident = threading.get_ident()
                # 留一份**母本**：它永不参与生成（生成只会用克隆体），长度永远停在
                # 「system + 工具 schema」。思考模式下历史重放对不齐时，用它把
                # 全量 prefill 降级为「只 prefill 头之后的对话部分」。
                self._head_cache = _clone_prompt_cache(cache)
                self._head_ids = list(ids)
                # 母本同样是线程局部的（同 _kv_thread_ident 的理由）。
                self._head_thread_ident = threading.get_ident()
                logger.info("MLX 预热: 已预填 %d token 的 system+tools 前缀", len(ids))
            except Exception as e:  # pragma: no cover - 预热失败不影响正常对话
                logger.error("MLX 预热失败（忽略）: %s", e, exc_info=True)

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self._ensure_loaded()
        with self._generation_lock:
            try:
                from mlx_lm import stream_generate
                from mlx_lm.sample_utils import make_logits_processors, make_sampler

                prompt = self._prompt(messages, tools)
                self.last_raw_output = ""
                sampler = make_sampler(
                    temp=max(0.0, float(temperature)),
                    top_p=float(getattr(self.config, "top_p", 0.95)),
                    min_p=float(getattr(self.config, "min_p", 0.0)),
                    top_k=int(getattr(self.config, "top_k", 0)),
                )
                logits_processors = make_logits_processors(
                    repetition_penalty=float(getattr(self.config, "repetition_penalty", 1.0)),
                )
                parser = _TaggedOutputParser()
                # prompt 已预置思考开标签（qwen3.x 模板）→ 生成内容直接是
                # 思考正文，只有闭标签。不预置模式的话思考整段漏进对话区。
                if _prompt_opens_thinking(prompt):
                    parser.begin_reasoning()

                # ---- KV 前缀复用（对应 Ollama 的前缀缓存）----
                # 每轮 prompt = 固定头部（system + 18 个工具 schema，实测约
                # 2432 token）+ 历史 + 新问题。旧实现每轮把整个 prompt 重新
                # prefill：M1 上约 160 tok/s，光头部就要 ~15s，用户感知就是
                # 「本地模型特别慢」/「每次都像重新加载」。Ollama 快很大程度上
                # 靠缓存这段不变的前缀。
                #
                # 约束：qwen3_5 是混合架构，线性层用 ArraysCache，
                # trim_prompt_cache 不支持（can_trim 返回 False）。所以不裁剪，
                # 只按「公共前缀」复用；对不齐就整体丢弃重建——最坏退化成无缓存，
                # 绝不会因缓存错位产生乱码。
                #
                # 要让前缀对得上，历史必须忠实于模型原始输出。agent 为此维护
                # 「双视图」：content 存解析后正文（HTTP 后端/记忆抽取用），
                # raw_content 存原始生成（含工具调用 XML，仅本地后端读）。
                # _template_messages 据此重建序列——实测与上轮生成逐 token 一致
                # （2600/2600）。历史被改写（上下文收缩、reset）时 agent 调
                # invalidate_kv_cache 主动失效。
                ids = None
                encode = getattr(self._tokenizer, "encode", None)
                if callable(encode):
                    try:
                        ids = encode(prompt)
                        if not isinstance(ids, list):
                            ids = list(ids)
                    except Exception as e:  # pragma: no cover - 防御
                        logger.debug("tokenizer.encode 失败，退化为无缓存: %s", e)
                        ids = None
                reuse = 0
                cache = None
                # 线程亲和性检查（见 _kv_thread_ident 的注释）：缓存不是本线程
                # 建的就整份作废，在本线程重建。跨线程复用会让 MLX 在**当前**
                # 线程的 stream 表里找不到建缓存那个线程的 stream：
                #   RuntimeError: There is no Stream(gpu, N) in current thread
                # 因此 AgentWorker 必须常驻同一 OS 线程（不能每条消息 start 一次）。
                if (self._kv_cache is not None
                        and self._kv_thread_ident != threading.get_ident()):
                    logger.info(
                        "MLX KV 缓存由另一线程创建（%s != %s），本轮作废重建",
                        self._kv_thread_ident, threading.get_ident())
                    self._kv_cache = None
                    self._kv_ids = []
                if ids is not None and self._kv_cache is not None and self._kv_ids:
                    # 两种命中形态，都要求「新 prompt 与缓存序列逐 token 一致」：
                    #   a) 缓存更长（常态）：缓存 = 上轮 prompt + 上轮生成，本轮
                    #      prompt 是它的前缀 → 只需喂本轮新增的尾部；
                    #   b) 新 prompt 更长：恰好完全包含缓存（重试/追加场景）。
                    # 两种情况都取公共前缀长度，未命中的部分交给 prefill。
                    common = self._common_prefix_len(ids, self._kv_ids)
                    if common == len(ids) or common == len(self._kv_ids):
                        cache, reuse = self._kv_cache, common
                    else:
                        # 前缀对不齐：缓存整体作废（不可裁剪，只能重建）。
                        # 诊断：打印公共前缀位置，配合「本轮比上轮短多少」判断是
                        # 历史被改写（common 远小于缓存）还是头部被改（common 很小）。
                        logger.info(
                            "MLX KV 前缀未命中: 公共前缀 %d（缓存 %d / 本轮 %d）→ 整份重建",
                            common, len(self._kv_ids), len(ids),
                        )
                        try:
                            dec = self._tokenizer.decode
                            lo = max(0, common - 6)
                            logger.info(
                                "MLX KV 分歧点: 缓存侧=%r || 本轮侧=%r",
                                dec(self._kv_ids[lo:common + 6]),
                                dec(ids[lo:common + 6]),
                            )
                        except Exception:  # pragma: no cover - 仅诊断
                            pass

                        self._kv_cache = None
                        self._kv_ids = []
                        # 降级：改用**静态头缓存的克隆体**。头（system + 工具
                        # schema）恒定且重放必然一致，所以即便历史里的思考段被
                        # 模板剥掉，头这一段照样对得上——本轮只需 prefill 头之后
                        # 的对话部分，而不是整个 prompt。
                        head, head_ids = self._usable_head_cache()
                        if head is not None and head_ids:
                            head_common = self._common_prefix_len(ids, head_ids)
                            if head_common == len(head_ids):
                                cloned = _clone_prompt_cache(head)
                                if cloned is not None:
                                    cache, reuse = cloned, head_common
                                    logger.info(
                                        "MLX KV 降级复用静态头: 跳过 %d token（本 prompt %d）",
                                        reuse, len(ids),
                                    )
                if cache is not None and reuse >= len(ids):
                    # 缓存已完整覆盖本轮 prompt → 没有可喂的新 token，
                    # ids[reuse:] 为空，mlx_lm 直接报
                    # "Either input_embeddings or prompt (or both) must be provided"。
                    # 这种情况（重发同一条、历史改写后 prompt 变短）不复用，
                    # 整轮重新 prefill——正确性优先，代价只是一次全量 prefill。
                    logger.info("MLX KV 缓存已覆盖全部 prompt（%d/%d），本轮不复用",
                                reuse, len(ids))
                    cache, reuse = None, 0
                if cache is None:
                    # 同样先试静态头缓存的克隆体（没有预热过、或缓存刚被作废）。
                    head, head_ids = self._usable_head_cache()
                    if (head is not None and head_ids and ids is not None
                            and self._common_prefix_len(ids, head_ids) == len(head_ids)):
                        cloned = _clone_prompt_cache(head)
                        if cloned is not None:
                            cache, reuse = cloned, len(head_ids)
                            logger.info(
                                "MLX KV 复用静态头: 跳过 %d token（本 prompt %d）",
                                reuse, len(ids),
                            )
                if cache is None:
                    try:
                        from mlx_lm.models.cache import make_prompt_cache

                        cache = make_prompt_cache(self._model_obj)
                    except Exception as e:  # pragma: no cover - 老版本兜底
                        logger.debug("make_prompt_cache 不可用，退化为无缓存: %s", e)
                        cache = None
                        reuse = 0
                if reuse:
                    logger.info(
                        "MLX KV 前缀复用: 跳过 %d token 的 prefill（本 prompt %d token）",
                        reuse, len(ids),
                    )
                gen_kwargs = {}
                if cache is not None:
                    gen_kwargs["prompt_cache"] = cache
                # prefill 分块：一次前向喂多少 token。默认 4096 让常见前缀
                # （system+tools ≈ 2900 token）一次喂完，省掉多余的同步开销。
                gen_kwargs["prefill_step_size"] = max(
                    512, int(getattr(self.config, "local_llm_prompt_batch_size", 4096))
                )
                # KV cache int8 量化（省显存）：MLX 默认不量化。整段量化
                # （quantized_kv_start=0）——agent 上下文常 <5000 token，用默认
                # 起点(5000)会完全不量化而达不到省内存目的。仅对支持量化的缓存层
                # 生效，不支持的架构由 mlx_lm 自动跳过，不会报错。
                # config.llama_cpp_kv_q8（辅助模型 aux_llama_cpp_kv_q8 经
                # _AuxConfigView 映射）可关（设置面板「KV量化」点选框）。
                if bool(getattr(self.config, "llama_cpp_kv_q8", True)):
                    gen_kwargs["kv_bits"] = 8
                    gen_kwargs["kv_group_size"] = 64
                    gen_kwargs["quantized_kv_start"] = 0
                # 投机解码：草稿模型 = 已加载的辅助模型（同词表，无需额外加载）。
                # 注意：mlx_lm 要求 prompt cache 可裁剪才允许投机解码，而 qwen3.5
                # 的 ArraysCache 不可裁剪 → 传 prompt_cache 时必须关掉，否则
                # ValueError。所以只有本轮没有缓存可复用（reuse == 0）才启用。
                draft = None if (cache is not None and reuse) else self._draft_model_obj()
                draft_lock = None
                if draft is not None:
                    draft_lock = getattr(self._draft_backend, "_generation_lock", None)
                    if draft_lock is not None:
                        draft_lock.acquire()
                    gen_kwargs["draft_model"] = draft
                    gen_kwargs["num_draft_tokens"] = max(
                        1, int(getattr(self.config, "llm_num_draft_tokens", 3))
                    )
                    logger.info("MLX 投机解码启用：草稿模型=%s，num_draft_tokens=%d",
                                getattr(self._draft_backend, "model", "?"),
                                gen_kwargs["num_draft_tokens"])

                prompt_tokens = 0
                completion_tokens = 0
                gen_ids: List[int] = []
                raw_text = ""
                try:
                    for response in stream_generate(
                        self._model_obj,
                        self._tokenizer,
                        ids[reuse:] if reuse else prompt,
                        max_tokens=int(getattr(self.config, "max_tokens", 32768)),
                        sampler=sampler,
                        logits_processors=logits_processors,
                        **gen_kwargs,
                    ):
                        tok_id = getattr(response, "token", None)
                        if tok_id is not None:
                            gen_ids.append(int(tok_id))
                        # 原始生成文本（含工具调用 XML）——agent 无法从事件重建：
                        # tool_call 段被 _TaggedOutputParser 吃掉转成结构化调用，
                        # 文本不进 content/reasoning 事件。这里逐片攒下，交给
                        # agent 存进历史的 raw_content，供下一轮前缀复用。
                        raw_text += getattr(response, "text", "") or ""
                        prompt_tokens = int(getattr(response, "prompt_tokens", prompt_tokens) or prompt_tokens)
                        completion_tokens = int(
                            getattr(response, "generation_tokens", completion_tokens + 1)
                            or completion_tokens + 1
                        )
                        for event in parser.feed(getattr(response, "text", "")):
                            yield event
                finally:
                    if draft_lock is not None:
                        draft_lock.release()
                yield from parser.finish()
                # 记录本轮完整序列，供下一轮前缀比对。被取消（生成中途 break）
                # 时 gen_ids 不完整，序列对不上，下一轮自动重建，不会错位。
                if cache is not None:
                    self._kv_cache = cache
                    # 记下建缓存的线程：下一轮只有同一线程才能复用（见上方检查）。
                    self._kv_thread_ident = threading.get_ident()
                    if ids is not None:
                        self._kv_ids = ids + gen_ids
                    # ids 为 None（tokenizer 无 encode）→ 不记录缓存序列，
                    # 下一轮退化为无缓存字符串路径。
                logger.info(
                    "MLX 本轮生成结束: gen_ids=%d raw_len=%d draft=%s",
                    len(gen_ids), len(raw_text), draft is not None,
                )
                self.last_raw_output = raw_text
                yield StreamEvent(
                    "done",
                    tool_calls=parser.tool_calls,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    finish_reason="tool_calls" if parser.tool_calls else "stop",
                )
            except LLMError:
                raise
            except Exception as e:
                raise LLMError(f"MLX-LM 推理失败: {type(e).__name__}: {e}") from e

    def generate_batch(
        self, message_lists, batch_size=5, temperature=0.7, max_tokens=1024
    ) -> List[str]:
        """真·批量生成：把多组消息按 ``batch_size`` 分组，每组 pad 成 ``[B, T]`` 一次
        前向、并行 prefill + 自回归解码，显著快于逐条 ``chat_stream``。

        底层引擎见模块级 ``_mlx_batched_generate``（主/辅后端共用）。本方法额外：
          * 调用期间临时关闭思考模式（摘要是单轮直答，思考只拖慢且浪费预算）；
          * 整组在同一把 ``_generation_lock`` 下完成，不与主对话的 ``chat_stream``
            并发，避免 MLX 权重/缓存的线程竞争；
          * 异常时降级 ``_generate_batch_sequential``（逐条 ``chat_stream``）。
        """
        self._ensure_loaded()
        cfg = getattr(self, "config", None)
        saved_think = getattr(cfg, "llm_enable_thinking", None) if cfg is not None else None
        if cfg is not None:
            try:
                cfg.llm_enable_thinking = False
            except Exception:
                saved_think = None
        try:
            with self._generation_lock:
                return _mlx_batched_generate(
                    self._model_obj,
                    self._tokenizer,
                    self.config,
                    message_lists,
                    batch_size,
                    temperature,
                    max_tokens,
                    self._prompt,
                )
        except LLMError:
            raise
        except Exception as e:
            logger.warning(
                "[MLX batch] 批量生成异常，降级逐条 chat_stream: %s", e, exc_info=True
            )
            return self._generate_batch_sequential(
                message_lists, temperature=temperature
            )
        finally:
            if cfg is not None and saved_think is not None:
                try:
                    cfg.llm_enable_thinking = saved_think
                except Exception:
                    pass

    def _generate_batch_sequential(self, message_lists, temperature=0.7) -> List[str]:
        """generate_batch 失败时的顺序兜底（复用 chat_stream）。"""
        out: List[str] = []
        for msgs in message_lists:
            try:
                buf: List[str] = []
                for ev in self.chat_stream(msgs, tools=None, temperature=temperature):
                    if ev.type == "content":
                        buf.append(ev.content)
                out.append("".join(buf).strip())
            except Exception as e:  # pragma: no cover - 单条容错
                logger.warning("[batch] 单条生成失败: %s", e)
                out.append("")
        return out

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self._model_obj = None
            self._tokenizer = None
            self._kv_cache = None
            self._kv_ids = []
            self._head_cache = None
            self._head_ids = []
            self._kv_thread_ident = None
            try:
                import gc
                import mlx.core as mx
                # 先 gc 断开权重引用，再 clear_cache 让 MLX 分配器把显存归还系统，
                # 否则 _model_obj 置空后底层 MLX 缓存池仍会占着 GPU 统一内存。
                gc.collect()
                mx.clear_cache()
            except Exception:
                pass

    def invalidate_kv_cache(self) -> None:
        """丢弃跨轮 KV 缓存（历史被改写、用户取消、切换会话等场景调用）。

        混合架构的 ArraysCache 不可裁剪，缓存只能整用或整弃；对不齐时
        chat_stream 会自动重建，这里只是让调用方在已知历史变更时提前释放。

        **静态头缓存保留**：它只含 system + 工具 schema，与历史无关，历史改写
        后依然可用（下一轮直接降级复用它，省掉头部的 prefill）。
        """
        self._kv_cache = None
        self._kv_ids = []
        self._kv_thread_ident = None


class MLXAuxBackend:
    """轻量辅助模型后端：原生支持批量生成，专为高频长文本摘要 / 归并等任务设计。

    相比主模型后端（``MLXLanguageBackend``）刻意砍掉：工具调用、思考模式
    （辅助只做单轮直答，思考纯属浪费）、KV 前缀复用、静态头缓存、投机解码、
    流式首包延迟等主对话专用逻辑——辅助模型要的是**快、稳、省、批量**，把大量
    不适合主模型做的长文本处理在闲时一次性并行跑完，避免交互被卡住。

    批量引擎复用模块级 ``_mlx_batched_generate``（与主模型同一套并行 prefill +
    解码、内存安全闸），只换一个始终「关思考」的极简 chat 模板渲染。
    """

    def __init__(self, model_path: str, config: Any):
        self.model_path = resolve_model_path(model_path)
        self.model = os.path.basename(self.model_path.rstrip(os.sep)) or self.model_path
        self.config = config
        self._model_obj = None
        self._tokenizer = None
        self._load_lock = threading.RLock()
        self._generation_lock = threading.RLock()  # 辅助推理串行化，与主模型锁独立
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise LLMError("辅助模型后端已关闭")

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._model_obj is not None:
            return
        with self._load_lock:
            if self._model_obj is not None:
                return
            try:
                from mlx_lm import load
            except ImportError as e:
                raise LLMError("MLX-LM 未安装，请安装: pip install mlx-lm") from e
            logger.info("正在从本地加载辅助 MLX-LM: %s", self.model_path)
            # 统一 MLX 内存策略（与主模型共用，避免两者同时占满显存）。
            apply_mlx_memory_policy()
            self._model_obj, self._tokenizer = load(
                self.model_path,
                lazy=False,
                trust_remote_code=False,
                tokenizer_config={"fix_mistral_regex": True},
            )
            logger.info("辅助 MLX-LM 加载完成: %s", self.model_path)

    def _aux_prompt(self, messages: List[Dict[str, Any]]) -> Any:
        """极简 chat 模板渲染：始终关思考、不挂工具、只加 generation prompt。

        辅助模型只跑摘要 / 归纳这类单轮直答，无需工具与思考链；剥离这些只渲染
        必要模板，既快又稳（开思考会多出一大段 reasoning 文本拖慢且对归纳无益）。
        """
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        messages = _template_messages(messages, think_opened=False)
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            pruned = dict(kwargs)
            pruned.pop("enable_thinking", None)
            return self._tokenizer.apply_chat_template(messages, **pruned)

    def generate_batch(
        self, message_lists, batch_size=5, temperature=0.7, max_tokens=1024
    ) -> List[str]:
        """并行批量生成（辅助模型主力路径）。

        用于闲时摘要 map-reduce 的分片并行：把多个正文分片一次性 pad 成 ``[B, T]``
        走并行前向，显著快于逐条。思考恒定关闭，整组在 ``_generation_lock`` 下完成。
        """
        self._ensure_loaded()
        try:
            with self._generation_lock:
                return _mlx_batched_generate(
                    self._model_obj,
                    self._tokenizer,
                    self.config,
                    message_lists,
                    batch_size,
                    temperature,
                    max_tokens,
                    self._aux_prompt,
                )
        except LLMError:
            raise
        except Exception as e:
            logger.warning(
                "[aux batch] 辅助批量生成异常，降级逐条 chat_stream: %s", e, exc_info=True
            )
            return self._aux_generate_sequential(message_lists, temperature=temperature)

    def _aux_generate_sequential(self, message_lists, temperature=0.7) -> List[str]:
        """generate_batch 失败时的顺序兜底（复用 chat_stream）。"""
        out: List[str] = []
        for msgs in message_lists:
            try:
                buf: List[str] = []
                for ev in self.chat_stream(msgs, tools=None, temperature=temperature):
                    if ev.type == "content":
                        buf.append(ev.content)
                out.append("".join(buf).strip())
            except Exception as e:  # pragma: no cover - 单条容错
                logger.warning("[aux batch] 单条生成失败: %s", e)
                out.append("")
        return out

    def chat_stream(
        self, messages, tools=None, temperature: float = 0.7
    ) -> Iterator[StreamEvent]:
        """单条非流式直答（辅助模型无需真正流式：摘要不在意首包延迟）。

        用 ``batch_size=1`` 的批量引擎一次跑完，对外仍以 ``StreamEvent`` 流式协议
        吐出内容 + done，兼容 ``book_summary._safe_chat`` 等调用方。
        """
        self._ensure_loaded()
        temp = (
            float(temperature)
            if temperature is not None
            else float(getattr(self.config, "temperature", 0.7))
        )
        max_tokens = int(getattr(self.config, "max_tokens", 2048) or 2048)
        try:
            with self._generation_lock:
                out = _mlx_batched_generate(
                    self._model_obj,
                    self._tokenizer,
                    self.config,
                    [messages],
                    batch_size=1,
                    temperature=temp,
                    max_tokens=max_tokens,
                    prompt_fn=self._aux_prompt,
                )
            text = out[0] if out else ""
            if text:
                yield StreamEvent("content", content=text)
        except LLMError:
            raise
        except Exception as e:
            logger.warning("[aux] chat_stream 生成失败: %s", e)
        yield StreamEvent("done")

    def generate(self, messages, temperature=0.7, max_tokens=1024) -> str:
        """单条一次性生成（便捷封装，等价于 batch_size=1）。"""
        out = self.generate_batch(
            [messages], batch_size=1, temperature=temperature, max_tokens=max_tokens
        )
        return out[0] if out else ""

    @property
    def context_window(self) -> int:
        """辅助模型有效上下文窗口（token）。优先模型 config，回落到 aux_max_context_tokens。"""
        cfg_max = int(getattr(self.config, "max_context_tokens", 0) or 0)
        try:
            mcfg = getattr(self._model_obj, "config", None)
            if isinstance(mcfg, dict):
                m = int(mcfg.get("max_position_embeddings", 0) or 0)
            else:
                m = int(getattr(mcfg, "max_position_embeddings", 0) or 0)
            if m >= 512:
                return m
        except Exception:
            pass
        return cfg_max if cfg_max >= 512 else 16384

    def list_models(self) -> List[str]:
        return [self.model]

    def close(self) -> None:
        self._closed = True
        self._model_obj = None
        self._tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass


class MLXVLMBackend(_LazyLocalBackend):
    """Direct MLX-VLM backend for text/image/audio/video chat models."""

    def __init__(self, model_path: str, config: Any):
        super().__init__(resolve_model_path(model_path), config)
        self._model_obj = None
        self._processor = None
        self._model_config: Dict[str, Any] = {}
        # 跨轮复用的 KV 缓存与它对应的 token 序列（见 chat_stream 的前缀复用）。
        # 混合架构（线性层 ArraysCache）不可裁剪，只在「新 prompt 完整包含
        # 旧序列」时复用，否则整体重建——最坏退化为无缓存，不会出错。
        self._kv_cache = None
        self._kv_ids: List[int] = []
        # 静态头缓存（system + 工具 schema）的**母本**与它的 token 序列：只读，
        # 每轮生成都用它的克隆体（见 _clone_prompt_cache）。
        self._head_cache = None
        self._head_ids: List[int] = []
        # 创建该缓存的 **OS 线程**标识（threading.get_ident()）。
        # MLX 的 prompt cache 与其 stream 是线程局部的：缓存（及 stream）
        # 在 A 线程创建、却在 B 线程用于生成时，MLX 在 B 线程的 stream 表里
        # 找不到建缓存那个线程的 stream（RuntimeError: No Stream in current
        # thread），所以必须校验线程标识。AgentWorker 常驻同一 OS 线程保证了
        # 缓存可跨轮命中；get_ident() 在线程终止后会被回收复用，故新建 worker
        # 时 UI 会主动 invalidate_kv_cache()（见 _ensure_agent_worker）。
        self._kv_thread_ident = None
        self._head_thread_ident = None
        # 上一轮的模型原始输出（含工具调用 XML）。agent 把它存进历史的
        # raw_content，本地后端据此重建忠实的历史序列以命中前缀复用。
        self.last_raw_output = ""

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._model_obj is not None:
            return
        with self._load_lock:
            if self._model_obj is not None:
                return
            try:
                from mlx_vlm import load
            except ImportError as e:
                raise LLMError("MLX-VLM 未安装，请安装: pip install mlx-vlm") from e
            config_path = Path(self.model_path) / "config.json"
            if config_path.exists():
                self._model_config = json.loads(config_path.read_text(encoding="utf-8"))
            logger.info("正在从本地加载 MLX-VLM: %s", self.model_path)
            # 同 MLXLLMBackend：统一 MLX 内存策略（总内存封顶 65% RAM、buffer
            # cache 封顶 ~6GB），避免长对话/prefill 长上下文抖动或内存吃满换页卡死。
            apply_mlx_memory_policy()
            self._model_obj, self._processor = load(self.model_path, lazy=False, strict=True)
            logger.info("MLX-VLM 加载完成: %s", self.model_path)

    @staticmethod
    def _media(messages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        media = {"image": [], "audio": [], "video": []}
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                for kind, value in re.findall(
                    r"\[(image|audio|video):([^\]]+)\]", content, flags=re.IGNORECASE
                ):
                    value = value.strip()
                    if value:
                        media[kind.lower()].append(value)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", ""))
                if part_type in ("image", "input_image", "image_url"):
                    value = part.get("image") or part.get("input_image") or part.get("image_url")
                    if isinstance(value, dict):
                        value = value.get("url")
                    if value:
                        media["image"].append(str(value))
                elif part_type in ("audio", "input_audio"):
                    value = part.get("audio") or part.get("input_audio")
                    if value:
                        media["audio"].append(str(value))
                elif part_type in ("video", "input_video"):
                    value = part.get("video") or part.get("input_video")
                    if value:
                        media["video"].append(str(value))
        return media

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self._ensure_loaded()
        with self._generation_lock:
            try:
                from mlx_lm.sample_utils import make_sampler
                from mlx_vlm import apply_chat_template, stream_generate

                media = self._media(messages)
                template_kwargs: Dict[str, Any] = {
                    "num_images": len(media["image"]),
                    "num_audios": len(media["audio"]),
                }
                if tools:
                    template_kwargs["tools"] = tools
                # 思考模式开关。mlx_vlm 的 apply_chat_template 是否接受
                # enable_thinking 取决于具体模型的模板，不能假定——先带上试，
                # 不认识（TypeError）就退掉重试，只牺牲这一个 kwarg。
                template_kwargs["enable_thinking"] = _resolve_enable_thinking(
                    self.config
                )
                try:
                    prompt = apply_chat_template(
                        self._processor,
                        self._model_config,
                        messages,
                        **template_kwargs,
                    )
                except TypeError:
                    template_kwargs.pop("enable_thinking", None)
                    prompt = apply_chat_template(
                        self._processor,
                        self._model_config,
                        messages,
                        **template_kwargs,
                    )
                sampler = make_sampler(
                    temp=max(0.0, float(temperature)),
                    top_p=float(getattr(self.config, "top_p", 0.95)),
                    min_p=float(getattr(self.config, "min_p", 0.0)),
                    top_k=int(getattr(self.config, "top_k", 0)),
                )
                parser = _TaggedOutputParser()
                # prompt 已预置思考开标签（qwen3.x 模板）→ 生成内容直接是
                # 思考正文，只有闭标签。不预置模式的话思考整段漏进对话区。
                if _prompt_opens_thinking(prompt):
                    parser.begin_reasoning()
                prompt_tokens = 0
                completion_tokens = 0
                gen_ids: List[int] = []
                raw_text = ""
                # ---- KV 前缀复用（对齐 MLXLanguageBackend）----
                # 每轮 prompt = 固定头部（system + 工具 schema）+ 历史 + 新问题。
                # 缓存只能整用或整弃（ArraysCache 不可裁剪），命中条件「新 prompt
                # 完整包含旧序列」→ 只 prefill 新增的尾部 token。
                cache = None
                reuse = 0
                ids = None
                encode = getattr(self._processor, "encode", None)
                if callable(encode):
                    try:
                        ids = encode(prompt)
                        if not isinstance(ids, list):
                            ids = list(ids)
                    except Exception:  # noqa: BLE001 - 退化为无缓存
                        ids = None
                try:
                    if (self._kv_cache is not None
                            and self._kv_thread_ident != threading.get_ident()):
                        # 缓存不是本线程建的就整份作废，在本线程重建。
                        self._kv_cache = None
                        self._kv_ids = []
                    if ids is not None and self._kv_cache is not None and self._kv_ids:
                        common = _common_prefix_len(ids, self._kv_ids)
                        if common == len(ids) or common == len(self._kv_ids):
                            cache, reuse = self._kv_cache, common
                        else:
                            # 前缀对不齐：缓存整体作废（不可裁剪，只能重建）。
                            logger.info(
                                "MLX-VLM KV 前缀未命中: 公共前缀 %d（缓存 %d / 本轮 %d）→ 整份重建",
                                common, len(self._kv_ids), len(ids),
                            )
                            self._kv_cache = None
                            self._kv_ids = []
                            # 降级：改用**静态头缓存的克隆体**。
                            head, head_ids = self._usable_head_cache()
                            if head is not None and head_ids:
                                head_common = _common_prefix_len(ids, head_ids)
                                if head_common == len(head_ids):
                                    cache, reuse = _clone_prompt_cache(head), head_common
                                    logger.info(
                                        "MLX-VLM KV 降级复用静态头: 跳过 %d token（本 prompt %d）",
                                        reuse, len(ids),
                                    )
                except Exception:  # noqa: BLE001 - 任何异常都退化为无缓存
                    cache, reuse = None, 0
                if cache is None:
                    # 没有预热过、或缓存刚被作废：试静态头缓存的克隆体。
                    head, head_ids = self._usable_head_cache()
                    if (head is not None and head_ids and ids is not None
                            and _common_prefix_len(ids, head_ids) == len(head_ids)):
                        cache, reuse = _clone_prompt_cache(head), len(head_ids)
                        logger.info(
                            "MLX-VLM KV 复用静态头: 跳过 %d token（本 prompt %d）",
                            reuse, len(ids),
                        )
                if cache is not None and reuse >= len(ids):
                    # 缓存已完整覆盖本轮 prompt → 没有可喂的新 token，不复用。
                    logger.info(
                        "MLX-VLM KV 缓存已覆盖全部 prompt（%d/%d），本轮不复用",
                        reuse, len(ids),
                    )
                    cache, reuse = None, 0
                if reuse:
                    logger.info(
                        "MLX-VLM KV 前缀复用: 跳过 %d token 的 prefill（本 prompt %d token）",
                        reuse, len(ids),
                    )
                kwargs = {
                    "max_tokens": int(getattr(self.config, "max_tokens", 32768)),
                    "sampler": sampler,
                    "verbose": False,
                    # KV cache int8 量化（省显存），同 MLX 文本后端：整段量化。
                    "kv_bits": 8,
                    "kv_group_size": 64,
                    "quantized_kv_start": 0,
                }
                if cache is not None:
                    kwargs["prompt_cache"] = cache
                # 分块 prefill：一次前向喂多少 token。
                kwargs["prefill_step_size"] = max(
                    512, int(getattr(self.config, "local_llm_prompt_batch_size", 4096))
                )
                try:
                    for response in stream_generate(
                        self._model_obj,
                        self._processor,
                        prompt,
                        image=media["image"] or None,
                        audio=media["audio"] or None,
                        video=media["video"] or None,
                        **kwargs,
                    ):
                        prompt_tokens = int(getattr(response, "prompt_tokens", prompt_tokens) or prompt_tokens)
                        completion_tokens = int(
                            getattr(response, "generation_tokens", completion_tokens + 1)
                            or completion_tokens + 1
                        )
                        tok_id = getattr(response, "token", None)
                        if tok_id is not None:
                            gen_ids.append(int(tok_id))
                        raw_text += getattr(response, "text", "") or ""
                        for event in parser.feed(getattr(response, "text", "")):
                            yield event
                except TypeError as e:
                    # 该 mlx_vlm 版本不接受 prompt_cache / prefill_step_size：
                    # 退化为无缓存全量 prefill，绝不因缓存错配炸掉生成。
                    if "prompt_cache" in str(e) or "prefill_step_size" in str(e):
                        logger.info(
                            "MLX-VLM: 本版本不接受 prompt_cache，退化为无缓存 prefill"
                        )
                        kwargs.pop("prompt_cache", None)
                        kwargs.pop("prefill_step_size", None)
                        self._kv_cache = None
                        self._kv_ids = []
                        for response in stream_generate(
                            self._model_obj,
                            self._processor,
                            prompt,
                            image=media["image"] or None,
                            audio=media["audio"] or None,
                            video=media["video"] or None,
                            **kwargs,
                        ):
                            prompt_tokens = int(getattr(response, "prompt_tokens", prompt_tokens) or prompt_tokens)
                            completion_tokens = int(
                                getattr(response, "generation_tokens", completion_tokens + 1)
                                or completion_tokens + 1
                            )
                            for event in parser.feed(getattr(response, "text", "")):
                                yield event
                    else:
                        raise
                yield from parser.finish()
                # 记录本轮完整序列，供下一轮前缀比对。被取消（生成中途 break）
                # 时 gen_ids 不完整，序列对不上，下一轮自动重建，不会错位。
                if cache is not None:
                    self._kv_cache = cache
                    self._kv_thread_ident = threading.get_ident()
                    if ids is not None:
                        self._kv_ids = ids + gen_ids
                self.last_raw_output = raw_text
                yield StreamEvent(
                    "done",
                    tool_calls=parser.tool_calls,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    finish_reason="tool_calls" if parser.tool_calls else "stop",
                )
            except Exception as e:
                raise LLMError(f"MLX-VLM 推理失败: {type(e).__name__}: {e}") from e

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self._model_obj = None
            self._processor = None
            try:
                import gc
                import mlx.core as mx

                # 与 MLXLLMBackend.close 保持一致：先断开权重引用再清分配器缓存，
                # 否则 _model_obj 置空后底层 MLX 缓存池仍占着 GPU 统一内存。
                gc.collect()
                mx.clear_cache()
            except Exception:
                pass

    def _usable_head_cache(self):
        """返回可用的静态头缓存母本 (cache, ids)；不可用返回 (None, [])。

        线程亲和性：母本与其 stream 同样是线程局部的，换线程复用会让 MLX 抛
        ``No Stream in current thread``，所以必须校验建缓存的线程。
        """
        head = getattr(self, "_head_cache", None)
        head_ids = getattr(self, "_head_ids", None)
        if head is None or not head_ids:
            return None, []
        if getattr(self, "_head_thread_ident", None) != threading.get_ident():
            return None, []
        return head, list(head_ids)

    def prime_prefix_cache(self, system_prompt: str, tools=None) -> None:
        """只 **prefill**（不生成）「system + tools」这段公共前缀，填好 KV 缓存。

        与 MLXLanguageBackend.prime_prefix_cache 同款语义：真实生成会往缓存里
        追加输出 token，而混合架构（ArraysCache）不可裁剪，一旦缓存里多了几个
        token，下一条 prompt 就不再是它的前缀 → 缓存整体作废 → 首条真实输入仍
        要全量 prefill（~20s）。这里只跑前向、不采样，缓存恰好等于「system+tools」
        前缀；首条真实 prompt 以它开头，可直接命中复用（只 prefill 新增的 user 轮
        token）。
        """
        self._ensure_loaded()
        with self._generation_lock:
            try:
                from mlx_lm.models.cache import make_prompt_cache
                from mlx_vlm import apply_chat_template

                template_kwargs: Dict[str, Any] = {
                    "enable_thinking": _resolve_enable_thinking(self.config),
                }
                if tools:
                    template_kwargs["tools"] = tools
                try:
                    prompt = apply_chat_template(
                        self._processor,
                        self._model_config,
                        [{"role": "system", "content": system_prompt}],
                        **template_kwargs,
                    )
                except TypeError:
                    template_kwargs.pop("enable_thinking", None)
                    prompt = apply_chat_template(
                        self._processor,
                        self._model_config,
                        [{"role": "system", "content": system_prompt}],
                        **template_kwargs,
                    )
                encode = getattr(self._processor, "encode", None)
                if not callable(encode):
                    return
                ids = encode(prompt)
                if not isinstance(ids, list):
                    ids = list(ids)
                if not ids:
                    return
                import mlx.core as mx

                cache = make_prompt_cache(self._model_obj)
                # 头部只有文本（无图/音/视频），纯文本前向即可，与真实 prompt 的
                # 文本前缀逐 token 对齐。分块前向填 KV，避免一次性喂超长序列。
                step = max(512, int(getattr(self.config, "local_llm_prompt_batch_size", 4096)))
                for i in range(0, len(ids), step):
                    chunk = ids[i:i + step]
                    self._model_obj(mx.array([chunk]), cache=cache)
                    mx.eval([c.state for c in cache])
                self._kv_cache = cache
                self._kv_ids = list(ids)
                self._kv_thread_ident = threading.get_ident()
                # 留一份**母本**：它永不参与生成（生成只会用克隆体），长度永远停在
                # 「system + 工具 schema」。思考模式下历史重放对不齐时，用它把
                # 全量 prefill 降级为「只 prefill 头之后的对话部分」。
                self._head_cache = _clone_prompt_cache(cache)
                self._head_ids = list(ids)
                self._head_thread_ident = threading.get_ident()
                logger.info(
                    "MLX-VLM 预热: 已预填 %d token 的 system+tools 前缀", len(ids)
                )
            except Exception as e:  # pragma: no cover - 预热失败不影响正常对话
                logger.warning("MLX-VLM 预热失败（忽略）: %s", e)

    def invalidate_kv_cache(self) -> None:
        """丢弃跨轮 KV 缓存（历史被改写、用户取消、切换会话等场景调用）。

        混合架构的 ArraysCache 不可裁剪，缓存只能整用或整弃；对不齐时
        chat_stream 会自动重建，这里只是让调用方在已知历史变更时提前释放。

        **静态头缓存保留**：它只含 system + 工具 schema，与历史无关，历史改写
        后依然可用（下一轮直接降级复用它，省掉头部的 prefill）。
        """
        self._kv_cache = None
        self._kv_ids = []
        self._kv_thread_ident = None


class ONNXLanguageBackend(_LazyLocalBackend):
    """Direct ONNX Runtime + Transformers + NumPy local generation backend.

    It intentionally recomputes the full sequence instead of requiring a
    model-specific KV-cache naming convention. This supports ordinary exported
    decoder graphs and multimodal graphs whose processor inputs match the ONNX
    input names. Unknown mandatory inputs produce an actionable error.
    """

    def __init__(self, model_path: str, config: Any):
        raw = resolve_model_path(model_path, expect_file=Path(model_path).suffix.lower() == ".onnx")
        super().__init__(raw, config)
        self._session = None
        self._tokenizer = None
        self._processor = None
        self._model_file = ""

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
                from transformers import AutoProcessor, AutoTokenizer
            except ImportError as e:
                raise LLMError(
                    "ONNX 本地后端需要 onnxruntime 与 transformers"
                ) from e
            path = Path(self.model_path)
            model_dir = path if path.is_dir() else path.parent
            if path.is_file():
                model_file = path
            else:
                candidates = sorted(path.glob("*.onnx")) + sorted(path.glob("onnx/*.onnx"))
                if not candidates:
                    raise LLMError(f"目录中未找到 ONNX 文件: {path}")
                model_file = candidates[0]
            # tokenizer 目录：优先用独立配置的 tokenizer 路径（ONNX 模型目录外
            # 单独的 tokenizer 导出），缺失时回退到模型目录（多数 ONNX 导出会把
            # tokenizer 一起放进模型目录）。
            tok_path = getattr(self.config, "llm_onnx_tokenizer_path", "") or str(model_dir)
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(getattr(self.config, "onnx_threads", 4)))
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_file),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._model_file = str(model_file)
            try:
                self._processor = AutoProcessor.from_pretrained(
                    tok_path, local_files_only=True, trust_remote_code=False
                )
            except Exception:
                self._processor = None
            self._tokenizer = AutoTokenizer.from_pretrained(
                tok_path, local_files_only=True, trust_remote_code=False
            )
            logger.info("ONNX 本地模型加载完成(CPU): %s", model_file)

    def _prompt(self, messages, tools):
        owner = self._processor or self._tokenizer
        # 思考模式开关（config.llm_enable_thinking，辅助模型经 _AuxConfigView
        # 映射）。与 MLX 后端一致：显式传值，不缺省——缺省会被模板当成开思考。
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": _resolve_enable_thinking(self.config),
        }
        if tools:
            kwargs["tools"] = tools
        messages = _template_messages(
            messages,
            think_opened=bool(kwargs.get("enable_thinking", True)),
        )
        # 老 tokenizer 可能不认识 enable_thinking / tools：逐个退掉再试，
        # 避免一次 TypeError 把两个参数一起丢掉。
        try:
            return owner.apply_chat_template(messages, **kwargs)
        except TypeError:
            for extra in ("enable_thinking", "tools"):
                if extra in kwargs:
                    pruned = dict(kwargs)
                    pruned.pop(extra)
                    try:
                        return owner.apply_chat_template(messages, **pruned)
                    except TypeError:
                        continue
            kwargs.pop("tools", None)
            kwargs.pop("enable_thinking", None)
            return owner.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _sample(logits, generated, temperature, top_p, min_p, top_k, repetition_penalty):
        import numpy as np

        values = np.asarray(logits, dtype=np.float64).copy()
        if repetition_penalty != 1.0:
            for token in set(generated[-64:]):
                values[token] = values[token] / repetition_penalty if values[token] > 0 else values[token] * repetition_penalty
        if temperature <= 0:
            return int(values.argmax())
        values /= max(temperature, 1e-5)
        if top_k > 0 and top_k < values.size:
            cutoff = np.partition(values, -top_k)[-top_k]
            values[values < cutoff] = -np.inf
        values -= np.nanmax(values)
        probs = np.exp(values)
        probs /= probs.sum()
        if min_p > 0:
            probs[probs < probs.max() * min_p] = 0
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[order])
            remove = cumulative > top_p
            if remove.size:
                remove[0] = False
            probs[order[remove]] = 0
        total = probs.sum()
        if not math.isfinite(float(total)) or total <= 0:
            return int(values.argmax())
        probs /= total
        return int(np.random.default_rng().choice(values.size, p=probs))

    def _initial_inputs(self, prompt: str, messages):
        import numpy as np

        media = MLXVLMBackend._media(messages)
        if self._processor is not None and any(media.values()):
            images = media["image"] or None
            if images:
                try:
                    from PIL import Image
                    images = [Image.open(x).convert("RGB") if os.path.isfile(x) else x for x in images]
                except Exception:
                    pass
            processor_kwargs: Dict[str, Any] = {
                "text": prompt,
                "return_tensors": "np",
            }
            if images:
                processor_kwargs["images"] = images
            if media["audio"]:
                processor_kwargs["audio"] = media["audio"]
            if media["video"]:
                processor_kwargs["videos"] = media["video"]
            batch = self._processor(**processor_kwargs)
        else:
            batch = self._tokenizer(prompt, return_tensors="np")
        return {k: np.asarray(v) for k, v in dict(batch).items()}

    def _feeds(self, encoded: Dict[str, Any], input_ids):
        import numpy as np

        input_names = {item.name: item for item in self._session.get_inputs()}
        feeds: Dict[str, Any] = {}
        for name, spec in input_names.items():
            if name == "input_ids":
                value = input_ids
            elif name == "attention_mask":
                value = np.ones_like(input_ids, dtype=np.int64)
            elif name == "position_ids":
                if name in encoded:
                    base = np.asarray(encoded[name], dtype=np.int64)
                    missing = input_ids.shape[1] - base.shape[-1]
                    if missing > 0:
                        tail = base[..., -1:] + np.arange(1, missing + 1, dtype=np.int64)
                        base = np.concatenate((base, tail), axis=-1)
                    value = base[..., :input_ids.shape[1]]
                else:
                    declared = list(spec.shape or [])
                    if len(declared) == 3:
                        raise LLMError(
                            "该 ONNX VLM 需要显式三维 M-RoPE position_ids；"
                            "processor 未生成此输入，不能用普通二维位置替代"
                        )
                    value = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
            elif name == "token_type_ids":
                value = np.zeros_like(input_ids, dtype=np.int64)
            elif name in encoded:
                value = encoded[name]
            else:
                raise LLMError(
                    f"ONNX 模型需要未能自动构造的输入 '{name}' ({spec.shape})；"
                    "请使用与该导出图匹配的专用 processor/后端"
                )
            feeds[name] = np.asarray(value)
        return feeds

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self._ensure_loaded()
        with self._generation_lock:
            try:
                import numpy as np

                prompt = self._prompt(messages, tools)
                encoded = self._initial_inputs(prompt, messages)
                if "input_ids" not in encoded:
                    raise LLMError("ONNX processor 未返回 input_ids")
                input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
                prompt_tokens = int(input_ids.size)
                generated: List[int] = []
                parser = _TaggedOutputParser()
                # prompt 已预置思考开标签（qwen3.x 模板）→ 生成内容直接是
                # 思考正文，只有闭标签。不预置模式的话思考整段漏进对话区。
                if _prompt_opens_thinking(prompt):
                    parser.begin_reasoning()
                eos = self._tokenizer.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
                # eos 缺失时无法识别终止符：退化为安全上限，避免跑满整段
                # max_tokens（默认 32768 步，每步都空转推理，纯无效开销）。
                _max_steps = int(getattr(self.config, "max_tokens", 32768))
                if not eos_ids:
                    _max_steps = min(_max_steps, 4096)
                    logger.warning("[onnx] 模型未提供 eos_token_id，生成上限降至 %d 步", _max_steps)
                _prev_len = 0
                for _ in range(_max_steps):
                    outputs = self._session.run(None, self._feeds(encoded, input_ids))
                    logits = np.asarray(outputs[0])
                    if logits.ndim < 3:
                        raise LLMError(
                            f"ONNX 输出形状 {logits.shape} 不是生成模型 logits；"
                            "分类/Embedding 模型不能作为主 LLM"
                        )
                    token = self._sample(
                        logits[0, -1], generated, float(temperature),
                        float(getattr(self.config, "top_p", 0.95)),
                        float(getattr(self.config, "min_p", 0.0)),
                        int(getattr(self.config, "top_k", 0)),
                        float(getattr(self.config, "repetition_penalty", 1.0)),
                    )
                    if token in eos_ids:
                        break
                    generated.append(token)
                    input_ids = np.concatenate(
                        (input_ids, np.asarray([[token]], dtype=np.int64)), axis=1
                    )
                    # 增量解码：每步只解最新一个 token 再累加，避免全量
                    # decode(generated) 的 O(n²) 总时间/内存。
                    delta = self._tokenizer.decode(
                        generated[_prev_len:], skip_special_tokens=False
                    )
                    _prev_len = len(generated)
                    yield from parser.feed(delta)
                yield from parser.finish()
                yield StreamEvent(
                    "done",
                    tool_calls=parser.tool_calls,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(generated),
                        "total_tokens": prompt_tokens + len(generated),
                    },
                    finish_reason="tool_calls" if parser.tool_calls else "stop",
                )
            except LLMError:
                raise
            except Exception as e:
                raise LLMError(f"ONNX 本地生成失败: {type(e).__name__}: {e}") from e

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self._session = None
            self._tokenizer = None
            self._processor = None


class LlamaCppBackend(_LazyLocalBackend):
    """Direct llama-cpp-python backend for GGUF LLM, VLM, and tool chat."""

    def __init__(self, model_path: str, config: Any):
        super().__init__(resolve_model_path(model_path, expect_file=True), config)
        self._llama = None
        # KV cache 预分配上限：见 config.local_llm_n_ctx。llama.cpp 加载时会按
        # n_ctx 一次性预占整段 KV cache，取 max_context_tokens（默认 32768）会让
        # 7B 模型预占 ~17GB，直接把内存吃爆（#158）。这里封到 local_llm_n_ctx，
        # 且不超过逻辑窗口 max_context_tokens。
        self._n_ctx = max(
            512,
            min(
                int(getattr(config, "max_context_tokens", 32768)),
                int(getattr(config, "local_llm_n_ctx", 16384)),
            ),
        )
        # 跨轮 KV / 前缀复用（对齐 MLX 后端的 _kv_cache 行为）：用 llama.cpp 的
        # LlamaCache 在不随时间变化的 system prompt + 对话前缀上复用 KV，每轮只
        # prefill 增量。默认开；内存吃紧可关或调小容量（见 config.llama_cpp_kv_cache）。
        self._kv_cache_enabled = bool(getattr(config, "llama_cpp_kv_cache", True))
        self._kv_cache_bytes = max(
            16 * 1024 * 1024,
            int(getattr(config, "llama_cpp_kv_cache_mb", 512)) * 1024 * 1024,
        )
        # KV cache 量化到 int8（config.llama_cpp_kv_q8，设置面板「KV量化」）：
        # 旧版硬编码 Q8_0，现改为可配置；关闭时保持 llama.cpp 默认 f16。
        self._kv_q8 = bool(getattr(config, "llama_cpp_kv_q8", True))
        self._kv_cache = None  # 当前挂到 Llama 上的 LlamaCache（失效时换新的）

    @property
    def context_window(self) -> int:
        """后端硬上下文上限（#158）：agent 收缩阈值据此计算，避免喂超 n_ctx。"""
        return self._n_ctx

    def _chat_handler(self):
        projection = getattr(self.config, "llm_vision_projection_path", "") or ""
        if not projection:
            return None
        projection = resolve_model_path(projection, expect_file=True)
        try:
            from llama_cpp import llama_chat_format as formats
        except ImportError as e:
            raise LLMError("llama-cpp-python 未安装") from e
        handler_name = str(getattr(self.config, "llm_vlm_handler", "mtmd")).lower()
        handlers = {
            "mtmd": formats.MTMDChatHandler,
            "llava15": formats.Llava15ChatHandler,
            "llava16": formats.Llava16ChatHandler,
            "qwen25vl": formats.Qwen25VLChatHandler,
            "minicpmv26": formats.MiniCPMv26ChatHandler,
            "moondream": formats.MoondreamChatHandler,
            "nanollava": formats.NanoLlavaChatHandler,
        }
        cls = handlers.get(handler_name)
        if cls is None:
            raise LLMError(f"不支持的 llama.cpp VLM handler: {handler_name}")
        kwargs = {"clip_model_path": projection, "verbose": False}
        if handler_name == "mtmd":
            kwargs["use_gpu"] = int(getattr(self.config, "local_gpu_layers", -1)) != 0
        return cls(**kwargs)

    def _ensure_loaded(self) -> None:
        self._check_open()
        if self._llama is not None:
            return
        with self._load_lock:
            if self._llama is not None:
                return
            try:
                import llama_cpp
                from llama_cpp import Llama
            except ImportError as e:
                raise LLMError(
                    "GGUF 后端需要 llama-cpp-python: pip install llama-cpp-python"
                ) from e
            self._llama = Llama(
                model_path=self.model_path,
                n_ctx=self._n_ctx,
                n_batch=max(1, int(getattr(self.config, "local_model_batch_size", 512))),
                n_threads=max(1, int(getattr(self.config, "local_model_threads", 4))),
                n_gpu_layers=int(getattr(self.config, "local_gpu_layers", -1)),
                chat_format=getattr(self.config, "llm_chat_format", "") or None,
                chat_handler=self._chat_handler(),
                verbose=False,
                # KV cache int8 量化（省显存）：默认 f16 → q8_0，KV 显存减半。
                # config.llama_cpp_kv_q8 可关（设置面板「KV量化」点选框）。
                type_k=(llama_cpp.GGML_TYPE_Q8_0 if self._kv_q8
                        else llama_cpp.GGML_TYPE_F16),
                type_v=(llama_cpp.GGML_TYPE_Q8_0 if self._kv_q8
                        else llama_cpp.GGML_TYPE_F16),
            )
            if self._kv_cache_enabled:
                self._attach_kv_cache()
            logger.info("GGUF 模型加载完成: %s", self.model_path)

    def _attach_kv_cache(self) -> None:
        """给 Llama 挂一个（全新空的）LlamaCache，启用跨轮 KV / 前缀复用。

        llama.cpp 在每次生成时自动做「最长 token 前缀命中 → load_state 复用 KV
        → 只 eval 后缀」，生成完再把整段序列状态存回缓存；不变的 system prompt
         ／对话前缀因此每轮只 prefill 一次。
        """
        if self._llama is None:
            return
        try:
            from llama_cpp import LlamaCache
        except ImportError as e:
            logger.warning("llama-cpp-python 无 LlamaCache，GGUF 前缀复用不可用: %s", e)
            self._kv_cache = None
            return
        cache = LlamaCache(capacity_bytes=self._kv_cache_bytes)
        self._llama.set_cache(cache)
        self._kv_cache = cache
        logger.info("GGUF 跨轮 KV 前缀复用已启用 (容量 %d MB)",
                    self._kv_cache_bytes // (1024 * 1024))

    def invalidate_kv_cache(self) -> None:
        """丢弃跨轮 KV / 前缀缓存（历史被改写、用户取消、切换会话时由 agent 调用）。

        对齐 MLXLanguageBackend.invalidate_kv_cache：换一个全新的空 LlamaCache，
        旧状态全部作废——下一轮无前缀可命中 → 全量 prefill（结果正确，只是没了
        加速），杜绝前缀错位命中到过期内容。刻意不在生成中调用 reset()，避免与
        流式生成争抢 llama.cpp 上下文。
        """
        if not self._kv_cache_enabled:
            return
        if self._llama is not None:
            try:
                self._attach_kv_cache()
            except Exception as e:  # noqa: BLE001 - 失效失败不应炸掉主流程
                logger.warning("GGUF KV 缓存失效失败: %s", e)
        else:
            self._kv_cache = None

    def prime_prefix_cache(self, system_prompt: str, tools=None) -> None:
        """只 prefill「system + tools」头部（不生成），让 llama.cpp 的 LlamaCache
        把这段不变前缀缓存下来，首条真实 prompt 直接命中复用（否则首条要全量
        prefill system + 工具 schema，约 20s）。

        与 MLX 后端同款语义：必须在常驻对话线程上调用（KV 缓存线程局部），由
        AgentWorker 的 warmup 任务保证。对关闭了 KV 缓存或模型未加载的后端是
        no-op。llama.cpp 在每次 ``create_chat_completion`` 时自动按 token 前缀命中
        已 attach 的 LlamaCache → 只 eval 增量，所以这里只需「把头部 eval 一遍」。
        """
        if not self._kv_cache_enabled or self._llama is None:
            return
        self._ensure_loaded()
        with self._generation_lock:
            try:
                # max_tokens=0：只把 system+tools 这段 prompt 前向一遍、不采样，
                # KV 写进已 attach 的 LlamaCache；首条真实 prompt 以它开头即可命中。
                # cache_prompt 默认 True，配合 set_cache 的 LlamaCache 做前缀匹配。
                self._llama.create_chat_completion(
                    messages=[{"role": "system", "content": system_prompt}],
                    tools=tools or None,
                    temperature=0.0,
                    max_tokens=0,
                    stream=False,
                )
                logger.info(
                    "GGUF 预热: 已预填 system+tools 前缀（LlamaCache 自动复用）"
                )
            except Exception as e:  # pragma: no cover - 预热失败不影响对话
                logger.warning("GGUF 预热失败（忽略）: %s", e)

    @staticmethod
    def _multimodal_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        pattern = re.compile(r"\[image:([^\]]+)\]", flags=re.IGNORECASE)
        for message in messages:
            copied = dict(message)
            content = copied.get("content")
            if isinstance(content, str):
                paths = [value.strip() for value in pattern.findall(content) if value.strip()]
                if paths:
                    text = pattern.sub("", content).strip()
                    parts: List[Dict[str, Any]] = [
                        {"type": "image_url", "image_url": {"url": path}}
                        for path in paths
                    ]
                    if text:
                        parts.append({"type": "text", "text": text})
                    copied["content"] = parts
            normalized.append(copied)
        return normalized

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self._ensure_loaded()
        with self._generation_lock:
            parser = _TaggedOutputParser()
            tool_acc: Dict[int, Dict[str, Any]] = {}
            usage: Dict[str, int] = {}
            finish_reason = ""
            try:
                stream = self._llama.create_chat_completion(
                    messages=self._multimodal_messages(messages),
                    tools=tools or None,
                    temperature=float(temperature),
                    top_p=float(getattr(self.config, "top_p", 0.95)),
                    top_k=int(getattr(self.config, "top_k", 20)),
                    min_p=float(getattr(self.config, "min_p", 0.05)),
                    repeat_penalty=float(getattr(self.config, "repetition_penalty", 1.0)),
                    max_tokens=int(getattr(self.config, "max_tokens", 32768)),
                    stream=True,
                )
                for obj in stream:
                    if obj.get("usage"):
                        usage = dict(obj["usage"])
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    yield from parser.feed(delta.get("content") or "")
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield StreamEvent("reasoning", content=reasoning)
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        slot = tool_acc.setdefault(
                            idx,
                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        function = tc.get("function") or {}
                        name = function.get("name") or ""
                        if name and name not in slot["function"]["name"]:
                            slot["function"]["name"] += name
                        slot["function"]["arguments"] += function.get("arguments") or ""
                yield from parser.finish()
                calls = [tool_acc[k] for k in sorted(tool_acc)] or parser.tool_calls
                for call in calls:
                    if not call.get("id"):
                        call["id"] = f"call_{uuid.uuid4().hex[:12]}"
                yield StreamEvent(
                    "done",
                    tool_calls=calls,
                    usage=usage,
                    finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
                )
            except Exception as e:
                raise LLMError(f"llama.cpp GGUF 推理失败: {type(e).__name__}: {e}") from e

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            llama = self._llama
            self._llama = None
            self._kv_cache = None
            if llama is not None:
                try:
                    llama.close()
                except Exception:
                    pass


def create_llm_backend(config: Any) -> LLMBackend:
    backend = str(getattr(config, "llm_backend", "openai") or "openai").lower()
    if backend == "openai":
        return LLMClient(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=config.request_timeout,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
        )
    path = getattr(config, "llm_model_path", "")
    if backend == "mlx_lm":
        return MLXLanguageBackend(path, config)
    if backend == "mlx_vlm":
        return MLXVLMBackend(path, config)
    if backend == "onnx":
        return ONNXLanguageBackend(path, config)
    if backend == "llama_cpp":
        return LlamaCppBackend(path, config)
    raise LLMError(f"未知主模型后端: {backend}; 可选值: {', '.join(LLM_BACKENDS)}")


class _AuxConfigView:
    """实时读取 ``aux_*`` 参数的配置视图（#140）。

    本地后端统一用 ``getattr(self.config, "top_p")`` 这套名字读采样参数。
    辅助模型要用自己的一套——主模型重生成质量、辅助模型重快与稳，温度与
    生成长度都不同——所以这里造一个视图，而不是去改每个后端类的读取代码
    （那样很容易在某个后端里漏改，悄悄读到主模型的值）。

    **必须是实时的，不能是快照**：用户在配置窗口改完辅助模型的 Temp /
    context 再点「应用配置」，快照里的值还是旧的——参数看起来没生效。
    这里每次属性访问都回到源 config 现读，改完即生效、无需重新加载模型。
    """

    # 标准字段名 → 辅助模型的源字段名。
    _SAMPLING_MAP = {
        "temperature": "aux_temperature",
        "max_tokens": "aux_max_tokens",
        "max_context_tokens": "aux_max_context_tokens",
        "top_p": "aux_top_p",
        "min_p": "aux_min_p",
        "top_k": "aux_top_k",
        "repetition_penalty": "aux_repetition_penalty",
        "request_timeout": "aux_request_timeout",
        # 思考模式：不登记的话辅助模型会经 __getattr__ 回落到主模型的
        # llm_enable_thinking，UI 上辅助模型的开关就成了摆设。
        "llm_enable_thinking": "aux_llm_enable_thinking",
        # KV cache 开关（2026-10-27）：不登记的话辅助 GGUF 的复用/量化开关
        # 会回落到主模型字段，辅助 MLX 读 llama_cpp_kv_q8 同理——UI 上辅助
        # 模型自己的「KV复用」「KV量化」就成了摆设。
        "llama_cpp_kv_cache": "aux_llama_cpp_kv_cache",
        "llama_cpp_kv_q8": "aux_llama_cpp_kv_q8",
        # GGUF VLM 的 mmproj / chat format / handler（见 LlamaCppBackend）。
        "llm_vision_projection_path": "aux_llm_vision_projection_path",
        "llm_onnx_tokenizer_path": "aux_llm_onnx_tokenizer_path",
        "llm_chat_format": "aux_llm_chat_format",
        "llm_vlm_handler": "aux_llm_vlm_handler",
    }

    def __init__(self, config: Any):
        object.__setattr__(self, "_src", config)

    def __getattr__(self, name: str):
        # 只有不在实例字典里的名字才会走到这里；_src 在字典里，不会递归。
        src = object.__getattribute__(self, "_src")
        mapped = self._SAMPLING_MAP.get(name)
        if mapped is not None:
            return getattr(src, mapped)
        return getattr(src, name)

    def __setattr__(self, name: str, value) -> None:
        # 后端偶尔会往 config 上挂东西（罕见，但存在）：转写到源 config，
        # 否则写进视图就丢了。
        src = object.__getattribute__(self, "_src")
        mapped = self._SAMPLING_MAP.get(name)
        setattr(src, mapped or name, value)


def aux_sampling_config(config: Any) -> Any:
    """返回把 ``aux_*`` 参数映射到标准字段名的实时配置视图（#140）。"""
    return _AuxConfigView(config)


def create_aux_llm_backend(config: Any) -> LLMBackend:
    """辅助模型后端（#140）：进程内直载，无网络端点。"""
    backend = str(getattr(config, "aux_llm_backend", "mlx_lm") or "mlx_lm").lower()
    if backend not in AUX_LLM_BACKENDS:
        raise LLMError(
            f"未知辅助模型后端: {backend}; 可选值: {', '.join(AUX_LLM_BACKENDS)}"
        )
    path = getattr(config, "aux_llm_model_path", "")
    view = aux_sampling_config(config)
    if backend == "mlx_lm":
        return MLXAuxBackend(path, view)
    if backend == "mlx_vlm":
        return MLXVLMBackend(path, view)
    if backend == "onnx":
        return ONNXLanguageBackend(path, view)
    return LlamaCppBackend(path, view)
