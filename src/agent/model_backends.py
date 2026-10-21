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
    """Resolve a configured local model path without performing network access."""
    raw = os.path.expandvars(os.path.expanduser((path or "").strip()))
    if not raw:
        raise LLMError("未配置本地模型路径")
    candidate = Path(raw)
    if not candidate.is_absolute():
        executable = Path(sys.executable).resolve()
        candidates = [
            Path.cwd() / candidate,
            project_root() / candidate,
            executable.parent / candidate,
        ]
        # macOS .app: <App>.app/Contents/MacOS/<exe>，允许 models 与 .app 同级。
        if len(executable.parents) >= 3:
            candidates.append(executable.parents[2] / candidate)
        candidate = next((item for item in candidates if item.exists()), candidates[1])
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


def _template_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 agent 历史消息适配成 chat template 友好的形状。两件事：

    1. **tool_calls 的 arguments 还原成 dict。** agent 内部（以及 OpenAI 兼容
       API）里它是 JSON 字符串，而 qwen3.5-agentic-coder 的模板按
       ``tool_call.arguments is mapping`` 渲染——字符串会让整个参数段被静默
       丢掉，模型下一轮看到的就是「调了 task_plan 但没有任何参数」。
    2. **有 raw_content 时用它重建忠实序列（KV 前缀复用的关键）。**
       本地后端会把模型原始输出（含工具调用 XML）记在 ``raw_content`` 上；
       直接渲染它，模板输出就与上一轮生成序列**逐 token 一致**，下一轮
       prefill 才能命中缓存。做法（已用真实 tokenizer 验证 2600/2600 对齐）：
         - ``content`` = 思考开标签 + 换行 + raw_content；
         - ``reasoning_content`` = ""（**字符串**）。
       后者是模板的暗门：它是字符串时模板跳过「按 think 闭标签自动拆分」，
       否则标签会被剥掉、序列就对不上了。
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
            copied["content"] = THO + "\n" + raw
            copied["reasoning_content"] = ""
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

    def list_models(self) -> List[str]:
        return [self.model]

    def _check_open(self) -> None:
        if self._closed:
            raise LLMError("本地模型后端已关闭")

    def clone(self):
        # Local runtimes keep multi-GB weights. Sharing one synchronized backend
        # is intentional; constructing an extraction-only copy would double RAM.
        return self


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
            # wired memory limit：mlx_lm CLI 默认会设，不设时 macOS 可能把
            # GPU 工作集换页，吞吐明显下降（实测 decode 与 prefill 都受影响）。
            # 失败不影响加载，只是少一层性能保障。
            try:
                from mlx_lm.utils import maybe_set_recommended_wired_limit

                limit = maybe_set_recommended_wired_limit()
                if limit:
                    logger.info("MLX wired memory limit 已设置: %.1f GB", limit / 1024 ** 3)
            except Exception as e:  # pragma: no cover - 性能优化，失败无所谓
                logger.debug("设置 wired memory limit 失败（忽略）: %s", e)
            # 限制 MLX Metal buffer cache 上限（默认≈95% RAM，即 16G 机器 ~15G）。
            # 不调的话，长对话生成时 buffer cache 一路涨到系统上限，物理内存被吃满后
            # 系统换页 → 内存冲到十几 G 并卡死（ovis_ocr.py 同款根因）。这里只封顶
            # 可复用的 scratch buffer（cache），不限制模型权重体积——避免大模型因
            # 内存上限被压到 8GB 而加载失败（RAG 的辅助模型才需要更严的全局上限）。
            try:
                import mlx.core as mx
                mx.set_cache_limit(1024 ** 3)  # 1GB
            except Exception as e:  # pragma: no cover - 失败不影响加载
                logger.debug("设置 MLX cache 上限失败（忽略）: %s", e)
            self._model_obj, self._tokenizer = load(
                self.model_path,
                lazy=False,
                trust_remote_code=False,
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

    def _prompt(self, messages: List[Dict[str, Any]], tools) -> Any:
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
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": bool(
                getattr(self.config, "llm_enable_thinking", True)
            ),
        }
        if tools:
            kwargs["tools"] = tools
        # arguments 还原成 dict：模板按 mapping 渲染历史 tool_calls 的参数。
        messages = _template_messages(messages)
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
                # 每条消息都是新的 AgentWorker 线程，所以不检查就是「第二轮必崩」。
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
                        self._kv_cache = None
                        self._kv_ids = []
                if cache is not None and reuse >= len(ids):
                    # 缓存已完整覆盖本轮 prompt → 没有可喂的新 token，
                    # ids[reuse:] 为空，mlx_lm 直接报
                    # "Either input_embeddings or prompt (or both) must be provided"。
                    # 这种情况（重发同一条、历史改写后 prompt 变短）不复用，
                    # 整轮重新 prefill——正确性优先，代价只是一次全量 prefill。
                    logger.debug("MLX KV 缓存已覆盖全部 prompt（%d/%d），本轮不复用",
                                 reuse, len(ids))
                    cache, reuse = None, 0
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

                prompt_tokens = 0
                completion_tokens = 0
                gen_ids: List[int] = []
                raw_text = ""
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

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self._model_obj = None
            self._tokenizer = None
            self._kv_cache = None
            self._kv_ids = []
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
        """
        self._kv_cache = None
        self._kv_ids = []
        self._kv_thread_ident = None


class MLXVLMBackend(_LazyLocalBackend):
    """Direct MLX-VLM backend for text/image/audio/video chat models."""

    def __init__(self, model_path: str, config: Any):
        super().__init__(resolve_model_path(model_path), config)
        self._model_obj = None
        self._processor = None
        self._model_config: Dict[str, Any] = {}

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
            # 同 MLXLLMBackend：封顶 MLX Metal buffer cache，避免长对话生成把
            # 物理内存吃满到系统上限后换页卡死。只限 cache，不限制模型权重体积。
            try:
                import mlx.core as mx
                mx.set_cache_limit(1024 ** 3)  # 1GB
            except Exception as e:  # pragma: no cover - 失败不影响加载
                logger.debug("设置 MLX cache 上限失败（忽略）: %s", e)
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
                template_kwargs["enable_thinking"] = bool(
                    getattr(self.config, "llm_enable_thinking", True)
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
                kwargs = {
                    "max_tokens": int(getattr(self.config, "max_tokens", 32768)),
                    "sampler": sampler,
                    "verbose": False,
                }
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
                yield from parser.finish()
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
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass


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
                    str(model_dir), local_files_only=True, trust_remote_code=False
                )
            except Exception:
                self._processor = None
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), local_files_only=True, trust_remote_code=False
            )
            logger.info("ONNX 本地模型加载完成(CPU): %s", model_file)

    def _prompt(self, messages, tools):
        owner = self._processor or self._tokenizer
        # 思考模式开关（config.llm_enable_thinking，辅助模型经 _AuxConfigView
        # 映射）。与 MLX 后端一致：显式传值，不缺省——缺省会被模板当成开思考。
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": bool(
                getattr(self.config, "llm_enable_thinking", True)
            ),
        }
        if tools:
            kwargs["tools"] = tools
        messages = _template_messages(messages)
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
            )
            logger.info("GGUF 模型加载完成: %s", self.model_path)

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
        # GGUF VLM 的 mmproj / chat format / handler（见 LlamaCppBackend）。
        "llm_vision_projection_path": "aux_llm_vision_projection_path",
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
        return MLXLanguageBackend(path, view)
    if backend == "mlx_vlm":
        return MLXVLMBackend(path, view)
    if backend == "onnx":
        return ONNXLanguageBackend(path, view)
    return LlamaCppBackend(path, view)
