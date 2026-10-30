"""Contract tests for selectable local language-model backends."""
from __future__ import annotations

import json
import os
import sys

import pytest

from agent.config import AgentConfig
from agent.llm_client import LLMClient
from agent.model_backends import (
    MLXLanguageBackend,
    MLXAuxBackend,
    _TaggedOutputParser,
    _coerce_param,
    _prompt_opens_thinking,
    _template_messages,
    create_llm_backend,
    discover_local_models,
    resolve_model_path,
)
from agent.rag.model_backends import (
    JinaV5NanoEmbeddingFunction,
    JinaV5NanoONNXEmbeddingFunction,
    create_embedding_backend,
)

# XML / 思考标签：用字符拼接构造，源码里不出现完整标签。
_LT, _GT = chr(60), chr(62)
_THO = _LT + "think" + _GT
_THE = _LT + "/think" + _GT
_TC = _LT + "tool_call" + _GT
_TCE = _LT + "/tool_call" + _GT
_FN = _LT + "function="
_FE = _LT + "/function" + _GT
_PA = _LT + "parameter="
_PE = _LT + "/parameter" + _GT


def _xml_tool_call(name, **params):
    """构造 qwen3.5-agentic-coder 格式的单个 XML 工具调用。"""
    body = _FN + name + ">\n"
    for k, v in params.items():
        body += _PA + k + ">\n" + str(v) + "\n" + _PE + "\n"
    return _TC + "\n" + body + _FE + "\n" + _TCE


def _feed(parser, text, step=3):
    """按固定步长喂流，模拟 token 级别的分片。"""
    events = []
    for i in range(0, len(text), step):
        events.extend(parser.feed(text[i : i + step]))
    events.extend(parser.finish())
    return events


def test_llamacpp_n_ctx_bounded_by_local_llm_n_ctx(monkeypatch):
    """#158：llama.cpp 的 n_ctx 必须封顶到 local_llm_n_ctx，否则加载时按默认
    max_context_tokens(32768) 预分配 KV cache，7B 模型约 17GB，直接把内存吃爆。"""
    from agent.model_backends import LlamaCppBackend

    monkeypatch.setattr(
        "agent.model_backends.resolve_model_path", lambda p, *a, **k: "/tmp/fake.gguf"
    )

    cfg = AgentConfig()
    cfg.llm_backend = "llama_cpp"
    cfg.llm_model_path = "/tmp/fake.gguf"
    cfg.max_context_tokens = 32768
    cfg.local_llm_n_ctx = 16384
    backend = LlamaCppBackend(cfg.llm_model_path, cfg)
    assert backend._n_ctx == 16384
    assert backend.context_window == 16384

    # local_llm_n_ctx 超过逻辑窗口时不反向放大（保持 ≤ max_context_tokens）。
    cfg.local_llm_n_ctx = 65536
    backend2 = LlamaCppBackend(cfg.llm_model_path, cfg)
    assert backend2._n_ctx == 32768

    # 极端小值有下限保护。
    cfg.local_llm_n_ctx = 1
    backend3 = LlamaCppBackend(cfg.llm_model_path, cfg)
    assert backend3._n_ctx >= 512


def _install_fake_llama_cpp(monkeypatch):
    """注入假 llama_cpp 模块（Llama + LlamaCache），返回 (Llama, LlamaCache)。

    Llama.created 记录每次构造出的实例；set_cache 写到的缓存可从实例 .cache 读。
    """
    pkg = type(sys)("llama_cpp")

    class LlamaCache:
        def __init__(self, capacity_bytes: int = 0):
            self.capacity_bytes = capacity_bytes

    class Llama:
        created: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.cache = None
            self.closed = False
            Llama.created.append(self)

        def set_cache(self, cache):
            self.cache = cache

        def reset(self):
            pass

        def close(self):
            self.closed = True

    pkg.Llama = Llama
    pkg.LlamaCache = LlamaCache
    # KV int8 量化（commit 1698078）在 Llama() 构造里引用该常量，假模块必须补齐。
    pkg.GGML_TYPE_Q8_0 = 8
    pkg.GGML_TYPE_F16 = 1  # 「KV量化」关闭时的回退类型
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    return Llama, LlamaCache


def test_llamacpp_kv_q8_on_by_default(monkeypatch):
    """KV量化默认开：type_k/type_v = Q8_0（int8，KV 显存减半）。"""
    _, llama, _ = _make_llamacpp(monkeypatch)
    assert llama.kwargs["type_k"] == 8 and llama.kwargs["type_v"] == 8


def test_llamacpp_kv_q8_configurable_off(monkeypatch):
    """「KV量化」点选框关闭 → KV 保持 f16（llama.cpp 默认精度）。"""
    _, llama, _ = _make_llamacpp(monkeypatch, llama_cpp_kv_q8=False)
    assert llama.kwargs["type_k"] == 1 and llama.kwargs["type_v"] == 1


def test_aux_config_view_maps_kv_fields():
    """辅助模型 KV 开关经 _AuxConfigView 映射，不回落主模型字段。"""
    from agent.model_backends import aux_sampling_config

    cfg = AgentConfig()
    cfg.llama_cpp_kv_cache = True
    cfg.aux_llama_cpp_kv_cache = False
    cfg.llama_cpp_kv_q8 = True
    cfg.aux_llama_cpp_kv_q8 = False
    view = aux_sampling_config(cfg)
    assert view.llama_cpp_kv_cache is False
    assert view.llama_cpp_kv_q8 is False


def _make_llamacpp(monkeypatch, **cfg_over):
    from agent.model_backends import LlamaCppBackend
    monkeypatch.setattr(
        "agent.model_backends.resolve_model_path", lambda p, *a, **k: "/tmp/fake.gguf"
    )
    Llama, LlamaCache = _install_fake_llama_cpp(monkeypatch)
    cfg = AgentConfig()
    cfg.llm_backend = "llama_cpp"
    cfg.llm_model_path = "/tmp/fake.gguf"
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    backend = LlamaCppBackend(cfg.llm_model_path, cfg)
    backend._ensure_loaded()
    return backend, Llama.created[-1], LlamaCache


def test_llamacpp_kv_cache_attached_on_load(monkeypatch):
    """默认开启时，GGUF 后端加载后应给 Llama 挂上 LlamaCache（跨轮前缀复用）。"""
    backend, llama, LlamaCache = _make_llamacpp(
        monkeypatch, llama_cpp_kv_cache=True, llama_cpp_kv_cache_mb=256)
    try:
        assert llama.cache is not None
        assert isinstance(llama.cache, LlamaCache)
        assert llama.cache.capacity_bytes == 256 * 1024 * 1024
        assert backend._kv_cache is llama.cache
    finally:
        backend.close()
    assert llama.closed


def test_llamacpp_kv_cache_disabled_attaches_nothing(monkeypatch):
    """关闭开关时不应挂任何缓存。"""
    backend, llama, _ = _make_llamacpp(monkeypatch, llama_cpp_kv_cache=False)
    try:
        assert llama.cache is None
        assert backend._kv_cache is None
    finally:
        backend.close()


def test_llamacpp_kv_cache_capacity_floor(monkeypatch):
    """容量有 16MB 下限，防止设成 0/负数导致缓存永远放不下一条状态。"""
    backend, llama, _ = _make_llamacpp(
        monkeypatch, llama_cpp_kv_cache=True, llama_cpp_kv_cache_mb=1)
    try:
        assert llama.cache.capacity_bytes >= 16 * 1024 * 1024
    finally:
        backend.close()


def test_llamacpp_kv_cache_invalidate_swaps_fresh(monkeypatch):
    """invalidate_kv_cache 应换成全新的空缓存，旧状态作废（避免前缀错位命中）。"""
    backend, llama, _ = _make_llamacpp(monkeypatch, llama_cpp_kv_cache=True)
    try:
        first = llama.cache
        assert first is not None
        backend.invalidate_kv_cache()
        assert llama.cache is not None and llama.cache is not first
        assert backend._kv_cache is llama.cache
    finally:
        backend.close()


def test_llamacpp_kv_cache_invalidate_noop_when_disabled(monkeypatch):
    """关闭时 invalidate 是空操作，不会凭空挂上缓存。"""
    backend, llama, _ = _make_llamacpp(monkeypatch, llama_cpp_kv_cache=False)
    try:
        backend.invalidate_kv_cache()
        assert llama.cache is None
        assert backend._kv_cache is None
    finally:
        backend.close()


def _make_fake_onnx_llm(inputs=None, **cfg_over):
    """构造注入了假 session/tokenizer 的 ONNXLanguageBackend，跑生成回路。

    本地没有 decoder ONNX 权重、无法真正加载；这里绕过 _ensure_loaded，直接注入
    假 InferenceSession（get_inputs/run）与假 tokenizer，验证 chat_stream 的
    prompt→feeds→run→sample→eos→增量 decode→流式→done 全链路与报错路径。
    ``inputs`` 为 ``[(name, shape), ...]``，用于模拟不同的 ONNX 图输入。
    """
    import threading

    import numpy as np

    from agent.model_backends import ONNXLanguageBackend

    class _FakeInput:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    class _FakeSession:
        # 当前序列长 L -> 下一 token：5->10,6->11,7->12,8->99(eos)
        _NEXT = {5: 10, 6: 11, 7: 12, 8: 99}

        def __init__(self, inputs):
            self._inputs = inputs

        def get_inputs(self):
            return self._inputs

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, outputs, feeds):
            ids = np.asarray(feeds["input_ids"])
            length = ids.shape[1]
            logits = np.zeros((1, length, 100), dtype=np.float32)
            logits[0, -1, self._NEXT.get(length, 99)] = 100.0
            return [logits]

    class _FakeTok:
        eos_token_id = 99

        def apply_chat_template(self, messages, **kw):
            return "PROMPT"

        def __call__(self, text, return_tensors="np", **kw):
            return {"input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
                    "attention_mask": np.ones((1, 5), dtype=np.int64)}

        def decode(self, ids, skip_special_tokens=False):
            return "".join({10: "A", 11: "B", 12: "C"}.get(int(i), "?") for i in ids)

    cfg = AgentConfig()
    cfg.temperature = 0.0
    cfg.max_tokens = 64
    for k, v in cfg_over.items():
        setattr(cfg, k, v)

    be = ONNXLanguageBackend.__new__(ONNXLanguageBackend)
    be.config = cfg
    be.model = "fake.onnx"
    be.model_path = "/tmp/fake.onnx"
    be._closed = False
    be._load_lock = threading.RLock()
    be._generation_lock = threading.RLock()
    be._session = _FakeSession([_FakeInput(n, s) for n, s in (
        inputs if inputs is not None
        else [("input_ids", [1, -1]), ("attention_mask", [1, -1])])])
    be._tokenizer = _FakeTok()
    be._processor = None
    be._model_file = "/tmp/fake.onnx"
    return be


def test_onnx_llm_chat_stream_generates_and_stops_on_eos():
    """ONNXLanguageBackend 生成回路：贪心解码出 ABC、命中 eos 停止、上报用量。"""
    be = _make_fake_onnx_llm()
    events = list(be.chat_stream(
        [{"role": "user", "content": "hi"}], tools=None, temperature=0.0))
    content = "".join(e.content for e in events if e.type == "content")
    done = [e for e in events if e.type == "done"]
    assert content == "ABC", repr(content)
    assert len(done) == 1
    assert done[0].usage["prompt_tokens"] == 5
    assert done[0].usage["completion_tokens"] == 3
    assert done[0].finish_reason == "stop"


def test_onnx_llm_feeds_unknown_input_raises_actionable_error():
    """ONNX 图含无法自动构造的必需输入时，应抛出点名该输入的可操作错误。"""
    from agent.llm_client import LLMError

    be = _make_fake_onnx_llm(inputs=[
        ("input_ids", [1, -1]), ("attention_mask", [1, -1]), ("weird_input", [1, -1])])
    with pytest.raises(LLMError) as ei:
        list(be.chat_stream([{"role": "user", "content": "hi"}], temperature=0.0))
    assert "weird_input" in str(ei.value)


def test_onnx_llm_rejects_non_generation_logits():
    """输出不是 [batch, seq, vocab] logits（如分类/Embedding 模型）应明确报错。"""
    import numpy as np

    from agent.llm_client import LLMError

    be = _make_fake_onnx_llm()

    class _FlatSession:
        def get_inputs(self):
            return [type("I", (), {"name": "input_ids", "shape": [1, -1]})(),
                    type("I", (), {"name": "attention_mask", "shape": [1, -1]})()]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, outputs, feeds):
            return [np.zeros((1, 100), dtype=np.float32)]  # 2D，非 logits

    be._session = _FlatSession()
    with pytest.raises(LLMError):
        list(be.chat_stream([{"role": "user", "content": "hi"}], temperature=0.0))


def test_apply_mlx_memory_policy_caps_cache_and_memory(monkeypatch):
    """统一 MLX 内存策略（2026-10-26 用户终值）：cache 一律 1GB，memory_limit =
    60% × 系统当前**可用**内存（下限 1GB）。cache_limit 必须无条件设置——
    本机 MLX 没有 get_cache_limit，任何先 get 再 min 的写法都会抛
    AttributeError（#140：cache 无界冲到 11.12GB 的根因）。同时验证幂等。"""
    from agent import model_backends as mb
    import sys, types

    class FakeMX:
        """仿本机 MLX：有 get_memory_limit，**没有** get_cache_limit。"""

        def __init__(self):
            self.memory_limit = 15 * 1024 ** 3   # 默认 95% RAM（16G 机器）
        def get_memory_limit(self):
            return self.memory_limit
        def set_memory_limit(self, v):
            self.memory_limit = v
        def set_cache_limit(self, v):
            self.cache_limit = v

    fake = FakeMX()
    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.get_memory_limit = fake.get_memory_limit
    mlx_core.set_memory_limit = fake.set_memory_limit
    mlx_core.set_cache_limit = fake.set_cache_limit
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setattr(mb, "_mlx_ram_bytes", lambda: 16 * 1024 ** 3)
    # psutil available = 10GB → mem_cap = 6GB（< 默认 15GB，取 min）
    import psutil
    monkeypatch.setattr(
        psutil, "virtual_memory",
        lambda: types.SimpleNamespace(available=10 * 1024 ** 3))

    mb.apply_mlx_memory_policy()
    # 总内存封顶到 60% × 可用 10GB = 6GB
    assert fake.memory_limit == 6 * 1024 ** 3
    # buffer cache 一律 1GB（无条件设置，无视 get_cache_limit 缺失）
    assert fake.cache_limit == 1 * 1024 ** 3
    # 幂等：再调用值不变、不报错
    mb.apply_mlx_memory_policy()
    assert fake.memory_limit == 6 * 1024 ** 3
    assert fake.cache_limit == 1 * 1024 ** 3


def test_apply_mlx_memory_policy_floor_and_fallback(monkeypatch):
    """psutil 不可用回退总内存 60%；可用内存极低时 1GB 保底。"""
    from agent import model_backends as mb
    import sys, types

    class FakeMX:
        def __init__(self):
            self.memory_limit = 15 * 1024 ** 3
        def get_memory_limit(self):
            return self.memory_limit
        def set_memory_limit(self, v):
            self.memory_limit = v
        def set_cache_limit(self, v):
            self.cache_limit = v

    fake = FakeMX()
    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.get_memory_limit = fake.get_memory_limit
    mlx_core.set_memory_limit = fake.set_memory_limit
    mlx_core.set_cache_limit = fake.set_cache_limit
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    # psutil 导入失败 → 回退 _mlx_ram_bytes（16G）× 60% = 9.6GB
    real_psutil = __import__("psutil")  # 先取真模块，稍后恢复
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(mb, "_mlx_ram_bytes", lambda: 16 * 1024 ** 3)
    mb.apply_mlx_memory_policy()
    assert fake.memory_limit == int(16 * 1024 ** 3 * 0.60)

    # 可用内存极低（100MB）→ cap 60MB 低于保底 → 取 1GB
    monkeypatch.setitem(sys.modules, "psutil", real_psutil)
    monkeypatch.setattr(
        real_psutil, "virtual_memory",
        lambda: types.SimpleNamespace(available=100 * 1024 ** 2))
    mb.apply_mlx_memory_policy()
    assert fake.memory_limit == 1 * 1024 ** 3


def test_tagged_parser_splits_reasoning_content_and_tool_call():
    parser = _TaggedOutputParser()
    events = []
    for chunk in [
        "hello <thi", "nk>reason", "ing</think> visible ",
        "<tool_call>{\"name\":\"file_read\",\"arguments\":{\"path\":\"a.txt\"}}",
        "</tool_call>",
    ]:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())

    content = "".join(event.content for event in events if event.type == "content")
    reasoning = "".join(event.content for event in events if event.type == "reasoning")
    assert content == "hello  visible "
    assert reasoning == "reasoning"
    assert len(parser.tool_calls) == 1
    call = parser.tool_calls[0]
    assert call["function"]["name"] == "file_read"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}
    assert call["id"].startswith("call_")


def test_tagged_parser_parses_xml_tool_call_with_parameters():
    """qwen3.5-agentic-coder 用 XML 工具调用（见模型自带 chat_template.jinja）。

    旧实现只会 json.loads，XML payload 必然解析失败 → tool_calls 恒为空 →
    task_plan 永不被调用 → 任务计划面板一直空着。
    """
    raw = (
        "先想一下。\n"
        + _xml_tool_call("task_plan", op="create", goal="整理项目结构")
    )
    parser = _TaggedOutputParser()
    events = _feed(parser, raw, step=5)

    assert len(parser.tool_calls) == 1, f"tool_calls={parser.tool_calls}"
    call = parser.tool_calls[0]
    assert call["function"]["name"] == "task_plan"
    args = json.loads(call["function"]["arguments"])
    assert args == {"op": "create", "goal": "整理项目结构"}
    # 工具调用段本身不应泄漏进对话区
    content = "".join(e.content for e in events if e.type == "content")
    assert content == "先想一下。\n"


def test_tagged_parser_xml_tool_call_closer_split_across_chunks():
    """闭合标签跨 chunk 拆分时仍要能解析出工具调用。

    旧实现在 tool 模式下把整段 buffer 倒进 _tool_buffer 后清空，跨 chunk 的
    闭合标签永远等不到——tool_calls 恒为空。
    """
    raw = _xml_tool_call("task_plan", op="get")
    # 步长 1：闭合标签必然被拆成多片
    parser = _TaggedOutputParser()
    _feed(parser, raw, step=1)
    assert len(parser.tool_calls) == 1, f"tool_calls={parser.tool_calls}"
    assert json.loads(parser.tool_calls[0]["function"]["arguments"]) == {"op": "get"}


def test_tagged_parser_xml_multiple_functions_in_one_block():
    """同一个 tool_call 块里的多个 <function>（并行调用）都要解析出来。"""
    raw = (
        _TC + "\n"
        + _FN + "task_plan>\n" + _PA + "op>\nget\n" + _PE + "\n" + _FE + "\n"
        + _FN + "file_read>\n" + _PA + "path>\na.txt\n" + _PE + "\n" + _FE + "\n"
        + _TCE
    )
    parser = _TaggedOutputParser()
    _feed(parser, raw, step=4)
    names = [c["function"]["name"] for c in parser.tool_calls]
    assert names == ["task_plan", "file_read"], names


def test_tagged_parser_xml_typed_parameters():
    """XML 参数值是纯文本，需还原成 bool / 数字 / list / dict。"""
    raw = _xml_tool_call(
        "task_plan",
        op="create",
        steps=json.dumps(["a", "b"], ensure_ascii=False),
        retries=3,
        enabled="true",
        disabled="false",
        ratio=1.5,
    )
    parser = _TaggedOutputParser()
    _feed(parser, raw, step=6)
    assert len(parser.tool_calls) == 1
    args = json.loads(parser.tool_calls[0]["function"]["arguments"])
    assert args["steps"] == ["a", "b"]
    assert args["retries"] == 3
    assert args["enabled"] is True
    assert args["disabled"] is False
    assert args["ratio"] == 1.5
    assert args["op"] == "create"


def test_tagged_parser_json_tool_call_still_works():
    """回归：原有 JSON 工具调用格式必须继续可用。"""
    raw = (
        _TC + "\n"
        + json.dumps({"name": "file_read", "arguments": {"path": "a.txt"}})
        + "\n" + _TCE
    )
    parser = _TaggedOutputParser()
    _feed(parser, raw, step=1)
    assert len(parser.tool_calls) == 1
    call = parser.tool_calls[0]
    assert call["function"]["name"] == "file_read"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}


def test_coerce_param_scalar_and_container():
    assert _coerce_param("create") == "create"
    assert _coerce_param("true") is True
    assert _coerce_param("false") is False
    assert _coerce_param("123") == 123
    assert _coerce_param("1.5") == 1.5
    assert _coerce_param('["a", "b"]') == ["a", "b"]
    assert _coerce_param('{"k": 1}') == {"k": 1}
    assert _coerce_param("") == ""
    assert _coerce_param("  普通文本  ") == "普通文本"


def test_template_messages_restores_arguments_as_mapping():
    """agent 历史里 arguments 是 JSON 字符串；qwen3.5 模板按 mapping 渲染，
    字符串会让参数整段消失。_template_messages 负责还原成 dict。"""
    history = [
        {"role": "user", "content": "帮我制定一个计划"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1_0",
                    "type": "function",
                    "function": {
                        "name": "task_plan",
                        "arguments": '{"op": "create", "goal": "g"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1_0", "name": "task_plan", "content": "ok"},
    ]
    normalized = _template_messages(history)

    args = normalized[1]["tool_calls"][0]["function"]["arguments"]
    assert args == {"op": "create", "goal": "g"}
    # 原始历史不被改写（agent 内部仍需 JSON 字符串形状）
    assert history[1]["tool_calls"][0]["function"]["arguments"] == '{"op": "create", "goal": "g"}'
    # 非 tool_calls 消息原样通过
    assert normalized[0] == {"role": "user", "content": "帮我制定一个计划"}


def test_template_messages_keeps_malformed_arguments_untouched():
    """解析不了的 arguments 字符串保持原样，不让模板渲染炸掉。"""
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{不是合法 JSON"},
                }
            ],
        }
    ]
    normalized = _template_messages(history)
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == "{不是合法 JSON"


def test_prompt_opens_thinking_detects_qwen3_generation_prompt():
    """qwen3.x 模板的 generation prompt 以思考开标签收尾（见模型自带
    chat_template.jinja 的 add_generation_prompt 分支）。"""
    assert _prompt_opens_thinking("assistant\n" + _THO + "\n")
    assert _prompt_opens_thinking("assistant\n" + _THO)
    # 历史里残留的开标签不算，只认末尾
    assert not _prompt_opens_thinking("user\nhi\nassistant\n" + _THO + "\n答案\nassistant\n")
    assert not _prompt_opens_thinking("assistant\n")
    assert not _prompt_opens_thinking("")
    assert not _prompt_opens_thinking(None)


def test_tagged_parser_reasoning_when_prompt_preopened_think():
    """核心回归：prompt 预置思考开标签时，生成内容里只有闭标签。

    qwen3.5-agentic-coder 的真实输出就是这样（开标签在 prompt 里）。旧实现按
    content 模式傻等一个永远不会出现的 think 开标签，整段思考被当成普通回复
    塞进对话区——「思考过程不显示」的直接原因。
    """
    raw = (
        "用户想要制定计划，先扫描目录。\n\n"
        "让我规划步骤：\n1. 扫描\n2. 输出清单\n\n"
        "先创建任务计划。" + _THE + "\n\n"
        + _xml_tool_call("task_plan", op="create", goal="扫描并输出清单")
    )
    parser = _TaggedOutputParser()
    parser.begin_reasoning()
    events = _feed(parser, raw, step=9)

    reasoning = "".join(e.content for e in events if e.type == "reasoning")
    content = "".join(e.content for e in events if e.type == "content")
    assert "用户想要制定计划" in reasoning
    assert "让我规划步骤" in reasoning
    assert _THE not in reasoning
    # 思考段结束后紧跟的工具调用仍要解析出来
    assert len(parser.tool_calls) == 1
    assert json.loads(parser.tool_calls[0]["function"]["arguments"]) == {
        "op": "create",
        "goal": "扫描并输出清单",
    }


def test_tagged_parser_begin_reasoning_is_noop_after_content_started():
    """已经开始产出 content 后再调用 begin_reasoning 不应改写模式。"""
    parser = _TaggedOutputParser()
    # 喂足够长的文本：feed 会按 opener 长度预留尾部，太短不产出 content。
    events = list(parser.feed("已经说了半句话，足够长了。"))
    assert events and events[0].type == "content"
    parser.begin_reasoning()
    assert parser._mode == "content"


def test_tagged_parser_reasoning_without_preopen_still_works():
    """回归：prompt 未预置思考标签时，生成内容自带 think 开标签照样分流。"""
    parser = _TaggedOutputParser()
    assert not _prompt_opens_thinking("assistant\n")
    events = _feed(parser, _THO + "思考内容" + _THE + "答复", step=4)
    reasoning = "".join(e.content for e in events if e.type == "reasoning")
    content = "".join(e.content for e in events if e.type == "content")
    assert reasoning == "思考内容"
    assert content == "答复"


def test_tagged_parser_preopened_reasoning_survives_truncation():
    """预置思考模式但模型被截断（没有闭标签）时，残段归思考而非对话区。"""
    parser = _TaggedOutputParser()
    parser.begin_reasoning()
    events = list(parser.feed("思考到一半被截断"))
    events.extend(parser.finish())
    reasoning = "".join(e.content for e in events if e.type == "reasoning")
    content = "".join(e.content for e in events if e.type == "content")
    assert reasoning == "思考到一半被截断"
    assert content == ""


def test_resolve_model_path_accepts_project_relative_directory():
    resolved = resolve_model_path("models/nomic-embed-text-v1.5-mlx")
    assert os.path.isabs(resolved)
    assert resolved.endswith("models/nomic-embed-text-v1.5-mlx")


def test_discover_local_models_filters_by_backend(tmp_path):
    mlx_dir = tmp_path / "mlx-model"
    mlx_dir.mkdir()
    (mlx_dir / "config.json").write_text("{}")
    (mlx_dir / "model.safetensors").write_bytes(b"x")
    onnx_dir = tmp_path / "onnx-model"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"x")
    (tmp_path / "embed.gguf").write_bytes(b"x")

    assert discover_local_models(str(tmp_path), "mlx_lm") == [str(mlx_dir.resolve())]
    assert discover_local_models(str(tmp_path), "onnx") == [str(onnx_dir.resolve())]
    assert discover_local_models(str(tmp_path), "llama_cpp") == [str((tmp_path / "embed.gguf").resolve())]


def test_factory_preserves_openai_backend():
    # 显式指定 openai：默认值现在是本地 MLX（#140），但接口模式仍可用。
    cfg = AgentConfig(llm_mode="openai", llm_backend="openai")
    backend = create_llm_backend(cfg)
    try:
        assert isinstance(backend, LLMClient)
    finally:
        backend.close()


def test_factory_defaults_to_local_mlx_backend():
    """#140：默认本地模式 + MLX-LM，且是懒加载（不读权重）。"""
    cfg = AgentConfig()
    assert cfg.llm_mode == "local"
    backend = create_llm_backend(cfg)
    try:
        assert isinstance(backend, MLXLanguageBackend)
        assert backend._model_obj is None
    finally:
        backend.close()


def test_aux_factory_builds_local_backend_with_aux_sampling(tmp_path):
    """辅助工厂只建本地后端，并把 aux_* 采样参数映射到标准字段名。"""
    from agent.model_backends import aux_sampling_config, create_aux_llm_backend

    cfg = AgentConfig(
        aux_llm_backend="mlx_lm",
        aux_llm_model_path=str(tmp_path),
        aux_temperature=0.15,
        aux_max_tokens=1234,
        aux_top_p=0.5,
        aux_min_p=0.02,
        aux_top_k=7,
        aux_repetition_penalty=1.3,
        aux_request_timeout=33.0,
        aux_max_context_tokens=8192,
        # 主模型参数必须被盖掉，不能被辅助模型读到。
        temperature=0.9,
        max_tokens=9999,
        max_context_tokens=32768,
        top_p=0.11,
    )
    backend = create_aux_llm_backend(cfg)
    try:
            assert isinstance(backend, MLXAuxBackend)
            view = backend.config
            assert view.temperature == 0.15
            assert view.max_tokens == 1234
            # 辅助模型用自己的上下文窗口（默认 16384），不跟主模型共用。
            assert view.max_context_tokens == 8192
            assert view.top_p == 0.5
            assert view.min_p == 0.02
            assert view.top_k == 7
            assert view.repetition_penalty == 1.3
            assert view.request_timeout == 33.0
            # 硬件约束类参数仍与主模型共用。
            assert view.local_model_threads == cfg.local_model_threads
            # 原始 config 不被视图污染。
            assert cfg.temperature == 0.9
            assert cfg.max_tokens == 9999
            # 视图是实时的：改完 aux_* 参数立即生效，不用重新加载模型。
            cfg.aux_temperature = 1.5
            cfg.aux_max_tokens = 42
            cfg.aux_max_context_tokens = 512
            assert view.temperature == 1.5
            assert view.max_tokens == 42
            assert view.max_context_tokens == 512
    finally:
        backend.close()

    # 辅助后端不接受 openai：没有网络端点可配。
    with pytest.raises(Exception):
        create_aux_llm_backend(AgentConfig(aux_llm_backend="openai"))

    # 缺省字段时视图有兜底值，不会 AttributeError。
    bare = AgentConfig(aux_llm_backend="mlx_lm")
    view = aux_sampling_config(bare)
    assert view.max_tokens == bare.aux_max_tokens


def test_factory_builds_lazy_mlx_backend_without_loading_weights(tmp_path):
    cfg = AgentConfig(llm_backend="mlx_lm", llm_model_path=str(tmp_path))
    backend = create_llm_backend(cfg)
    assert isinstance(backend, MLXLanguageBackend)
    assert backend._model_obj is None
    backend.close()


def test_unknown_local_backend_names_are_rejected(tmp_path):
    from agent.model_backends import create_llm_backend as factory
    from agent.rag.model_backends import create_embedding_backend, create_reranker_backend

    cfg = AgentConfig(llm_backend="bogus")
    try:
        factory(cfg)
    except Exception as exc:  # LLMError
        assert "bogus" in str(exc)
    else:  # pragma: no cover - 未拒绝即为失败
        raise AssertionError("未知主模型后端应被拒绝")

    try:
        create_embedding_backend("bogus", "m", "models")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知 Embedding 后端应被拒绝")

    assert create_reranker_backend("disabled", "m", "models") is None
    # MLX 交叉编码器是目录，走独立的 mlx_model_path（有则优先）。
    from agent.rag.model_backends import MLXReranker

    mlx_dir = tmp_path / "mlx-reranker"
    mlx_dir.mkdir()
    reranker = create_reranker_backend(
        "mlx", "m", "", mlx_model_path=str(mlx_dir), batch_size=4,
    )
    try:
        assert isinstance(reranker, MLXReranker)
        assert reranker._model is None  # 懒加载
    finally:
        reranker.close()


def test_mlx_backends_fall_back_to_fastembed(monkeypatch):
    """MLX 是默认后端，非 Apple Silicon / 未安装 / 路径缺失时不能让 RAG 挂掉。"""
    import agent.rag.model_backends as mb

    # Embedding：真实回退类（构造是懒加载，不会触发下载）。
    monkeypatch.setattr(mb, "mlx_runtime_blocker", lambda: "测试：MLX 不可用")
    ef = mb.create_embedding_backend("mlx", "m", "models", threads=2)
    assert type(ef).__name__ == "FastEmbedEmbeddingFunction"

    # Reranker：fastembed 会立即实例化，用哨兵替换以只验证「走了回退分支」。
    sentinel = object()
    monkeypatch.setattr(mb, "mlx_runtime_blocker", lambda: "")
    monkeypatch.setattr(mb, "MLXReranker", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("权重缺失")))
    monkeypatch.setattr(mb, "FastEmbedReranker", lambda *a, **kw: sentinel)
    assert mb.create_reranker_backend("mlx", "m", "models", threads=2) is sentinel


def test_benchmark_result_roundtrip_and_recommendation(tmp_path):
    from agent.model_benchmark import load_results, recommend, result_for, save_result, summarize

    path = str(tmp_path / "bench.json")
    embedding = {
        "kind": "embedding",
        "mlx": {"items_per_second": 160.0, "load_and_warmup_seconds": 5.0},
        "fastembed_cpu": {"items_per_second": 47.0, "load_and_warmup_seconds": 0.6},
        "speedup_mlx_over_fastembed": 3.4,
        "mean_vector_cosine_alignment": 0.962,
        "min_vector_cosine_alignment": 0.957,
    }
    # 每个后端小节自带 speedup / pearson（三方对比后由 benchmark_rerankers 写入）。
    reranker = {
        "kind": "reranker",
        "mlx": {
            "items_per_second": 33.9, "load_and_warmup_seconds": 8.26,
            "speedup_vs_fastembed": 2.23, "pearson_correlation": 0.997,
        },
        "onnx_direct_cpu": {
            "items_per_second": 14.2, "load_and_warmup_seconds": 3.18,
            "speedup_vs_fastembed": 0.93, "pearson_correlation": 1.0,
        },
        "fastembed_cpu": {"items_per_second": 15.2, "load_and_warmup_seconds": 1.99},
    }
    save_result(embedding, path)
    save_result(reranker, path)

    stored = load_results(path)
    assert len(stored) == 2
    assert result_for("embedding", path)["speedup_mlx_over_fastembed"] == 3.4
    assert "建议切换到 mlx" in recommend(result_for("embedding", path))
    # 三方里 MLX 最快且一致性达标 → 推荐 mlx 而不是 onnx。
    assert "建议切换到 mlx" in recommend(result_for("reranker", path))
    assert "MLX 161" not in summarize(embedding)  # 摘要跟随实际数值
    assert "160.0" in summarize(embedding)

    # 同 kind 覆盖：切换逻辑只关心最近一次测量。
    slower = dict(embedding, speedup_mlx_over_fastembed=0.8)
    save_result(slower, path)
    stored = load_results(path)
    assert len(stored) == 2
    assert result_for("embedding", path)["speedup_mlx_over_fastembed"] == 0.8
    assert "保持 fastembed" in recommend(result_for("embedding", path))

    # 只快但向量不一致 → 必须回归，不能直接建议切换。
    risky = dict(embedding, min_vector_cosine_alignment=0.5)
    assert "回归检索质量" in recommend(risky)


def test_reranker_max_length_follows_model_position_embeddings(tmp_path):
    """bge-reranker-v2-m3 是 8192，旧版 base 只有 512，不能一刀切。"""
    from agent.rag.model_backends import _model_max_length

    v2 = tmp_path / "v2-m3"
    v2.mkdir()
    (v2 / "config.json").write_text(json.dumps({"max_position_embeddings": 8194}))
    assert _model_max_length(v2) == 8192

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"max_position_embeddings": 514}))
    assert _model_max_length(base) == 512

    # config 缺失/非法时退回保守的 512，而不是无限放大。
    assert _model_max_length(tmp_path / "missing") == 512
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{ not json")
    assert _model_max_length(broken) == 512


# ---------------------------------------------------------------------------
# KV 前缀复用（对应 Ollama 的前缀缓存）。
#
# 每轮 prompt = 固定头部（system + 18 个工具 schema，实测约 2432 token）+ 历史
# + 新问题。旧实现每轮把整个 prompt 重新 prefill，M1 上约 160 tok/s，光头部就
# ~15s，用户感知「本地模型特别慢 / 每次都像重新加载」。Ollama 快很大程度上靠
# 缓存这段不变的前缀。复用的前提是：历史渲染与上一轮生成序列逐 token 一致——
# 由 agent 的「双视图历史」（content 解析后 + raw_content 原始）保证，详见
# _template_messages 与 agent._run。
# ---------------------------------------------------------------------------

def _install_mock_mlx(monkeypatch, replies, with_encode=True):
    """安装一个最小 mlx_lm 替身，依次吐出 replies（文本, 编码后的 token 列表）。

    gen_ids 必须是文本的编码，这样缓存序列（prompt 编码 + 生成编码）才能与第二
    轮 prompt 里对应的位置逐 token 对齐——这正是真实 tokenizer 天然满足、而缓存
    复用成立的前提。
    """

    class _FakeTok:
        def apply_chat_template(self, messages, **kw):
            # 极简渲染：assistant 用 raw_content 重建，与真实模板一致的关键属性
            # （前缀对齐）由 _template_messages 负责，这里只保证可编码、可反推。
            parts = []
            for m in messages:
                parts.append("<|im_start|>" + m["role"])
                if m["role"] == "assistant" and m.get("raw_content"):
                    parts.append("\n" + _THO + "\n" + m["raw_content"])
                else:
                    parts.append("\n" + (m.get("content") or ""))
            parts.append("<|im_start|>assistant")
            return "\n".join(parts)

        def encode(self, text, **kw):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    tok_instance = _FakeTok()
    if not with_encode:
        tok_instance.encode = None

    class _Cache:
        def __init__(self):
            self.fed = 0

    seen = {"n": 0}

    def _stream_generate(model, tok, prompt, **kw):
        idx = seen["n"]
        seen["n"] += 1
        text, ids = replies[idx]
        enc = getattr(tok, "encode", None)
        if enc is not None and isinstance(prompt, str):
            n_in = len(enc(prompt))
        elif isinstance(prompt, str):
            n_in = len(prompt)  # 无 encode：退化为字符数，仅用于记录
        else:
            n_in = len(list(prompt))
        # 记录本次实际喂进去的 token 数（= 增量 prefill 量）。
        _stream_generate.calls.append(
            {"n_input_tokens": n_in, "used_cache": "prompt_cache" in kw}
        )
        for i, t in enumerate(ids):
            yield type("R", (), {"text": text[i:i + 1], "token": t,
                                 "prompt_tokens": n_in,
                                 "generation_tokens": i + 1})()

    _stream_generate.calls = []

    mlx = type(sys)("mlx_lm")
    mlx.load = lambda *a, **k: (object(), tok_instance)
    mlx.stream_generate = _stream_generate

    su = type(sys)("mlx_lm.sample_utils")
    su.make_sampler = lambda **k: None
    su.make_logits_processors = lambda **k: []

    cache_mod = type(sys)("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model, **kw: [_Cache() for _ in range(4)]

    models_pkg = type(sys)("mlx_lm.models")
    models_pkg.cache = cache_mod

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", su)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    return tok_instance, _stream_generate


def test_mlx_kv_prefix_reuse_skips_cached_prefill(monkeypatch, tmp_path):
    """多轮对话中，第二轮应跳过上一轮前缀的 prefill。

    用 mock mlx_lm 验证了整条链路：agent 把原始输出存进 raw_content →
    _template_messages 据此重建忠实序列 → 后端比对 token 前缀命中缓存、只喂增量。
    """
    tok, stream = _install_mock_mlx(
        monkeypatch,
        [
            ("第一轮回复" + _xml_tool_call("task_plan", op="create"),
             [ord(c) for c in "第一轮回复" + _xml_tool_call("task_plan", op="create")]),
            ("第二轮回复", [ord(c) for c in "第二轮回复"]),
        ],
    )

    cfg = AgentConfig(workspace=os.getcwd(), llm_backend="mlx_lm",
                     llm_model_path=str(tmp_path))
    backend = create_llm_backend(cfg)
    try:
        # 第一轮
        list(backend.chat_stream(
            [{"role": "user", "content": "开始"}], tools=None, temperature=0.0))
        first_n = stream.calls[0]["n_input_tokens"]
        raw = backend.last_raw_output

        # 第二轮：历史里带 raw_content（agent 的真实行为）
        history = [
            {"role": "user", "content": "开始"},
            {"role": "assistant", "content": "第一轮回复", "raw_content": raw},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": "继续"},
        ]
        prior_cache = len(backend._kv_ids)
        list(backend.chat_stream(history, tools=None, temperature=0.0))
    finally:
        backend.close()

    second = stream.calls[1]
    second_prompt_len = second["n_input_tokens"] + prior_cache
    assert second["used_cache"], "第二轮必须命中 KV 缓存"
    # 必须跳过上一轮前缀，只 prefill 增量。
    assert second["n_input_tokens"] == second_prompt_len - prior_cache, (
        "第二轮应只 prefill 增量，而非全量 %d" % second_prompt_len
    )
    assert second["n_input_tokens"] > 0, "仍有待 prefill 的增量"
    assert prior_cache > 0, "上一轮缓存非空"


def test_mlx_kv_cache_invalidated_on_reset(monkeypatch, tmp_path):
    """reset / 历史被改写后，缓存应被清掉，避免前缀错位。"""
    tok, stream = _install_mock_mlx(
        monkeypatch,
        [("回复一", [ord(c) for c in "回复一"]), ("回复二", [ord(c) for c in "回复二"])],
    )
    cfg = AgentConfig(workspace=os.getcwd(), llm_backend="mlx_lm",
                     llm_model_path=str(tmp_path))
    backend = create_llm_backend(cfg)
    try:
        list(backend.chat_stream([{"role": "user", "content": "a"}], tools=None))
        assert len(backend._kv_ids) > 0
        backend.invalidate_kv_cache()
        assert backend._kv_cache is None
        assert backend._kv_ids == []
    finally:
        backend.close()


def test_mlx_kv_reuse_silent_when_tokenizer_has_no_encode(monkeypatch, tmp_path):
    """tokenizer 没有 encode（极端情况）时，退化为无缓存，不报错。"""
    tok, stream = _install_mock_mlx(
        monkeypatch, [("回复", [ord(c) for c in "回复"])], with_encode=False)
    cfg = AgentConfig(workspace=os.getcwd(), llm_backend="mlx_lm",
                     llm_model_path=str(tmp_path))
    backend = create_llm_backend(cfg)
    try:
        list(backend.chat_stream([{"role": "user", "content": "a"}], tools=None))
        # 无 encode → 不记录缓存，下一轮不会尝试复用
        assert backend._kv_ids == []
    finally:
        backend.close()


def test_reranker_batch_shrinks_with_sequence_length():
    """交叉编码器注意力 O(L²)：长序列必须降批，否则内存被打爆。"""
    from agent.rag.model_backends import _rerank_batch_for

    assert _rerank_batch_for(16, 128) == 16
    assert _rerank_batch_for(16, 512) == 16   # chunk 默认 500 tokens 的常态
    assert _rerank_batch_for(16, 1024) == 8
    assert _rerank_batch_for(16, 2048) == 4
    assert _rerank_batch_for(16, 8192) == 1
    # 预算内永远至少跑 1 条。
    assert _rerank_batch_for(4, 99999) == 1


def test_onnx_reranker_splits_long_batches_without_losing_rows():
    """token 预算切批后，每一行都要被打分且顺序不变。"""
    import threading

    import numpy as np

    from agent.rag import model_backends as mb

    class _FakeTokenizer:
        def __call__(self, queries, docs, **kwargs):
            rows = len(docs)
            # seq=3000 → 预算 8192 只允许 2 条一批，5 条会被切成 2+2+1。
            ids = np.tile(np.arange(3000, dtype=np.int64), (rows, 1))
            return {"input_ids": ids, "attention_mask": np.ones_like(ids)}

    class _Input:
        def __init__(self, name):
            self.name = name

    class _FakeSession:
        def get_inputs(self):
            return [_Input("input_ids"), _Input("attention_mask")]

        def run(self, _names, feeds):
            rows = feeds["input_ids"]
            assert rows.shape[0] <= 2, rows.shape
            # 输出行号，便于核对覆盖与顺序。
            return [np.arange(rows.shape[0], dtype=np.float32).reshape(-1, 1)]

    reranker = mb.ONNXReranker.__new__(mb.ONNXReranker)
    reranker._session = _FakeSession()
    reranker._tokenizer = _FakeTokenizer()
    reranker._lock = threading.RLock()
    reranker._max_length = 8192
    reranker._threads = 4

    scores = reranker.rerank("q", ["a", "b", "c", "d", "e"], batch_size=5)
    assert scores == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_config_default_rerank_length_covers_v2_m3(tmp_path):
    cfg = AgentConfig()
    assert cfg.rag_rerank_max_length == 8192
    # 合法范围内的自定义值必须保留（真正的上限由各模型 config 夹紧）。
    # 注意：钳制发生在 load()→_migrate()，直接构造不触发。
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"rag_rerank_max_length": 1024}))
    assert AgentConfig.load(str(path)).rag_rerank_max_length == 1024
    path.write_text(json.dumps({"rag_rerank_max_length": 0}))
    assert AgentConfig.load(str(path)).rag_rerank_max_length == 8


def test_embedding_max_length_follows_model_position_embeddings(tmp_path):
    """NomicBERT 旋转式取 n_positions=8192；bge-small 绝对位置取 512；缺失保守 512。"""
    from agent.rag.model_backends import _model_max_length

    nomic = tmp_path / "nomic"
    nomic.mkdir()
    (nomic / "config.json").write_text(
        json.dumps({"n_positions": 8192, "max_position_embeddings": 2048})
    )
    assert _model_max_length(nomic) == 8192

    bge = tmp_path / "bge"
    bge.mkdir()
    (bge / "config.json").write_text(json.dumps({"max_position_embeddings": 514}))
    assert _model_max_length(bge) == 512

    # config 缺失/非法时退回保守的 512，而不是放大到旋转式上限。
    assert _model_max_length(tmp_path / "missing") == 512
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{ not json")
    assert _model_max_length(broken) == 512


def test_embedding_batch_shrinks_with_sequence_length():
    """Embedding 按后端取形状预算：mlx 放 B=4@8192，onnx 放 B=2@8192。"""
    from agent.rag.model_backends import _embedding_batch_for

    # 默认（无后端）沿用 6M：长切片预算早已不够，保底 2 条、不再夹到 1（#139）。
    assert _embedding_batch_for(100, 8192) == 2
    assert _embedding_batch_for(100, 2048) == 2

    # MLX 走 flash attention（O(S)），可放到 B=4@8192，短切片不受限。
    assert _embedding_batch_for(100, 8192, "mlx") == 4
    assert _embedding_batch_for(100, 2048, "mlx") == 71       # 300M//2048²
    assert _embedding_batch_for(100, 1024, "mlx") == 100       # 远未到预算

    # ONNX(CPU) 物化注意力（O(S²)），放到 B=2@8192，@2048 约 35 批（fp32 ~7GB 顶）。
    assert _embedding_batch_for(100, 8192, "onnx") == 2
    assert _embedding_batch_for(100, 2048, "onnx") == 35       # 150M//2048²
    assert _embedding_batch_for(8, 512, "onnx") == 8           # 默认切片常态不受限

    # 极小/非法序列长不崩，但下限仍是 2（除非请求本身只有 1 条）。
    assert _embedding_batch_for(4, 99999, "mlx") == 2
    assert _embedding_batch_for(1, 99999, "mlx") == 1


def test_onnx_embedding_function_splits_long_batches_without_losing_rows():
    """token 预算切批后，每一行都要被嵌入且顺序不变。"""
    import threading

    import numpy as np

    from agent.rag import model_backends as mb

    class _FakeTokenizer:
        def __call__(self, batch, **kwargs):
            rows = len(batch)
            ids = np.tile(np.arange(3000, dtype=np.int64), (rows, 1))  # seq=3000
            return {"input_ids": ids, "attention_mask": np.ones_like(ids)}

    class _Input:
        def __init__(self, name):
            self.name = name

    class _FakeSession:
        def get_inputs(self):
            return [_Input("input_ids"), _Input("attention_mask")]

        def run(self, _names, feeds):
            rows = feeds["input_ids"]
            # ONNX 预算 150M、seq=3000 → 内批上限 16；batch_size=8 故实际 ≤8。
            assert 1 <= rows.shape[0] <= 8, rows.shape
            return [np.zeros((rows.shape[0], 4), dtype=np.float32)]  # 4 维假向量

    ef = mb.ONNXEmbeddingFunction.__new__(mb.ONNXEmbeddingFunction)
    ef._session = _FakeSession()
    ef._tokenizer = _FakeTokenizer()
    ef._lock = threading.RLock()
    ef._max_length = 8192
    ef._batch_size = 8
    ef._document_prefix = ""
    ef._query_prefix = ""

    vectors = ef._embed(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    assert len(vectors) == 10                # 10 行全部被嵌入，无遗漏
    assert all(len(v) == 4 for v in vectors) # 形状正确


def test_config_default_embedding_length_1024(tmp_path):
    """2026-10-28：嵌入输入上限默认收紧到 1024（切片 ≤800 字符已全覆盖）。

    - 旧默认 8192 的存量配置经迁移重写为 1024；
    - 用户显式设置的其它值（如 2048，配合加大切片）不被迁移；
    - 非正值被拉到下限 8，不会把 tokenizer 的 max_length 置成 0/负。
    """
    cfg = AgentConfig()
    assert cfg.rag_embedding_max_length == 1024
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"rag_embedding_max_length": 8192}))
    assert AgentConfig.load(str(path)).rag_embedding_max_length == 1024
    path.write_text(json.dumps({"rag_embedding_max_length": 2048}))
    assert AgentConfig.load(str(path)).rag_embedding_max_length == 2048
    path.write_text(json.dumps({"rag_embedding_max_length": 0}))
    assert AgentConfig.load(str(path)).rag_embedding_max_length == 8


def test_template_messages_replays_raw_content_without_extra_think_block():
    """历史里带 raw_content 的 assistant 消息必须渲染出与生成时相同的序列。

    之前 `reasoning_content = ""` 会让模板插入空思考块 `<think>\\n\\n</think>\\n\\n`，
    raw 里没有 `</think>`（模型直接吐工具调用）时同样会插——两种都会让公共前缀
    掉到上一轮 prompt 的最后一个 token，KV 缓存整份作废。
    """
    from agent.model_backends import _template_messages

    tool_raw = "<tool_call>\n<function=rag_status>\n</tool_call>"
    out = _template_messages(
        [{"role": "assistant", "content": None, "raw_content": tool_raw,
          "tool_calls": [{"id": "c1"}]}],
        think_opened=True,
    )
    msg = out[0]
    assert msg["content"] == tool_raw + "\n</think>"   # 补闭合标签，不插空思考块
    assert "reasoning_content" not in msg              # 旧实现的空字符串暗门已移除
    assert "tool_calls" not in msg                     # raw 里已有 XML，避免重复渲染

    think_raw = "让我想想\n</think>\n\n答案是 42\n"
    out2 = _template_messages(
        [{"role": "assistant", "content": "答案是 42", "raw_content": think_raw}],
        think_opened=True,
    )
    assert out2[0]["content"] == think_raw             # 已有闭合标签则原样使用

    # 思考模式关闭时生成提示不预置 <think>，不能补闭合标签。
    out3 = _template_messages(
        [{"role": "assistant", "content": None, "raw_content": tool_raw}],
        think_opened=False,
    )
    assert out3[0]["content"] == tool_raw


# ---------------------------------------------------------------------------
# jina-embeddings-v5-text-nano embedding 后端路由
# ---------------------------------------------------------------------------

def test_jina_v5_nano_routes_to_mlx_backend():
    """v5-nano 标识 + mlx 后端 → JinaV5NanoEmbeddingFunction（构造不触发模型加载）。"""
    ef = create_embedding_backend(
        "mlx", "jinaai/jina-embeddings-v5-text-nano",
        "models/jina-embeddings-v5-text-nano-mlx-int4",
        batch_size=8, max_length=1024,
    )
    assert isinstance(ef, JinaV5NanoEmbeddingFunction)
    assert ef.name().startswith("jina_v5_nano:")


def test_jina_v5_nano_routes_to_onnx_backend():
    """v5-nano 标识 + onnx 后端 → JinaV5NanoONNXEmbeddingFunction（构造不触发模型加载）。"""
    ef = create_embedding_backend(
        "onnx", "jinaai/jina-embeddings-v5-text-nano",
        "models/jina-embeddings-v5-text-nano-onnx-int8",
        threads=4, batch_size=8, max_length=1024,
    )
    assert isinstance(ef, JinaV5NanoONNXEmbeddingFunction)
    assert ef.name().startswith("jina_v5_nano_onnx:")


def test_jina_v5_nano_rejects_unsupported_backend():
    """v5-nano 只支持 mlx / onnx，其它后端应抛出可操作错误（禁止静默回退）。"""
    with pytest.raises(ValueError):
        create_embedding_backend(
            "fastembed", "jinaai/jina-embeddings-v5-text-nano",
            "models/jina-embeddings-v5-text-nano-mlx-int4",
        )
