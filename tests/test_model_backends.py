"""Contract tests for selectable local language-model backends."""
from __future__ import annotations

import json
import os

import pytest

from agent.config import AgentConfig
from agent.llm_client import LLMClient
from agent.model_backends import (
    MLXLanguageBackend,
    _TaggedOutputParser,
    _coerce_param,
    _template_messages,
    create_llm_backend,
    discover_local_models,
    resolve_model_path,
)

# XML 工具调用标签：用字符拼接构造，源码里不出现完整标签。
_LT, _GT = chr(60), chr(62)
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
            assert isinstance(backend, MLXLanguageBackend)
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


def test_config_default_embedding_length_8192(tmp_path):
    cfg = AgentConfig()
    assert cfg.rag_embedding_max_length == 8192
    # 非正值被拉到下限 8，不会把 tokenizer 的 max_length 置成 0/负。
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"rag_embedding_max_length": 0}))
    assert AgentConfig.load(str(path)).rag_embedding_max_length == 8
