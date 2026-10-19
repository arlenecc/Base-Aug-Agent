"""Contract tests for selectable local language-model backends."""
from __future__ import annotations

import json
import os

from agent.config import AgentConfig
from agent.llm_client import LLMClient
from agent.model_backends import (
    MLXLanguageBackend,
    _TaggedOutputParser,
    create_llm_backend,
    discover_local_models,
    resolve_model_path,
)


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
    backend = create_llm_backend(AgentConfig())
    try:
        assert isinstance(backend, LLMClient)
    finally:
        backend.close()


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
