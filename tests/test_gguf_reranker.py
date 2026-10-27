"""GGUF Reranker（llama.cpp RANK 池化）单元测试。

用假 llama_cpp 模块验证：工厂路由、Llama 构造参数（RANK 池化 /
n_ctx=n_batch）、输入格式 query+sep+doc（special token 强制解析）、
分数提取与 close()。真模型端到端由人工验收。
"""

import sys
import types

import pytest

from agent.rag.model_backends import GGUFReranker, create_reranker_backend


class _FakeLlama:
    instances = []

    def __init__(self, model_path="", embedding=False, pooling_type=None,
                 n_ctx=512, n_batch=512, n_threads=4, verbose=False):
        self.model_path = model_path
        self.embedding = embedding
        self.pooling_type = pooling_type
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_threads = n_threads
        # bge 系 GGUF 的分隔符元数据（token id）
        self.metadata = {"tokenizer.ggml.separation_token_id": "2"}
        self.tokenize_calls = []
        self.embed_inputs = []
        self._scores = [1.5, 0.5, -2.0]
        _FakeLlama.instances.append(self)

    def tokenize(self, text, add_bos=True, special=False):
        self.tokenize_calls.append({"special": special})
        return list(text[:64])

    def detokenize(self, ids):
        return b"</s>"

    def embed(self, texts):
        texts = list(texts)
        # 仿真 llama-cpp-python：embed 内部逐条调用 self.tokenize（无显式
        # special 参数 → 走 _RerankLlama 覆盖后的默认 special=True）。
        for t in texts:
            self.tokenize(t.encode("utf-8"))
        self.embed_inputs.append(texts)
        return [[self._scores[i]] for i in range(len(texts))]

    def close(self):
        self.closed = True


@pytest.fixture()
def fake_llama_cpp(monkeypatch):
    _FakeLlama.instances = []
    pkg = types.ModuleType("llama_cpp")
    pkg.Llama = _FakeLlama
    pkg.LLAMA_POOLING_TYPE_RANK = 4
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    return pkg


def test_factory_routes_gguf(fake_llama_cpp):
    reranker = create_reranker_backend("gguf", "x", "/tmp/reranker.gguf")
    assert isinstance(reranker, GGUFReranker)
    llm = reranker._llm  # _RerankLlama，__getattr__ 委托到内部 _FakeLlama
    assert llm.embedding is True
    assert llm.pooling_type == 4          # LLAMA_POOLING_TYPE_RANK
    assert llm.n_ctx == llm.n_batch       # n_batch 不足会把输入截到 512
    assert llm.model_path == "/tmp/reranker.gguf"


def test_rerank_format_and_scores(fake_llama_cpp):
    reranker = create_reranker_backend("gguf", "x", "/tmp/reranker.gguf")
    scores = reranker.rerank("什么是MLX？", ["苹果的ML框架", "一种金属"], batch_size=2)
    llm = reranker._llm
    # 分隔符取自元数据 separation_token_id=2 → detokenize → </s>
    assert llm.embed_inputs[-1] == [
        "什么是MLX？</s>苹果的ML框架",
        "什么是MLX？</s>一种金属",
    ]
    assert scores == [1.5, 0.5]
    # special=True 强制开启：否则 </s> 被切成字面字符
    assert llm.tokenize_calls[-1]["special"] is True


def test_sep_falls_back_when_metadata_missing(fake_llama_cpp):
    r = GGUFReranker("/tmp/r.gguf")
    r._llm.metadata = {}
    assert r._sep_from_metadata() == "</s>"
    r._llm.metadata = {"tokenizer.ggml.separation_token": "<sep>"}
    assert r._sep_from_metadata() == "<sep>"


def test_close_releases_llm(fake_llama_cpp):
    reranker = create_reranker_backend("gguf", "x", "/tmp/r.gguf")
    llm = reranker._llm
    reranker.close()
    assert getattr(llm, "closed", False)


def test_missing_rank_pooling_raises(monkeypatch):
    pkg = types.ModuleType("llama_cpp")
    pkg.Llama = _FakeLlama
    # 老版本没有 LLAMA_POOLING_TYPE_RANK
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    with pytest.raises(RuntimeError, match="LLAMA_POOLING_TYPE_RANK"):
        GGUFReranker("/tmp/r.gguf")
