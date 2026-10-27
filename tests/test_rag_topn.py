"""Reranker TopN 配置（rag_rerank_top_n）链路测试。

TopN 在设置面板 Reranker 行配置，定义 reranker 精排后选取的结果数量：
rag_search 工具的 top_k 以它为上限，模型不传时用它做默认值。
"""

from types import SimpleNamespace

from agent.tools.rag_tool import RagSearchTool


class _FakeEngine:
    """记录 search() 收到的 top_k；返回可辨识的假结果。"""

    def __init__(self):
        self.calls = []

    def search(self, query, top_k=3, source=None):
        self.calls.append({"query": query, "top_k": top_k, "source": source})
        return [{"source": "d.md", "text": "内容", "score": 0.9}]

    def format_results(self, results, *args, **kwargs):
        return "格式化结果"


def _make_tool(cfg):
    tool = RagSearchTool(_FakeEngine())
    tool.bind(cfg, None)
    return tool


def test_topn_default_when_model_omits_top_k():
    cfg = SimpleNamespace(rag_rerank_top_n=7)
    tool = _make_tool(cfg)
    tool.run(query="测试查询")            # 模型未传 top_k
    assert tool._engine.calls[-1]["top_k"] == 7


def test_topn_clamps_model_top_k_upward_request():
    cfg = SimpleNamespace(rag_rerank_top_n=3)
    tool = _make_tool(cfg)
    tool.run(query="测试查询", top_k=10)  # 模型要 10 → 夹到 TopN=3
    assert tool._engine.calls[-1]["top_k"] == 3


def test_topn_allows_model_top_k_within_limit():
    cfg = SimpleNamespace(rag_rerank_top_n=8)
    tool = _make_tool(cfg)
    tool.run(query="测试查询", top_k=5)
    assert tool._engine.calls[-1]["top_k"] == 5


def test_topn_floor_is_one():
    cfg = SimpleNamespace(rag_rerank_top_n=1)
    tool = _make_tool(cfg)
    tool.run(query="测试查询", top_k=5)
    assert tool._engine.calls[-1]["top_k"] == 1


def test_topn_defaults_to_3_without_config():
    tool = RagSearchTool(_FakeEngine())   # 未 bind config
    tool.run(query="测试查询", top_k=None)
    assert tool._engine.calls[-1]["top_k"] == 3
