"""Tests for PDF engine routing in rag.engine (#134: docling | marker 并行)."""
from __future__ import annotations

import pytest


@pytest.fixture()
def engine_mod():
    from agent.rag import engine as mod

    yield mod


# ---------------------------------------------------------------------------
# _extract_with_pdf_engine 回退链
# ---------------------------------------------------------------------------

def test_primary_success(engine_mod, monkeypatch):
    """主引擎可用且转换成功 → 直接返回其 Markdown。"""
    import agent.rag.docling_parser as dp

    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "extract_markdown", lambda p: "# md")
    monkeypatch.setattr(engine_mod, "extract_text", lambda p: "SHOULD NOT BE CALLED")

    text, is_md = engine_mod._extract_with_pdf_engine("f.pdf", "docling")
    assert (text, is_md) == ("# md", True)


def test_primary_unavailable_alt_engine_takes_over(engine_mod, monkeypatch):
    """主引擎未安装 → 另一个 Markdown 引擎顶上（而非掉到轻量解析器）。"""
    import agent.rag.marker_parser as mp

    monkeypatch.setattr(mp, "is_marker_available", lambda: False)
    monkeypatch.setattr(mp, "extract_markdown", lambda p: None)
    # docling 侧（alt）必须被调用并成功
    import agent.rag.docling_parser as dp
    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "extract_markdown", lambda p: "# from docling")

    text, is_md = engine_mod._extract_with_pdf_engine("f.pdf", "marker")
    assert (text, is_md) == ("# from docling", True)


def test_primary_available_but_file_fails_goes_lightweight(engine_mod, monkeypatch):
    """主引擎可用但单文件失败 → 只走轻量兜底（不跨引擎冷启动）。"""
    import agent.rag.marker_parser as mp

    monkeypatch.setattr(mp, "is_marker_available", lambda: True)
    monkeypatch.setattr(mp, "extract_markdown", lambda p: None)  # 转换失败
    # docling（alt）绝不能被调用：mid-ingest 冷启动重型引擎是禁止的
    import agent.rag.docling_parser as dp
    monkeypatch.setattr(dp, "is_docling_available", lambda: True)
    monkeypatch.setattr(dp, "extract_markdown",
                        lambda p: (_ for _ in ()).throw(AssertionError("alt must not run")))
    monkeypatch.setattr(engine_mod, "extract_text", lambda p: "plain text")

    text, is_md = engine_mod._extract_with_pdf_engine("f.pdf", "marker")
    assert (text, is_md) == ("plain text", False)


def test_both_unavailable_goes_lightweight(engine_mod, monkeypatch):
    """两个 Markdown 引擎都不可用 → 轻量解析器兜底。"""
    import agent.rag.docling_parser as dp
    import agent.rag.marker_parser as mp

    monkeypatch.setattr(dp, "is_docling_available", lambda: False)
    monkeypatch.setattr(dp, "extract_markdown", lambda p: None)
    monkeypatch.setattr(mp, "is_marker_available", lambda: False)
    monkeypatch.setattr(mp, "extract_markdown", lambda p: None)
    monkeypatch.setattr(engine_mod, "extract_text", lambda p: "plain")

    text, is_md = engine_mod._extract_with_pdf_engine("f.pdf", "docling")
    assert (text, is_md) == ("plain", False)


# ---------------------------------------------------------------------------
# RAGEngine 构造参数
# ---------------------------------------------------------------------------

def test_engine_accepts_pdf_engine_param(tmp_path):
    """pdf_engine 参数被存储；未知值回退 docling。"""
    from agent.rag.engine import RAGEngine

    ws = str(tmp_path)
    eng = RAGEngine(workspace=ws, pdf_engine="marker")
    assert eng._pdf_engine == "marker"
    eng2 = RAGEngine(workspace=ws, pdf_engine="bogus")
    assert eng2._pdf_engine == "docling"
    eng3 = RAGEngine(workspace=ws)
    assert eng3._pdf_engine == "docling"


def test_config_field_roundtrip(tmp_path):
    """rag_pdf_engine 配置持久化往返。"""
    import json
    from agent.config import AgentConfig

    p = str(tmp_path / "config.json")
    cfg = AgentConfig()
    cfg.rag_pdf_engine = "marker"
    cfg.save(p)
    loaded = AgentConfig.load(p)
    assert loaded.rag_pdf_engine == "marker"
    # 非法值不阻断启动（引擎侧自行归一化为 docling）
    data = json.load(open(p))
    data["rag_pdf_engine"] = ""
    json.dump(data, open(p, "w"))
    cfg2 = AgentConfig.load(p)
    assert cfg2.rag_pdf_engine == "docling"
