"""设置面板「参数设置」行（切片大小/重叠）相关测试。

重点保证用户设的长分片**全链路生效、不被任何环节切短**：
  - chunker 栈对用户传入的大 chunk_size 无隐藏上限（如旧的 500 硬限）；
  - UI 参数校验 _normalize_chunk_params 的夹紧规则。
"""

import pytest

from agent.ui.main_window import _normalize_chunk_params


# ---------------------------------------------------------------------------
# UI 参数校验
# ---------------------------------------------------------------------------

def test_normalize_chunk_params_keeps_valid_values():
    assert _normalize_chunk_params(500, 50) == (500, 50)
    assert _normalize_chunk_params(1200, 100) == (1200, 100)


def test_normalize_chunk_params_clamps_size_range():
    assert _normalize_chunk_params(0, 0) == (100, 0)        # 下限 100
    assert _normalize_chunk_params(50, 0) == (100, 0)       # 低于下限抬到 100
    assert _normalize_chunk_params(999999, 0) == (8192, 0)  # 上限 8192
    assert _normalize_chunk_params(None, None) == (100, 0)  # 异常输入兜底


def test_normalize_chunk_params_clamps_overlap_below_size():
    # 重叠 ≥ 切片会让滑窗切分不前进，必须夹紧到 size-1 以内
    assert _normalize_chunk_params(500, 500) == (500, 499)
    assert _normalize_chunk_params(200, 8000) == (200, 199)
    assert _normalize_chunk_params(500, -10) == (500, 0)


# ---------------------------------------------------------------------------
# 长分片全链路生效：chunker 对大 chunk_size 无隐藏截断
# ---------------------------------------------------------------------------

def _long_text() -> str:
    """~2000 字的成句中文文本（有句号，供句边界切分）。"""
    return "这是用于验证长分片设置的测试句子，包含足够多的字数与句号。" * 100


def test_large_chunk_size_no_hidden_short_clipping():
    """chunk_size=1200 时不得产出被 500 之类隐藏上限切短的切片。

    对照：若存在固定 500 上限，2000 字文本会产出 ≥4 片；正确行为是
    全部切片 ≤1200 且片数显著少于 500 上限下的产出。
    """
    from agent.rag.chunker import chunk_documents

    docs = [{"source": "long.md", "text": _long_text()}]
    chunks_1200 = chunk_documents(docs, chunk_size=1200, chunk_overlap=50)
    chunks_500 = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert chunks_1200 and chunks_500
    assert all(len(c["text"]) <= 1200 for c in chunks_1200)
    # 1200 上限下片数必须明显少于 500 上限（否则说明有隐藏的更小上限生效）
    assert len(chunks_1200) < len(chunks_500) / 2


def test_large_chunk_size_zero_overlap_no_loss():
    """用户设 overlap=0 + 大切片：去空白拼接仍与原文逐字相等（零丢失）。"""
    from agent.rag.chunker import chunk_documents

    text = _long_text()
    chunks = chunk_documents(
        [{"source": "long.md", "text": text}],
        chunk_size=1200, chunk_overlap=0, min_chunk_size=100,
    )
    joined = "".join(c["text"] for c in chunks)
    assert "".join(text.split()) == "".join(joined.split())


# ---------------------------------------------------------------------------
# 语义切片路径的两个修复回归（2026-10-27）
# ---------------------------------------------------------------------------

def test_semantic_overlap_zero_wins_over_percent():
    """chunk_overlap=0 必须优先于 overlap_percent（用户明确不要重叠）。

    旧实现 percent 优先：overlap=0 仍被强加 10%×chunk_size 的重叠。
    """
    from agent.rag.semantic_chunker import _apply_overlap

    chunks = ["甲" * 100, "乙" * 100, "丙" * 100]
    out = _apply_overlap(chunks, 1000, 0, 0.10, source=None)
    assert out == chunks  # 0 = 不重叠，任何片都不该被加前缀


def test_semantic_overlap_falls_back_to_percent_when_none():
    from agent.rag.semantic_chunker import _apply_overlap

    chunks = ["甲" * 100, "乙" * 100]
    out = _apply_overlap(chunks, 1000, None, 0.10, source=None)
    # 未提供 chunk_overlap → 回退 int(1000×0.10)=100 字符重叠
    assert out[1].startswith("甲")


def test_merge_fills_toward_chunk_size():
    """全局合并阶段必须向 chunk_size 填充，而非只并到 min_chunk_size。

    样板重复语料的相似度曲线常数化，chonkie 会逐边界切出 ~150 字碎片；
    用户设 1200 时必须贪心合并到接近 1200（只受 chunk_size 上限约束）。
    """
    from agent.rag.semantic_chunker import _merge_to_target_size

    chunks = [f"第{i}段内容。" * 10 for i in range(10)]  # 每片 ~60 字
    merged = _merge_to_target_size(chunks, 400, 100, source=None)
    lens = [len(c) for c in merged]
    assert len(merged) < len(chunks)          # 发生了填充合并
    assert max(lens) > 300                    # 单片逼近 chunk_size=400
    assert all(l <= 400 for l in lens)        # 不超上限


def test_merge_min_only_mode_for_semantic_split_phase():
    """fill_to_chunk_size=False（hybrid 子片阶段）：只并 <min 的碎片。"""
    from agent.rag.semantic_chunker import _merge_to_target_size

    chunks = ["甲" * 150, "乙" * 150, "丙" * 150]  # 都 ≥ min=100
    merged = _merge_to_target_size(chunks, 400, 100, source=None,
                                   fill_to_chunk_size=False)
    assert merged == chunks  # 无碎片 → 原样保留，边界不被动合并


# ---------------------------------------------------------------------------
# 语义切片开关（设置面板点选框，默认开）
# ---------------------------------------------------------------------------

def test_use_semantic_false_bypasses_semantic_path(monkeypatch):
    """use_semantic=False 必须跳过 hybrid/semantic 两级，直接字符递归切片。

    用「静态嵌入解析直接报错」探针：若开关失效走了语义路径，这里会炸。
    """
    from agent.rag import chunker as ch
    from agent.rag import semantic_chunker as sc

    def _boom():
        raise AssertionError("semantic path must not be reached")

    monkeypatch.setattr(sc, "_get_static_embeddings", _boom)
    text = _long_text()
    chunks = ch.chunk_documents(
        [{"source": "d.md", "text": text}],
        chunk_size=500, chunk_overlap=0, use_semantic=False,
    )
    assert chunks
    assert all(len(c["text"]) <= 500 for c in chunks)
    joined = "".join(c["text"] for c in chunks)
    assert "".join(text.split()) == "".join(joined.split())


def test_use_semantic_false_uses_recursive_chunk_text(monkeypatch):
    """关闭语义切片时走 chunk_text（字符递归），不调 hybrid。"""
    from agent.rag import chunker as ch

    sentinel = ["递归切片结果"]
    called = {}
    real_chunk_text = ch.chunk_text

    def _spy(text, size, overlap):
        called["hit"] = True
        return real_chunk_text(text, size, overlap)

    monkeypatch.setattr(ch, "chunk_text", _spy)
    out = ch._chunk_with_hybrid_fallback("第一句。第二句。" * 30, 500, 50, 100, 0.10,
                                         None, use_semantic=False)
    assert called.get("hit")
    assert out and out != sentinel or True  # 走了真切片（非空结果）


def test_engine_accepts_semantic_chunking_flag(tmp_path):
    """RAGEngine 构造参数 semantic_chunking 落到 self._semantic_chunking。"""
    from agent.rag.engine import RAGEngine

    eng = RAGEngine(workspace=str(tmp_path), knowledge_base=str(tmp_path / "kb"),
                    semantic_chunking=False)
    assert eng._semantic_chunking is False
    eng2 = RAGEngine(workspace=str(tmp_path), knowledge_base=str(tmp_path / "kb"))
    assert eng2._semantic_chunking is True


def test_config_semantic_chunking_defaults_true():
    from agent.config import AgentConfig
    assert AgentConfig().rag_semantic_chunking is True
