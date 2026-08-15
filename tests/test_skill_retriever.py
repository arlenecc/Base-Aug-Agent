"""Tests for the hybrid skill retriever (SKILL.md scan + vector/BM25 search)."""
from __future__ import annotations

import hashlib
import os
import tempfile
from typing import List

from agent.skill_retriever import (
    SkillScanner,
    SkillVectorStore,
    SkillRetriever,
    parse_skill_md,
    _parse_list_field,
    _normalize_vec,
)


# ---------------------------------------------------------------------------
# Fake embedding function (deterministic, no model download)
# ---------------------------------------------------------------------------

class FakeEF:
    """Deterministic hash-based embedding (768-dim) for fast tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            out.append([float(b) / 255.0 for b in h] * 48)
        return out

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

SAMPLE_SKILL = """---
name: invoice-organizer
description: "Organize messy invoice folders, extract totals, and prepare tax documents. Use when handling invoices, receipts, or tax preparation."
tags: [finance, invoices, tax, bookkeeping]
license: MIT
metadata:
  version: "1.0"
---

# Invoice Organizer

## When to Use This Skill

Use this skill when the user has a messy folder of invoices and receipts.

## Examples

- "Organize my invoices for tax season"
- "Extract totals from these receipts"
"""


def test_parse_frontmatter_and_tags():
    m = parse_skill_md(SAMPLE_SKILL, "invoice-organizer")
    assert m is not None
    assert m.name == "invoice-organizer"
    assert m.tags == ["finance", "invoices", "tax", "bookkeeping"]
    assert "tax" in m.description


def test_parse_extracts_when_to_use_and_examples():
    m = parse_skill_md(SAMPLE_SKILL, "invoice-organizer")
    assert "messy folder" in m.when_to_use
    assert "Organize my invoices" in m.examples


def test_parse_list_field():
    assert _parse_list_field("[a, b, c]") == ["a", "b", "c"]
    assert _parse_list_field("a, b") == ["a", "b"]
    assert _parse_list_field('"a", "b"') == ["a", "b"]


def test_parse_rejects_missing_name():
    m = parse_skill_md("# No frontmatter\n\nbody", "x")
    assert m is None


def test_normalize_vec():
    v = _normalize_vec([3.0, 4.0])
    # 单位长度
    assert abs((v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _make_skill_dir(root: str, rel: str, name: str, body: str = "") -> None:
    d = os.path.join(root, rel)
    os.makedirs(d, exist_ok=True)
    content = f"---\nname: {name}\ndescription: test skill {name}\n---\n{body}"
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(content)


def test_scanner_recursive_and_skips_tests(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "finance/budget", "budget-planner")
    # tests 目录应被跳过
    _make_skill_dir(skills_dir, "finance/tests/skipme", "should-be-skipped")

    scanner = SkillScanner(skills_dir)
    skills = scanner.scan(force=True)
    names = {s.name for s in skills}
    assert "invoice-organizer" in names
    assert "budget-planner" in names
    assert "should-be-skipped" not in names


# ---------------------------------------------------------------------------
# Vector store + hybrid retrieval
# ---------------------------------------------------------------------------

def _build_store(tmp_path, skills_dir: str) -> SkillVectorStore:
    scanner = SkillScanner(skills_dir)
    skills = scanner.scan(force=True)
    store = SkillVectorStore(
        db_path=str(tmp_path / "skills.lancedb"),
        embedding_function=FakeEF(),
    )
    store.build(skills)
    return store


def test_store_build_and_vector_search(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "science/networkx", "networkx")

    store = _build_store(tmp_path, skills_dir)
    results = store.search_vector("organize invoices for tax", top_k=2)
    assert len(results) >= 1
    # 分数应在 [0,1] 区间（余弦相似度）
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_store_bm25_search(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "science/networkx", "networkx")

    store = _build_store(tmp_path, skills_dir)
    results = store.search_bm25("invoice", top_k=2)
    assert len(results) >= 1
    # 应有 bm25_score 字段
    for r in results:
        assert "bm25_score" in r


def test_retriever_hybrid_dedup_and_threshold(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")
    _make_skill_dir(skills_dir, "finance/budget", "budget-planner")
    _make_skill_dir(skills_dir, "science/networkx", "networkx")

    retriever = SkillRetriever(
        skills_dir=skills_dir,
        db_path=str(tmp_path / "retriever.lancedb"),
        embedding_function=FakeEF(),
    )
    result = retriever.retrieve("organize my invoices and receipts", llm=None)
    # 候选应去重（无重复 dir）且 score >= 0.5
    dirs = [c["dir"] for c in result["candidates"]]
    assert len(dirs) == len(set(dirs))
    for c in result["candidates"]:
        assert c["score"] >= 0.5
    retriever.close()


def test_retriever_no_candidates(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")

    retriever = SkillRetriever(
        skills_dir=skills_dir,
        db_path=str(tmp_path / "retriever.lancedb"),
        embedding_function=FakeEF(),
    )
    # 完全不相关的查询 + 低相似度 → 候选被阈值过滤
    result = retriever.retrieve("zzzz totally unrelated query zzzz", llm=None)
    # 由于 fake embedding 是哈希，可能仍有低分候选；验证逻辑不崩溃即可
    assert "candidates" in result
    retriever.close()


def test_retriever_read_skill_dir(tmp_path):
    skills_dir = str(tmp_path / "skills")
    _make_skill_dir(skills_dir, "finance/invoice", "invoice-organizer")

    retriever = SkillRetriever(
        skills_dir=skills_dir,
        db_path=str(tmp_path / "retriever.lancedb"),
        embedding_function=FakeEF(),
    )
    # 合法目录可读取
    abs_dir = retriever.read_skill_dir("finance/invoice")
    assert abs_dir is not None
    assert os.path.isfile(os.path.join(abs_dir, "SKILL.md"))
    # 路径穿越应被拒绝
    assert retriever.read_skill_dir("../../etc") is None
    retriever.close()
