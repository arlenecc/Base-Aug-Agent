"""Tests for document outline extraction + digest (Meta-context RAG)."""
from __future__ import annotations

from agent.rag.document_outline import (
    build_digest,
    build_toc,
    extract_chapters,
    find_chapters_by_title,
    summarize_chapters,
)


MARKDOWN = """# 第一章 引言

深度学习是机器学习的重要分支。本章介绍基本概念。

## 1.1 背景

神经网络的历史可以追溯到上世纪。感知机是早期的神经网络模型。

## 1.2 目标

本书目标是帮助读者系统掌握深度学习。

# 第二章 卷积神经网络

卷积神经网络在图像识别领域表现优异。LeNet 是最早的卷积网络之一。

## 2.1 卷积层

卷积层通过卷积核提取局部特征。常用参数包括卷积核大小和步长。
"""


def test_extract_chapters_heading_structure():
    chapters = extract_chapters(MARKDOWN)
    titles = [c.title for c in chapters]
    assert "第一章 引言" in titles
    assert "1.1 背景" in titles
    assert "第二章 卷积神经网络" in titles
    levels = {c.title: c.level for c in chapters}
    assert levels["第一章 引言"] == 1
    assert levels["1.1 背景"] == 2
    assert levels["第二章 卷积神经网络"] == 1


def test_extract_chapters_prefix():
    md = "这是前言内容。\n\n# 第一章\n\n正文。"
    chapters = extract_chapters(md)
    assert chapters[0].title == "前言"
    assert chapters[0].level == 0


def test_build_toc():
    chapters = extract_chapters(MARKDOWN)
    toc = build_toc(chapters)
    assert "第一章 引言" in toc
    assert "1.1 背景" in toc


def test_summarize_fallback_truncates():
    chapters = extract_chapters(MARKDOWN)
    summarize_chapters(chapters, summarizer=None, max_summary_chars=50)
    for c in chapters:
        if c.text:
            assert c.summary  # 非空
            assert len(c.summary) <= 51  # 50 + 省略号


def test_summarize_with_llm():
    chapters = extract_chapters(MARKDOWN)

    def fake_summarizer(title, text):
        return f"摘要:{title}"

    summarize_chapters(chapters, summarizer=fake_summarizer, max_summary_chars=50)
    for c in chapters:
        if c.text:
            assert c.summary.startswith("摘要:")


def test_build_digest_structure():
    digest, chapters = build_digest(MARKDOWN, summarizer=None, max_summary_chars=50)
    assert "# 目录" in digest
    assert "# 章节摘要" in digest
    assert "第一章 引言" in digest


def test_find_chapters_by_title():
    chapters = extract_chapters(MARKDOWN)
    hits = find_chapters_by_title(chapters, "卷积")
    # "卷积" 出现在 "第二章 卷积神经网络" 和 "2.1 卷积层" 两个标题中
    assert len(hits) == 2
    assert {c.title for c in hits} == {"第二章 卷积神经网络", "2.1 卷积层"}


def test_empty_markdown():
    assert extract_chapters("") == []
    digest, chapters = build_digest("", summarizer=None)
    assert chapters == []
