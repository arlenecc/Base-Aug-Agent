"""Tests for document outline extraction + digest (Meta-context RAG)."""
from __future__ import annotations

from agent.rag.document_outline import (
    build_digest,
    build_toc,
    convert_plain_text_to_markdown,
    extract_chapters,
    extract_chapters_plain,
    find_chapters_by_title,
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


def test_build_digest_is_toc_only():
    """digest 只含目录结构（标题层级），不再生成逐章摘要。"""
    digest, chapters = build_digest(MARKDOWN)
    assert "# 目录" in digest
    assert "第一章 引言" in digest
    assert "1.1 背景" in digest
    # 不再包含「章节摘要」段
    assert "# 章节摘要" not in digest
    assert "（无内容）" not in digest


def test_find_chapters_by_title():
    chapters = extract_chapters(MARKDOWN)
    hits = find_chapters_by_title(chapters, "卷积")
    # "卷积" 出现在 "第二章 卷积神经网络" 和 "2.1 卷积层" 两个标题中
    assert len(hits) == 2
    assert {c.title for c in hits} == {"第二章 卷积神经网络", "2.1 卷积层"}


def test_empty_markdown():
    assert extract_chapters("") == []
    digest, chapters = build_digest("")
    assert chapters == []


# ---------------------------------------------------------------------------
# 纯文本（无 Markdown 标题）目录提取
# ---------------------------------------------------------------------------

PLAIN_TOC_PDF = """人生的活法
（日）本多静六著

目 录
本多静六语录
作者简介
自序
1卷 我的财产告白
我的财产告白
征讨贫困和本多式储蓄法
2卷 我的人生活法
我的健康长寿法
如何健康长寿
3卷 我的人生计划
如何制定人生计划
人生为什么需要计划？

他，设计了明治神宫的森林和日比谷公园，被
称为“日本的公园之父”。
"""


def test_plain_toc_block_extraction():
    """纯文本「目录」块应提取出标题列表。"""
    chapters = extract_chapters_plain(PLAIN_TOC_PDF)
    titles = {c.title for c in chapters}
    assert "1卷 我的财产告白" in titles
    assert "2卷 我的人生活法" in titles
    assert "3卷 我的人生计划" in titles
    assert "自序" in titles
    # 目录块应在正文（以句号结尾的句子）处终止，不混入正文。
    assert not any("明治神宫" in t for t in titles)


def test_plain_cn_chapter_headings():
    """无「目录」标记、但正文含「第X章」标题的纯文本，应被识别。"""
    text = """内容提要
这是一段前言介绍。

第一章 人际关系的构成
假若黄金周这样度过，是不是很美妙。

第二章 研究方法
本章介绍研究亲密关系的方法。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第一章 人际关系的构成" in titles
    assert "第二章 研究方法" in titles


def test_plain_epub_link_toc():
    """EPUB 的 [TITLE](#anchor) 目录格式应被清理为纯标题。"""
    text = """TABLE OF CONTENTS

[INTRODUCTION TO GENERATIVE AI](#aid_45)

[HOW GENERATIVE AI WORKS](#aid_33)

INTRODUCTION TO GENERATIVE AI

Generative AI is a revolutionary technique.
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "INTRODUCTION TO GENERATIVE AI" in titles
    assert "HOW GENERATIVE AI WORKS" in titles
    # 不应残留 (#anchor) 后缀。
    assert not any("(#" in t for t in titles)


def test_build_digest_falls_back_to_plain():
    """build_digest 在无 Markdown 标题时应回退到纯文本提取。"""
    digest, chapters = build_digest(PLAIN_TOC_PDF)
    assert "1卷 我的财产告白" in digest
    assert "# 目录" in digest
    assert len(chapters) > 0


# ---------------------------------------------------------------------------
# PDF 页眉 / 页码噪声过滤
# ---------------------------------------------------------------------------

def test_plain_pdf_header_pageno_noise():
    """PDF 页眉「章节名 + 分隔符 + 页码」应被过滤，不当作标题。"""
    text = """第一章 引言
正文内容。

第二章 → 23
本章正文。

第三章 ／ 45
第三章正文。

第四章
第四章正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第一章 引言" in titles
    assert "第四章" in titles
    # 页眉噪声应被丢弃
    assert "第二章 → 23" not in titles
    assert "第三章 ／ 45" not in titles


def test_plain_pdf_header_chapter_pageno_noise():
    """PDF 页眉「章节名 + 空格 + 页码」（无分隔符）应被过滤。"""
    text = """第五章
正文。

第六章 767
正文。

第七章
正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第五章" in titles
    assert "第七章" in titles
    assert "第六章 767" not in titles


# ---------------------------------------------------------------------------
# 纯文本 → Markdown（注入 # 标题）
# ---------------------------------------------------------------------------

def test_convert_plain_cn_chapters_to_markdown():
    """正文含「第X章」标题的纯文本，应被注入成 # 标题，且正文保留。"""
    text = """内容提要
这是一段前言介绍。

第一章 人际关系的构成
假若黄金周这样度过，是不是很美妙。

第二章 研究方法
本章介绍研究亲密关系的方法。
"""
    md, chapters = convert_plain_text_to_markdown(text)
    assert "# 第一章 人际关系的构成" in md
    assert "# 第二章 研究方法" in md
    # 正文段落应保留
    assert "假若黄金周这样度过" in md
    assert "本章介绍研究亲密关系的方法。" in md
    # 标题层级
    assert {c.title: c.level for c in chapters} == {
        "第一章 人际关系的构成": 1,
        "第二章 研究方法": 1,
    }


def test_convert_plain_toc_block_dropped_when_body_has_headings():
    """正文自带标题时，前导「目录」块应被丢弃，不重复生成标题。"""
    text = """目 录
第一章 引言
第二章 方法

第一章 引言
引言正文。

第二章 方法
方法正文。
"""
    md, chapters = convert_plain_text_to_markdown(text)
    # 「目录」标记与 TOC 块条目都应在 markdown 中消失
    assert "目 录" not in md
    assert "# 第一章 引言" in md
    assert "# 第二章 方法" in md
    # 不应出现重复标题（TOC 块里的那一份已被丢弃）
    assert md.count("# 第一章 引言") == 1


def test_convert_plain_toc_block_becomes_headings():
    """正文无标题、只有「目录」块时，目录条目应转为 # 标题。"""
    text = """人生的活法

目 录
1卷 我的财产告白
2卷 我的人生活法

他，设计了明治神宫的森林。
"""
    md, chapters = convert_plain_text_to_markdown(text)
    assert "# 1卷 我的财产告白" in md
    assert "# 2卷 我的人生活法" in md
    # 「目录」标记本身被丢弃
    assert "目 录" not in md
    assert {c.title for c in chapters} == {
        "1卷 我的财产告白",
        "2卷 我的人生活法",
    }


def test_convert_plain_no_headings_is_noop():
    """没有可识别标题的普通纯文本，应保持原样（不误造标题）。"""
    text = """这是一段普通叙述文字，没有任何章节标题。
它包含多个句子。第二句话也平淡无奇。

第三段也是一样的内容。
"""
    md, chapters = convert_plain_text_to_markdown(text)
    assert chapters == []
    assert "#" not in md
    assert "这是一段普通叙述文字" in md


def test_plain_year_prefix_noise():
    """年份开头的正文行（如「1947）。…」「2007 年12月」）不应是标题。"""
    text = """第一章 引言
正文。

1947）。《薄伽梵歌》是其中的第二十三至第四十章。
2007 年12月
第二章 正文
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第一章 引言" in titles
    assert not any(t.startswith("1947") for t in titles)
    assert not any(t.startswith("2007") for t in titles)


def test_plain_toc_marker_not_title():
    """「目 录」目录标记本身不应出现在标题列表里。"""
    text = """《神之所欲》
翻 译 者 序
目 录
~1~ 不可置信之书
~2~ 史上最重要的问题
正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "~1~ 不可置信之书" in titles
    assert "~2~ 史上最重要的问题" in titles
    assert "目 录" not in titles


# ---------------------------------------------------------------------------
# Markdown 标题净化（docling 把正文句子误判为标题）
# ---------------------------------------------------------------------------

def test_md_drops_sentence_headings_but_keeps_anchors():
    """docling 常把正文句子标成 ## 标题；句子应丢弃，真正的章（锚点）应保留。"""
    md = """# 作者的话

## 但是，完全不是这么回事。

# 第一章 引言

## 便说：「若瑟的回答是正确的，因为他不知道。」

## 第二章 方法

正文。
"""
    chapters = extract_chapters(md)
    titles = {c.title for c in chapters}
    assert "第一章 引言" in titles
    assert "第二章 方法" in titles
    # 句子式误判标题应被丢弃
    assert "但是，完全不是这么回事。" not in titles
    assert "便说：「若瑟的回答是正确的，因为他不知道。」" not in titles


def test_md_strips_front_matter_title_page():
    """书名页/版权页噪声（在首个真实章节之前）应被丢弃，真实章节保留。"""
    md = """## 人生的亲证

## Rabindranath Tagore SADHANA

## 目录

## 作者序言

## 一 个人和宇宙的关系

正文。
"""
    chapters = extract_chapters(md)
    titles = {c.title for c in chapters}
    # 书名页噪声被丢弃
    assert "人生的亲证" not in titles
    assert "Rabindranath Tagore SADHANA" not in titles
    assert "目录" not in titles
    # 真实章节保留
    assert "作者序言" in titles
    assert "一 个人和宇宙的关系" in titles


def test_md_drops_colophon_back_matter():
    """版权页 / 勒口 / 地址等后 matter 噪声标题应被丢弃。"""
    md = """# 第一章 引言

正文。

## 定價220元

## 美術設計／朱陳毅

## 104 台北市民生東路二段141號

## 第二章 方法

正文。
"""
    chapters = extract_chapters(md)
    titles = {c.title for c in chapters}
    assert "第一章 引言" in titles
    assert "第二章 方法" in titles
    assert "定價220元" not in titles
    assert "美術設計／朱陳毅" not in titles
    assert "104 台北市民生東路二段141號" not in titles


def test_md_merges_ocr_variant_running_header():
    """同一章节被 OCR 识别成多个错字变体时（如「安顿心灵的七个步骤」），应合并为一条。"""
    md = """# 第二部

## 安頓心靈七個步驟

## 安頓心靈的七個步驟

## 安頓心靈钓七個步驟

## 安頓心蟹七個步驟

# 第三部

正文。
"""
    chapters = extract_chapters(md)
    titles = [c.title for c in chapters]
    variants = [t for t in titles if "安頓心靈" in t and "步驟" in t]
    assert len(variants) == 1, variants


def test_md_keeps_consecutive_numbered_chapters():
    """相邻编号章节（仅差编号一字）不应被 OCR 近重合并误并成一条。"""
    md = """# 第一百章 總論

正文。

# 第一百零一章 分論

正文。
"""
    chapters = extract_chapters(md)
    titles = [c.title for c in chapters]
    assert "第一百章 總論" in titles
    assert "第一百零一章 分論" in titles


# ---------------------------------------------------------------------------
# 纯文本路径的噪声抑制（句读行 / OCR 乱码 / 编号习题句 / 加粗目录标记 / 标题去页码）
# ---------------------------------------------------------------------------

def test_plain_bold_toc_marker():
    """加粗/变体目录标记（**总目录**）应被识别，提取其后的条目。"""
    text = """某套装总集

**总目录**

傲慢与偏见

呼啸山庄

基督山伯爵

正文开始了，这是一段很长很长的正文内容用来终止目录块的读取，需要超过八十个字符才会触发终止条件。
"""
    chapters = extract_chapters_plain(text)
    titles = [c.title for c in chapters]
    assert "傲慢与偏见" in titles
    assert "呼啸山庄" in titles
    assert "基督山伯爵" in titles
    assert "某套装总集" not in titles  # 目录标记之前的内容不收入


def test_plain_rejects_numbered_sentence_item():
    """单级编号的长句（正文列表项/习题，非章节）不应被当作标题。"""
    text = """2. 不是每一项旨在保留一个早已存在帝国的外交政策都是帝国主

第1章 国际政治现实主义理论

正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert not any("不是每一项" in t for t in titles)
    assert "第1章 国际政治现实主义理论" in titles


def test_plain_rejects_ocr_garbage_line():
    """有效字符占比过低的 OCR 乱码行（大量 ①•⋯）不应被当作标题。"""
    text = """第一章 引言

1. ①体••11.①西⋯•②秦⋯11. ①游戏-软件设计 N. ①TP311.5

第二章 方法

正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert "第一章 引言" in titles
    assert "第二章 方法" in titles
    assert not any("①" in t for t in titles)


def test_plain_rejects_sentence_with_midline_punct():
    """行中含句读（。，；！）的合并正文行不应被当作标题。"""
    text = """第三章 数据与统计，这是一段被并入标题行的正文说明。

第四章 抽样与抽样分布

正文。
"""
    chapters = extract_chapters_plain(text)
    titles = {c.title for c in chapters}
    assert not any("这是一段被并入" in t for t in titles)
    assert "第四章 抽样与抽样分布" in titles


def test_normalize_title_strips_trailing_page_and_quote():
    """标题行尾部的页码与并入的引文应被清掉。"""
    from agent.rag.document_outline import _normalize_title
    assert _normalize_title("第2章 优泰|：61") == "第2章 优泰"
    assert _normalize_title("第9章 界面 ｜ 223 “很不幸，看来…”") == "第9章 界面"
    assert _normalize_title("第一章 引言") == "第一章 引言"


def test_plain_keeps_chapter_merged_with_quote():
    """真实「第X章 标题」即使并入了含句读的引文，也应保留（引文被剥掉）。"""
    text = """第3章 重商主义、重农主义及古典经济思想的其他先驱 “值得注意的是，发明家们并不具备被称为‘科学态度’的客观性。”

正文段落。

第5章 李嘉图与马尔萨斯 “李嘉图在思维上的吸引力，过去和现在都体现为他运用英雄式抽象概念的非凡才能。”

正文。
"""
    chapters = extract_chapters_plain(text)
    titles = [c.title for c in chapters]
    assert any(t.startswith("第3章") and "先驱" in t for t in titles)
    assert any(t.startswith("第5章") and "李嘉图" in t for t in titles)
    # 并入的引文应被剥掉
    assert not any("值得注意的是" in t for t in titles)
    assert not any("英雄式抽象" in t for t in titles)
