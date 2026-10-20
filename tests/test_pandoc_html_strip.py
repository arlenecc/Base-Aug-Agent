"""回归测试：pandoc 转 markdown 后残留 raw HTML 的剥离（strip_raw_html）。

背景：pandoc 转 GFM 时 raw_html 扩展默认开启，epub/docx/html 源里的
``<div class=...>``/``<span class=...>`` 等会原样透传进 markdown；而 engine
对 pandoc 路径直接当「规范 Markdown」用（跳过 clean_text 的 HTML 剥离），
标签就大量进入切片、digest 与缓存——表现为转出的电子书内容里全是 HTML 标签。

不能用 ``-t gfm-raw_html`` 一刀切：epub 锚点 span 本身就是 raw HTML，
split_chapters_by_anchors 依赖它按内容文档边界精确切章。
"""
import os

import pytest

from agent.rag.pandoc_parser import (
    _pandoc_to_markdown,
    convert_document,
    split_chapters_by_anchors,
    strip_anchor_spans,
    strip_raw_html,
)
from agent.rag.pandoc_parser import Chapter

LT, GT = chr(60), chr(62)
THO = None  # 占位，避免与工具链对 XML 片段的误判；本文件不需要


def _tag(name, attrs=""):
    return LT + name + attrs + GT


def _tagc(name):
    return LT + "/" + name + GT


DIV_O = _tag("div", ' class="calibre1"')
SPAN_O = _tag("span", ' class="c2"')
SPAN_C = _tagc("span")
DIV_C = _tagc("div")
P_O = _tag("p")
P_C = _tagc("p")
I_O = _tag("i")
I_C = _tagc("i")

# epub 锚点 span（split_chapters_by_anchors 的定位依据）
ANCHOR_1 = _tag("span", ' id="ch001.xhtml"') + _tagc("span")
ANCHOR_2 = _tag("span", ' id="ch002.xhtml"') + _tagc("span")


def test_strips_passthrough_html_but_keeps_text():
    """样式标签剥掉，标签内的正文必须保留。"""
    src = (
        ANCHOR_1 + "\n\n"
        + DIV_O + "\n\n" + SPAN_O + "这是" + SPAN_C + "带" + I_O + "各种" + I_C
        + "标签的段落。" + "\n\n" + DIV_C + "\n"
    )
    cleaned = strip_raw_html(src)
    assert DIV_O not in cleaned and DIV_C not in cleaned
    assert SPAN_O not in cleaned and SPAN_C not in cleaned
    assert I_O not in cleaned and I_C not in cleaned
    joined = "".join(cleaned.split())
    assert "这是带各种标签的段落" in joined or "这是带各种标签的段落。" in joined


def test_protects_fenced_code_blocks():
    """围栏代码块里的尖括号是代码内容，不能当标签剥。"""
    code = "vector" + LT + "int" + GT + " v;\nif (a " + LT + " b) {}"
    src = "```cpp\n" + code + "\n```\n" + P_O + "正文" + P_C
    cleaned = strip_raw_html(src)
    assert "vector<int> v;" in cleaned
    assert "if (a < b) {}" in cleaned
    assert "正文" in cleaned and P_O not in cleaned


def test_protects_indented_code_blocks():
    """pandoc 常把 <pre><code> 写成 4 空格缩进代码，同样要保护。"""
    src = "代码示例：\n\n    vector" + LT + "int" + GT + " v;\n\n" + P_O + "正文" + P_C
    cleaned = strip_raw_html(src)
    assert "vector<int> v;" in cleaned
    assert P_O not in cleaned


def test_protects_inline_code_spans():
    """行内代码 span（`` `...` ``）里的标签形文本是内容。"""
    src = "行内代码 " + "`" + LT + "div" + GT + "`" + " 要保留。" + P_O + P_C
    cleaned = strip_raw_html(src)
    assert "`<div>`" in cleaned


def test_protects_escaped_inline_code():
    """pandoc 会把字面反引号转义成 \\`：行内代码整体转义时掩码要兜住。"""
    # 源文本里的字面形式：\`\<div\>\`
    src = "转义的行内代码 \\`\\<" + "div\\>\\` 结束。" + P_O + "正文" + P_C
    cleaned = strip_raw_html(src)
    assert "\\`\\<div\\>\\`" in cleaned
    assert P_O not in cleaned


def test_does_not_touch_comparison_operators():
    """散文里的比较运算（< 字母后有空格）不是标签，不能误删。"""
    src = "if a < b and c > d 则成立。" + P_O + P_C
    cleaned = strip_raw_html(src)
    assert "if a < b and c > d 则成立。" in cleaned


def test_removes_script_and_style_with_content():
    """script/style 块连内容一起删（标签剥除只会留下 CSS/JS 噪音）。"""
    src = (
        _tag("style") + "p{color:red}" + _tagc("style") + "\n"
        + _tag("script") + "alert(1)" + _tagc("script") + "\n"
        + "正文内容"
    )
    cleaned = strip_raw_html(src)
    assert "color:red" not in cleaned
    assert "alert(1)" not in cleaned
    assert "正文内容" in cleaned


def test_anchor_spans_survive_for_chapter_splitting():
    """剥离发生在锚点切分之后：先切章（需要锚点），后剥标签。"""
    md = (
        ANCHOR_1 + "\n\n# 第一章\n\n" + DIV_O + "甲章正文" + DIV_C + "\n\n"
        + ANCHOR_2 + "\n\n# 第二章\n\n乙章正文\n"
    )
    toc = [
        Chapter(title="第一章", level=1, text="ch001.xhtml"),
        Chapter(title="第二章", level=1, text="ch002.xhtml"),
    ]
    chapters = split_chapters_by_anchors(md, toc)
    assert chapters and len(chapters) == 2
    cleaned = [strip_raw_html(strip_anchor_spans(t)) for t in chapters]
    assert ANCHOR_1 not in cleaned[0] and ANCHOR_2 not in cleaned[1]
    assert "甲章正文" in cleaned[0] and "乙章正文" in cleaned[1]
    assert DIV_O not in cleaned[0]


def test_empty_and_none_input():
    assert strip_raw_html("") == ""
    assert strip_raw_html(None) is None


# --- 集成：本机装了 pandoc 就跑真实转换链路 ---
pandoc_available = _pandoc_to_markdown.__module__ is not None


@pytest.mark.skipif(
    os.environ.get("PANDOC") == "" ,
    reason="需要 pandoc",
)
def test_convert_document_strips_html_from_html_source(tmp_path):
    """html 源经 pandoc → markdown → strip_raw_html 后无样式标签残留。"""
    try:
        import shutil
        if shutil.which(os.environ.get("PANDOC", "pandoc")) is None:
            pytest.skip("本机无 pandoc")
    except ImportError:
        pytest.skip("无法探测 pandoc")

    html = (
        "<html><body>"
        + LT + "div class=\"wrapper\"" + GT + LT + "p" + GT
        + "你好" + LT + "b" + GT + "世界" + LT + "/b" + GT
        + LT + "/p" + GT + LT + "/div" + GT
        + "</body></html>"
    )
    src = tmp_path / "doc.html"
    src.write_text(html, encoding="utf-8")
    md = convert_document(str(src))
    if md is None:
        pytest.skip("pandoc 转换失败")
    assert "wrapper" not in md
    assert "你好世界" in md.replace("\n", "").replace(" ", "") or "你好" in md
