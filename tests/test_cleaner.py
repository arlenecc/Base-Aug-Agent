"""Tests for the text/markdown cleaner — noise removal with format preserved."""
from __future__ import annotations

from agent.rag.cleaner import (
    clean_markdown_document,
    clean_text,
    strip_markdown_noise_lines,
)


def test_md_removes_page_and_header_footer_noise():
    """独立页码 / 页眉页脚（带页码）/ 脚注标记 / 装饰线 应被删整行。"""
    md = """正文第一句。

23

第二章 → 23

第三章 ／ 45

第六章 767

[12]

*

────

正文最后一句。
"""
    out = strip_markdown_noise_lines(md)
    assert "正文第一句。" in out
    assert "正文最后一句。" in out
    for noise in ("\n23\n", "第二章 → 23", "第三章 ／ 45", "第六章 767",
                  "[12]", "────"):
        assert noise not in out


def test_md_preserves_format_headings_tables_code_quotes():
    """标题 / 表格结构 / 代码块 / 引用块 必须保留，代码内小数点不被误伤。"""
    md = """# 第一章 引言

正文。

## 1.1 背景

正文。

| 原序 | …… 2 |
| --- | --- |
| 引论 | …… 11 |

```python
x = 3.14
- 23 -
```

> 引用块保留

圆周率是 3.14，不是 3。
"""
    out = strip_markdown_noise_lines(md)
    # 结构全部保留
    assert "# 第一章 引言" in out
    assert "## 1.1 背景" in out
    assert "> 引用块保留" in out
    # 表格：只清点线页码，保留表格框架
    assert "| 原序 | |" in out
    assert "| --- | --- |" in out
    assert "| 引论 | |" in out
    # 代码块内逐字保留（含小数与页码样文本）
    assert "x = 3.14" in out
    assert "- 23 -" in out
    # 正文小数点不被误删
    assert "圆周率是 3.14，不是 3。" in out


def test_md_toc_dot_leader_stripped_keeps_title():
    """目录点线引导 + 页码：只删点线与页码，保留条目标题。"""
    md = "引论 ………………… 2\n\n第一章 引言 … 5\n"
    out = strip_markdown_noise_lines(md)
    assert "引论" in out
    assert "第一章 引言" in out
    # 点线与页码被清掉
    assert "………" not in out
    assert not any(ln.rstrip().endswith(" 2") or ln.rstrip().endswith(" 5")
                   for ln in out.splitlines() if ln.strip())


def test_md_decorative_separators_removed_but_hr_kept():
    """框线/点线装饰分隔被删；Markdown 水平线（---/***/___）与围栏保留。"""
    md = """正文。

────

…………

---

***

___

正文2。
"""
    out = strip_markdown_noise_lines(md)
    assert "────" not in out
    assert "…………" not in out
    # Markdown 水平线是格式，保留
    assert "\n---\n" in out or out.strip().startswith("---") or "\n---" in out
    assert "***" in out
    assert "___" in out


def test_clean_markdown_document_end_to_end_idempotent():
    """clean_markdown_document 组合流程应清噪、保格式且幂等。"""
    md = """<!-- image -->

# 标题

正文第一句。

第二章 → 23

引论 …… 2

正文第二句。
"""
    once = clean_markdown_document(md)
    twice = clean_markdown_document(once)
    assert once == twice                      # 幂等
    assert "<!-- image -->" not in once        # 注释剥掉
    assert "# 标题" in once                    # 标题保留
    assert "第二章 → 23" not in once           # 页眉噪音删掉
    assert "引论" in once and "……" not in once  # 目录点线清掉
    assert "正文第一句。" in once and "正文第二句。" in once


def test_clean_text_removes_running_header_and_footnote():
    """纯文本路径：页眉（带页码）/ 居中页码 / 脚注标记 也应被清掉。"""
    text = """第一章 引言
正文内容。

第二章 → 23

- 23 -

[12]

第三章 正文
更多内容。
"""
    out = clean_text(text)
    assert "第一章 引言" in out
    assert "第三章 正文" in out
    assert "第二章 → 23" not in out
    assert "[12]" not in out


def test_strip_markdown_noise_lines_no_redos_on_long_leader_runs():
    """长串点线/间隔号（目录点线、装饰分隔、OCR 垃圾）不得触发灾难性回溯。

    早期 _MD_TRAILING_LEADER_RE 用 ``[.…·•]{2,}...\\d{1,4}``，遇「一长串引导符
    且无页码」会立方/指数级回溯（·×4000 直接卡死）；而 strip_markdown_noise_lines
    对每行都跑 .sub，一次同步就被一行拖垮（内存飙+卡住）。用超时守卫锁定线性行为。
    """
    import time

    from agent.rag.cleaner import strip_markdown_noise_lines

    # 50 行 × 4000 个引导符、无页码：最坏回溯输入。
    doc = "\n".join("项目 " + "·" * 4000 for _ in range(50))
    t = time.monotonic()
    out = strip_markdown_noise_lines(doc)
    dt = time.monotonic() - t
    assert dt < 2.0, f"疑似灾难性回溯（ReDoS）：耗时 {dt:.2f}s"
    # 功能性：引导符被清掉、标题文字保留
    assert "项目" in out and "·" not in out


def test_strip_markdown_noise_lines_preserves_decimals():
    """点线清洗不得误伤小数点/版本号（3.14、1.2.3 只有单个点，不触发）。"""
    from agent.rag.cleaner import strip_markdown_noise_lines

    md = "圆周率是 3.14，版本 1.2.3 发布\n\n引论 ………… 2\n"
    out = strip_markdown_noise_lines(md)
    assert "3.14" in out
    assert "1.2.3" in out
    assert "………" not in out  # 目录点线被清
