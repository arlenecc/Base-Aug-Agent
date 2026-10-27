"""文件转 Markdown 转换工具（convert_tool_window）纯逻辑单元测试。

只测不依赖 Qt 实例化的部分：文件收集、单文件转换路由、输出命名、
解析器 configure 接线。UI 与线程部分由人工验收。
"""

import os

import pytest

from agent.ui.convert_tool_window import (
    CONVERT_EXTENSIONS,
    apply_parser_configure,
    collect_files,
    convert_file_to_markdown,
    unique_output_path,
)


# ---------------------------------------------------------------------------
# collect_files
# ---------------------------------------------------------------------------

def test_collect_files_single_file(tmp_path):
    fp = tmp_path / "a.pdf"
    fp.write_text("x")
    assert collect_files(str(fp)) == [str(fp.resolve())]


def test_collect_files_rejects_missing(tmp_path):
    with pytest.raises(ValueError):
        collect_files(str(tmp_path / "nope"))


def test_collect_files_rejects_empty():
    with pytest.raises(ValueError):
        collect_files("")


def test_collect_files_directory_whitelist(tmp_path):
    (tmp_path / "b.pdf").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("x")
    (sub / ".hidden.md").write_text("x")      # 隐藏文件跳过
    (sub / "d.xyz").write_text("x")           # 白名单外跳过
    got = [os.path.basename(p) for p in collect_files(str(tmp_path))]
    assert got == ["b.pdf", "c.txt"]


def test_convert_extensions_cover_user_formats():
    for ext in (".pdf", ".epub", ".docx", ".doc", ".xlsx", ".xls", ".pptx",
                ".ppt", ".html", ".jpg", ".jpeg", ".txt"):
        assert ext in CONVERT_EXTENSIONS, ext


# ---------------------------------------------------------------------------
# convert_file_to_markdown（纯文本路由：clean_text + 标题还原）
# ---------------------------------------------------------------------------

def test_convert_txt_file(tmp_path):
    fp = tmp_path / "note.txt"
    fp.write_text("第一章 概述\n\n这是正文内容。\n", encoding="utf-8")
    md = convert_file_to_markdown(str(fp))
    assert md.strip()
    assert "这是正文内容。" in md


def test_convert_md_file_keeps_content(tmp_path):
    fp = tmp_path / "doc.md"
    fp.write_text("# 标题\n\n正文段落。\n", encoding="utf-8")
    md = convert_file_to_markdown(str(fp))
    assert "正文段落。" in md


def test_convert_empty_file_raises(tmp_path):
    fp = tmp_path / "empty.txt"
    fp.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        convert_file_to_markdown(str(fp))


# ---------------------------------------------------------------------------
# unique_output_path
# ---------------------------------------------------------------------------

def test_unique_output_path_no_collision(tmp_path):
    src = tmp_path / "report.pdf"
    got = unique_output_path(str(tmp_path), str(src))
    assert got == os.path.join(str(tmp_path), "report.md")


def test_unique_output_path_collision_appends_counter(tmp_path):
    (tmp_path / "report.md").write_text("x")
    (tmp_path / "report (2).md").write_text("x")
    src = tmp_path / "report.pdf"
    got = unique_output_path(str(tmp_path), str(src))
    assert got == os.path.join(str(tmp_path), "report (3).md")


def test_unique_output_path_stemless_name(tmp_path):
    # 无扩展名/裸名文件也要有合理的默认 stem
    src = str(tmp_path / "data")
    got = unique_output_path(str(tmp_path), src)
    assert got == os.path.join(str(tmp_path), "data.md")


# ---------------------------------------------------------------------------
# apply_parser_configure（不构造引擎，只验证归一化与容错）
# ---------------------------------------------------------------------------

class _Cfg:
    """最小配置桩：getattr 默认值路径 + 显式字段路径都要走通。"""


def test_apply_parser_configure_normalizes_engine():
    cfg = _Cfg()
    cfg.rag_pdf_engine = "marker"
    assert apply_parser_configure(cfg) == "marker"


def test_apply_parser_configure_falls_back_to_docling():
    cfg = _Cfg()
    cfg.rag_pdf_engine = "nonsense"
    assert apply_parser_configure(cfg) == "docling"


def test_apply_parser_configure_defaults_when_attrs_missing():
    cfg = _Cfg()  # 一个字段都没有：全部 getattr 默认值，不应抛异常
    assert apply_parser_configure(cfg) == "docling"


# ---------------------------------------------------------------------------
# 冒烟：窗口构造 + worker 同步运行（Qt 但不依赖真实主窗口）
# ---------------------------------------------------------------------------

def test_window_smoke(qtbot):
    from agent.ui.convert_tool_window import ConvertToolWindow

    class _Main:  # 构造阶段只存引用，不会用到主窗口方法
        pass

    win = ConvertToolWindow(_Main())
    qtbot.addWidget(win)
    assert win.start_btn.text() == "开始转换"
    assert not win.cancel_btn.isEnabled()
    win._set_running(True)
    assert win.start_btn.isEnabled() is False
    assert win.cancel_btn.isEnabled()


def test_worker_converts_and_reports(tmp_path):
    from agent.ui.convert_tool_window import ConvertWorker

    src = tmp_path / "note.txt"
    src.write_text("第一章 概述\n\n正文内容。\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    events = []
    w = ConvertWorker([str(src)], str(out), "docling")
    w.file_done.connect(lambda s, o, ok, n: events.append(("file", o, ok)))
    w.all_done.connect(lambda ok, fail, c: events.append(("all", ok, fail, c)))
    w.run()  # 直接同步执行，规避线程时序
    md = out / "note.md"
    assert md.exists()
    assert "正文内容。" in md.read_text(encoding="utf-8")
    assert ("file", str(md), True) in events
    assert ("all", 1, 0, False) in events


def test_worker_counts_failures(tmp_path):
    from agent.ui.convert_tool_window import ConvertWorker

    events = []
    w = ConvertWorker([str(tmp_path / "missing.txt")], str(tmp_path), "docling")
    w.file_done.connect(lambda s, o, ok, n: events.append(("file", ok, n)))
    w.all_done.connect(lambda ok, fail, c: events.append(("all", ok, fail)))
    w.run()
    assert ("file", False, events[0][2]) in events  # 失败且带原因
    assert ("all", 0, 1) in events
