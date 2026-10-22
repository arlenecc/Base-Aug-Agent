"""Full worker -> signal -> UI flow test using pytest-qt's event loop.

Drives a real agent turn through the QThread worker with a scripted LLM and
verifies streaming content lands in the chat view.

We call MainWindow._on_send() directly and poll the `busy` flag with
qtbot.waitUntil: this is robust under the offscreen platform (where synthetic
mouse clicks may not be delivered) and avoids nested waitSignal subtleties.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from agent.agent import Agent
from agent.config import AgentConfig
from agent.llm_client import StreamEvent


def _evt(t, content="", tool_calls=None, usage=None):
    return StreamEvent(type=t, content=content, tool_calls=tool_calls or [], usage=usage or {})


class _ScriptedLLM:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    def chat_stream(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools})
        return iter(self._scripts.pop(0))


def _make_window(tmp_path, monkeypatch):
    import agent.ui.main_window as mw

    monkeypatch.setattr(mw, "CONFIG_PATH", str(tmp_path / "cfg.json"))
    win = mw.MainWindow()
    # Isolate the workspace + skill history so the agent never touches the
    # user's real ~/base-agent-workspace (which could trigger a skill-suggestion
    # modal mid-test and hang the offscreen event loop).
    win.workspace_edit.setText(str(tmp_path / "ws"))
    win._on_apply_config()
    return win


def _install_stub_warmup_worker(win, monkeypatch):
    """把常驻对话线程换成桩：只记录预热请求，不起真线程。"""
    class _StubWorker:
        def __init__(self):
            self.requests = 0

        def request_warmup(self):
            self.requests += 1

    stub = _StubWorker()
    monkeypatch.setattr(win, "_ensure_agent_worker", lambda: stub)
    return stub


def _flush_reason(win):
    """思考区走 100ms 攒批写入（见 MainWindow._on_reasoning）。

    UI 线程上的同步断言读不到缓冲区里的内容，必须先把攒批落地。
    """
    win._flush_reason_buffer()


# ---------------------------------------------------------------------------
# 配置窗口两分区 + 本地模型加载/卸载（#140）
# ---------------------------------------------------------------------------

def _section_titles(win):
    from PyQt6.QtWidgets import QGroupBox

    return [w.title() for w in win.config_window.findChildren(QGroupBox)]


def test_config_window_has_two_sections(qtbot, tmp_path, monkeypatch):
    """配置窗口分成「Agent 模型配置」与「知识同步配置」两个区。"""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    titles = _section_titles(win)
    assert "Agent 模型配置" in titles
    assert "知识同步配置" in titles

    # 默认本地模式：本地面板可见、接口面板隐藏。
    assert win.llm_mode_local_radio.isChecked()
    assert not win.llm_mode_openai_radio.isChecked()
    assert win.local_mode_panel.isVisibleTo(win.config_window)
    assert not win.openai_mode_panel.isVisibleTo(win.config_window)

    # 切到接口模式：面板互换，Base URL / API Key 可用。
    win.llm_mode_openai_radio.setChecked(True)
    assert not win.local_mode_panel.isVisibleTo(win.config_window)
    assert win.openai_mode_panel.isVisibleTo(win.config_window)
    win.llm_mode_local_radio.setChecked(True)
    assert win.local_mode_panel.isVisibleTo(win.config_window)


def test_config_window_default_models(qtbot, tmp_path, monkeypatch):
    """主/辅模型默认填 #140 指定的两个本地模型，类型默认 MLX-LM。"""
    from agent.ui.main_window import (
        DEFAULT_AUX_MODEL_PATH, DEFAULT_MAIN_MODEL_PATH, _combo_backend_value,
    )

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    assert _combo_backend_value(win.llm_backend_combo) == "mlx_lm"
    assert _combo_backend_value(win.aux_llm_backend_combo) == "mlx_lm"
    assert win.llm_model_path_edit.text() == DEFAULT_MAIN_MODEL_PATH
    assert win.aux_llm_model_path_edit.text() == DEFAULT_AUX_MODEL_PATH


def test_backend_combo_shows_gguf_but_stores_llama_cpp(qtbot, tmp_path, monkeypatch):
    """下拉框显示 GGUF，落到 config 的仍是 llama_cpp（#140 仅改名）。"""
    from agent.ui.main_window import _combo_backend_value

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    labels = [win.llm_backend_combo.itemText(i) for i in range(win.llm_backend_combo.count())]
    assert "GGUF" in labels
    assert "llama_cpp" not in labels
    assert labels[:2] == ["MLX-LM", "MLX-VLM"]

    win.llm_backend_combo.setCurrentIndex(labels.index("GGUF"))
    assert _combo_backend_value(win.llm_backend_combo) == "llama_cpp"

    emb_labels = [
        win.embedding_backend_combo.itemText(i)
        for i in range(win.embedding_backend_combo.count())
    ]
    assert "GGUF" in emb_labels
    win.embedding_backend_combo.setCurrentIndex(emb_labels.index("GGUF"))
    assert _combo_backend_value(win.embedding_backend_combo) == "llama_cpp"


def test_mode_and_sampling_round_trip(qtbot, tmp_path, monkeypatch):
    """模式、主/辅采样参数都能正确读写 config。"""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    win.llm_mode_openai_radio.setChecked(True)
    win.base_url_edit.setText("https://api.example.com/v1")
    win.api_key_edit.setText("k")
    win.temp_edit.setText("0.3")
    win.max_tokens_edit.setText("4096")
    win.aux_temp_edit.setText("0.11")
    win.aux_max_tokens_edit.setText("777")
    win.aux_top_p_edit.setText("0.42")
    win._on_apply_config()

    assert win.config.llm_mode == "openai"
    assert win.config.llm_backend == "openai"
    assert win.config.temperature == 0.3
    assert win.config.max_tokens == 4096
    assert win.config.aux_temperature == 0.11
    assert win.config.aux_max_tokens == 777
    assert win.config.aux_top_p == 0.42

    # 切回本地模式 + 换 GGUF 类型：后端必须是 llama_cpp 而非 openai。
    win.llm_mode_local_radio.setChecked(True)
    labels = [win.llm_backend_combo.itemText(i) for i in range(win.llm_backend_combo.count())]
    win.llm_backend_combo.setCurrentIndex(labels.index("GGUF"))
    win._on_apply_config()
    assert win.config.llm_mode == "local"
    assert win.config.llm_backend == "llama_cpp"


class _FakeLocalBackend:
    """最小本地后端：有 _ensure_loaded / close，权重用一个属性表示。"""

    def __init__(self, config, marker=""):
        self.config = config
        self.model_path = marker or "fake-model"
        self._model_obj = None
        self._closed = False
        self.ensure_calls = 0

    def _ensure_loaded(self):
        self.ensure_calls += 1
        self._model_obj = object()

    def close(self):
        self._closed = True
        self._model_obj = None


def _install_fake_model_factory(monkeypatch, mw, sink):
    def _make(config):
        backend = _FakeLocalBackend(config)
        sink.append(backend)
        return backend

    monkeypatch.setattr(mw, "create_llm_backend", _make)
    monkeypatch.setattr(mw, "create_aux_llm_backend", _make)
    return _make


def test_switching_mode_radio_switches_active_backend(qtbot, tmp_path, monkeypatch):
    """切模式 radio 必须让对话框实际用的后端跟着切（#140 的 bug：切到本地模式
    仍走 OpenAI 接口，因为 radio 只换了面板显示，没动 agent 的后端）。"""
    import agent.ui.main_window as mw

    created = []
    def _fake_create(config):
        tag = getattr(config, "llm_backend", "openai")

        class _B:
            def __init__(self, t, c):
                self.tag = t
                self.config = c
                self._model_obj = None
                self._closed = False
            def _ensure_loaded(self):
                self._model_obj = object()
            def close(self):
                self._closed = True
                self._model_obj = None
        b = _B(tag, config)
        created.append(tag)
        return b
    monkeypatch.setattr(mw, "create_llm_backend", _fake_create)
    monkeypatch.setattr(mw, "create_aux_llm_backend", _fake_create)

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    # 默认本地模式：对话框用的是本地后端（懒加载）。
    assert win.config.llm_mode == "local"
    assert win._agent.llm.tag == "mlx_lm"

    # 切到接口模式：agent 必须切回 LLMClient（openai 标记）。
    win.llm_mode_openai_radio.setChecked(True)
    assert win.config.llm_mode == "openai"
    assert win._agent.llm.tag == "openai"

    # 再切回本地模式：agent 必须切回本地后端，而不是残留接口客户端。
    win.llm_mode_local_radio.setChecked(True)
    assert win.config.llm_mode == "local"
    assert win._agent.llm.tag == "mlx_lm"


def test_load_and_unload_local_model(qtbot, tmp_path, monkeypatch):
    """加载模型把权重读进内存并交给 agent；卸载后 agent 换回新后端。"""
    import agent.ui.main_window as mw

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    created = []
    _install_fake_model_factory(monkeypatch, mw, created)
    # 关掉启动自动加载的干扰：这里手动驱动。
    monkeypatch.setattr(win, "_autoload_default_models", lambda: None)

    assert not win._is_model_loaded("main")

    win._on_load_model("main")
    qtbot.waitUntil(lambda: not win._model_busy.get("main", False), timeout=5000)

    assert len(created) == 1
    loaded = created[0]
    assert loaded.ensure_calls == 1          # 权重真的被加载了
    assert win._is_model_loaded("main")
    assert win._llm is loaded
    assert win._agent.llm is loaded           # agent 直接指向热后端
    assert win.llm_load_btn.text() == "重新加载"
    # 加载成功：按钮背景变绿，代表模型已就绪可用（用户反馈）。
    assert "#22c55e" in win.llm_load_btn.styleSheet()

    win._on_unload_model("main")
    assert not win._is_model_loaded("main")
    assert loaded._closed                     # 旧的被关掉释放内存
    assert win._agent.llm is not loaded       # agent 拿到了全新的后端
    assert win.llm_load_btn.text() == "加载模型"


def test_warmup_status_shown_in_status_bar(qtbot, tmp_path, monkeypatch):
    """预热期间状态栏显示「正在预热模型…」，完成后显示「就绪」。

    用户反馈：预热结束的提示只写在独立日志窗口里，主界面看不出什么时候
    才真正可用，所以状态栏必须承担这个「就绪」信号。
    """
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    class _StubWorker:
        def __init__(self):
            self.requests = 0

        def request_warmup(self):
            self.requests += 1

    stub = _StubWorker()
    monkeypatch.setattr(win, "_ensure_agent_worker", lambda: stub)
    win._busy = False
    win._llm = object()  # _on_warmup_done 只在模型仍在时改成就绪

    win._request_warmup()
    assert stub.requests == 1
    assert win._warmup_in_progress
    assert "正在预热模型" in win.status_label.text()

    win._on_warmup_done()
    assert not win._warmup_in_progress
    assert win.status_label.text().startswith("就绪")


def test_rag_status_result_kept_out_of_chat_view(qtbot, tmp_path, monkeypatch):
    """rag_status 的状态详情（含整份文件清单）只进日志窗口。

    用户反馈：一次 rag_status 就往对话里塞几十行文件清单，既挤满聊天区也白占
    上下文；这里确认聊天区只剩一行摘要，完整内容可在日志窗口查。
    """
    from agent.tools.base import ToolResult

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    win.chat_view.clear()
    win.log_view.clear()

    out = (
        "知识库状态:\n"
        "  知识库目录: /tmp/kb/\n"
        "  已存储切片数: 572200\n"
        "  已索引文件数: 80\n"
        + "\n".join(f"    - book-{i}.epub" for i in range(80))
    )
    win._on_tool_start("rag_status", {})
    win._on_tool_end("rag_status", ToolResult(success=True, output=out))

    chat = win.chat_view.toPlainText()
    assert "book-0.epub" not in chat        # 文件清单不进聊天区
    assert "572200" not in chat             # 状态细节也不进
    assert "↳ rag_status → 完成" in chat    # 只留一行完成提示

    log = win.log_view.toPlainText()
    assert "572200" in log                  # 完整状态在日志里
    assert "book-79.epub" in log


def test_context_shrink_and_clear_chat_rewarm(qtbot, tmp_path, monkeypatch):
    """上下文压缩 / 清空对话作废缓存后，要排队一次预热。

    这两个事件发生在「缓存已失效」之后，若不补预热，用户下一条消息就得等
    一次全量 prefill。压缩发生在一轮对话内部（_busy=True），此时仍要排队，
    由常驻 worker 在本轮结束后执行。
    """
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    stub = _install_stub_warmup_worker(win, monkeypatch)
    win._llm = object()

    # 1) 对话进行中发生压缩：照样排队，但不改状态栏、不记耗时。
    win._busy = True
    win.status_label.setText("正在生成…")
    win._on_context_shrunk("早期内容的摘要", "上下文超限")
    assert stub.requests == 1
    assert win._warmup_in_progress
    assert win._warmup_started is None      # 忙时不记时（预热要等本轮结束）
    assert win.status_label.text() == "正在生成…"   # 不覆盖当前状态

    # 2) 清空对话后也要补一次（此时空闲，正常显示预热中）。
    win._busy = False
    win._warmup_in_progress = False
    win._on_clear_chat()
    assert stub.requests == 2
    assert win._warmup_in_progress
    assert "正在预热模型" in win.status_label.text()
    assert win.llm_load_btn.styleSheet() == ""  # 卸载后恢复默认样式（不再绿色）


def test_aux_model_is_loaded_and_wired_into_agent(qtbot, tmp_path, monkeypatch):
    """辅助模型独立加载，并被 agent 用作摘要后端。"""
    import agent.ui.main_window as mw

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    created = []
    _install_fake_model_factory(monkeypatch, mw, created)
    monkeypatch.setattr(win, "_autoload_default_models", lambda: None)

    win._on_load_model("aux")
    qtbot.waitUntil(lambda: not win._model_busy.get("aux", False), timeout=5000)

    aux = created[-1]
    assert aux.ensure_calls == 1
    assert win._aux_llm is aux
    assert win._agent.aux_llm is aux

    win._on_unload_model("aux")
    assert aux._closed
    assert win._agent.aux_llm is not aux


def test_load_model_rejects_stale_config_change(qtbot, tmp_path, monkeypatch):
    """加载期间用户改了路径：结果丢弃，不装错模型。"""
    import agent.ui.main_window as mw

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    created = []
    _install_fake_model_factory(monkeypatch, mw, created)
    monkeypatch.setattr(win, "_autoload_default_models", lambda: None)

    # 拦截 worker：在「加载完成」前改掉路径，模拟加载期间配置变更。
    original = mw.LoadModelWorker

    class _SlowWorker(original):
        def run(self):
            win.llm_model_path_edit.setText("models/另一个模型")
            super().run()

    monkeypatch.setattr(mw, "LoadModelWorker", _SlowWorker)
    win._on_load_model("main")
    qtbot.waitUntil(lambda: not win._model_busy.get("main", False), timeout=5000)

    # 陈旧结果被丢弃：新加载出来的对象已关闭，且没有顶替掉 agent 在用的后端。
    assert created[-1]._closed
    assert win._llm is not created[-1]
    assert win._agent.llm is not created[-1]


def test_summarize_prefers_aux_llm_and_falls_back(qtbot, tmp_path, monkeypatch):
    """上下文摘要优先用辅助模型；辅助模型失败时回退主模型。"""
    from agent.agent import Agent
    from agent.config import AgentConfig

    class _Recording:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail
            self.calls = 0

        def chat_stream(self, messages, tools=None, temperature=0.7):
            self.calls += 1
            if self.fail:
                raise RuntimeError("辅助模型挂了")
            return iter([_evt("content", f"summary-by-{self.name}"),
                         _evt("done", usage={})])

    monkeypatch.setattr(mw_mod := __import__("agent.ui.main_window", fromlist=["x"]),
                        "create_llm_backend", lambda cfg: _Recording("main"))
    monkeypatch.setattr(mw_mod, "create_aux_llm_backend",
                        lambda cfg: _Recording("aux"))
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    aux = _Recording("aux")
    main = _Recording("main")
    agent = Agent(llm=main, config=win.config, callbacks=win._callbacks, aux_llm=aux)
    assert agent._summarize_via_llm("一些旧对话").startswith("summary-by-aux")
    assert aux.calls == 1 and main.calls == 0

    broken = _Recording("aux", fail=True)
    agent.aux_llm = broken
    out = agent._summarize_via_llm("一些旧对话")
    assert out.startswith("summary-by-main")
    assert broken.calls == 1 and main.calls == 1

    # 没有辅助模型时直接用主模型。
    agent.aux_llm = None
    assert agent._summarize_via_llm("x").startswith("summary-by-main")


def test_aux_model_block_is_independent_of_main_mode(qtbot, tmp_path, monkeypatch):
    """辅助模型常驻可见：主模型切到 OpenAI 接口模式时它不跟着消失（#140）。"""
    import agent.ui.main_window as mw

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    # 用假后端，避免测试真的去读 2B 权重。
    created = []
    _install_fake_model_factory(monkeypatch, mw, created)
    monkeypatch.setattr(win, "_autoload_default_models", lambda: None)

    # 本地模式：主/辅都可见。
    assert win.local_mode_panel.isVisibleTo(win.config_window)
    assert not win.openai_mode_panel.isVisibleTo(win.config_window)
    assert win.aux_llm_backend_combo.isVisibleTo(win.config_window)
    assert win.aux_llm_model_path_edit.isVisibleTo(win.config_window)
    assert win.aux_llm_load_btn.isVisibleTo(win.config_window)
    assert win.aux_llm_unload_btn.isVisibleTo(win.config_window)
    assert win.aux_llm_load_btn.isEnabled()

    # 切到接口模式：主模型的本地控件藏起来，辅助模型**仍在**且可操作。
    win.llm_mode_openai_radio.setChecked(True)
    assert not win.local_mode_panel.isVisibleTo(win.config_window)
    assert win.openai_mode_panel.isVisibleTo(win.config_window)
    assert win.aux_llm_backend_combo.isVisibleTo(win.config_window)
    assert win.aux_llm_model_path_edit.isVisibleTo(win.config_window)
    assert win.aux_llm_load_btn.isVisibleTo(win.config_window)
    assert win.aux_llm_load_btn.isEnabled()
    # 主模型的本地控件在接口模式下整个面板被隐藏。
    assert not win.llm_model_path_btn.isVisibleTo(win.config_window)
    assert not win.llm_backend_combo.isVisibleTo(win.config_window)

    # 接口模式下辅助模型照样能加载（后台线程，勿阻塞 UI）。
    win._on_load_model("aux")
    qtbot.waitUntil(lambda: not win._model_busy.get("aux", False), timeout=5000)
    assert win._is_model_loaded("aux")

    win.llm_mode_local_radio.setChecked(True)
    assert win.local_mode_panel.isVisibleTo(win.config_window)
    assert win.aux_llm_load_btn.isVisibleTo(win.config_window)


# ---------------------------------------------------------------------------
# 布局重构：配置/日志独立窗口 + 右栏任务计划
# ---------------------------------------------------------------------------

def test_config_and_log_live_in_own_windows(qtbot, tmp_path, monkeypatch):
    """配置区与日志区搬进独立窗口，主窗口顶栏只留入口按钮。"""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    assert win.log_view.parentWidget() is not None
    # 日志控件必须挂在日志窗口里，而不是主窗口中央部件。
    assert win.log_view.window() is win.log_window
    # 配置控件同理：base_url 输入框属于配置窗口。
    assert win.base_url_edit.window() is win.config_window

    # 顶栏按钮能唤出窗口（关闭是 hide，不会销毁控件）。
    win._on_open_config_window()
    win._on_open_log_window()
    assert win.config_window.isVisible()
    assert win.log_window.isVisible()

    # 主窗口仍持有对话 / 发送 / 思考 / 计划四个视图。
    for attr in ("chat_view", "send_editor", "think_view", "plan_view"):
        assert getattr(win, attr) is not None
    assert win.plan_view.window() is win
    assert win.think_view.window() is win


def test_task_plan_panel_tracks_steps(qtbot, tmp_path, monkeypatch):
    """任务制定计划后右上面板显示目标、进度，并标记进行中的步骤。"""
    from agent.planner import PlanStep, TaskPlan, save_plan

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    # 面板由 600ms 定时器轮询刷新，测试里直接调用一次而不是等定时器。
    win._poll_task_plan()
    assert "未制定计划" in win.plan_view.toPlainText()

    save_plan(win.config.workspace, TaskPlan(
        goal="重构主窗口布局",
        steps=[
            PlanStep(text="拆分配置区", status="done"),
            PlanStep(text="迁移思考区", status="in_progress", note="进行中"),
            PlanStep(text="接入计划面板", status="pending"),
        ],
    ))
    win._poll_task_plan()

    text = win.plan_view.toPlainText()
    assert "重构主窗口布局" in text
    assert "进度 1/3" in text
    assert "✓ 1. 拆分配置区" in text
    assert "▶ 2. 迁移思考区" in text
    assert "○ 3. 接入计划面板" in text
    assert win.plan_status_label.text() == "1/3 完成"
    # 进行中的步骤只有一个被高亮标记（QTextDocument 会把一个 div 拆成
    # block + span 两处 style，所以精确计数只能看渲染源串）。
    from agent.planner import load_plan

    plan = load_plan(win.config.workspace)
    assert win._render_plan_html(plan).count("background-color:#eff6ff") == 1
    assert "background-color:#eff6ff" in win.plan_view.toHtml()


def test_send_streams_content_into_chat(qtbot, tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    llm = _ScriptedLLM([[_evt("content", "He"), _evt("content", "llo"),
                         _evt("done", usage={"total_tokens": 3, "completion_tokens": 2})]])
    win._agent.llm = llm
    win._agent.max_iterations = 3

    win.send_editor.setPlainText("hi")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    html = win.chat_view.toHtml()
    assert "Hello" in html
    assert "你" in html
    assert "助手" in html
    assert "总计 3 tokens" in win.token_label.text()
    assert win.send_btn.isEnabled()
    assert not win.stop_btn.isEnabled()


def test_send_with_tool_call_shows_tool_line(qtbot, tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    with open(os.path.join(win.config.workspace, "d.txt"), "w") as f:
        f.write("42")
    tc = [{"id": "c1", "function": {"name": "file_read",
            "arguments": json.dumps({"path": "d.txt"})}}]
    llm = _ScriptedLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 4})],
        [_evt("content", "answer=42"), _evt("done", usage={"total_tokens": 6})],
    ])
    win._agent.llm = llm
    win._agent.max_iterations = 3

    win.send_editor.setPlainText("read d.txt")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    html = win.chat_view.toHtml()
    assert "file_read" in html
    assert "42" in html  # tool result snippet
    assert "answer=42" in html


def test_clear_chat_resets_history(qtbot, tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    win._append_chat("<p>hello</p>")
    win._agent._history.append({"role": "user", "content": "x"})
    win._on_clear_chat()
    assert win.chat_view.toPlainText() == ""


def test_task_plan_tool_refreshes_plan_panel(qtbot, tmp_path, monkeypatch):
    """task_plan 工具执行后，右上面板立即反映最新计划（不依赖 600ms 轮询），
    且清空对话会清掉上一轮残留的计划。"""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    def tc(name, args):
        return {"id": "c1", "function": {"name": name, "arguments": json.dumps(args)}}

    scripts = [
        [_evt("done", tool_calls=[tc("task_plan", {"op": "create", "goal": "任务A", "steps": ["a", "b"]})])],
        [_evt("content", "已创建"), _evt("done")],
        [_evt("done", tool_calls=[tc("task_plan", {"op": "update", "index": 1, "status": "done"})])],
        [_evt("content", "ok"), _evt("done")],
    ]
    llm = _ScriptedLLM(scripts)
    win._agent.llm = llm
    win._agent.max_iterations = 5

    win.send_editor.setPlainText("做任务A")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)
    text = win.plan_view.toPlainText()
    assert "任务A" in text
    assert "0/2" in text  # 刚创建，全部 pending

    win.send_editor.setPlainText("继续")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)
    text = win.plan_view.toPlainText()
    assert "✓" in text  # 第一步已标记完成
    assert "1/2" in text

    # 清空对话应清掉残留计划，面板回到「无计划」。
    win._on_clear_chat()
    assert "无计划" in win.plan_view.toPlainText() or "未制定计划" in win.plan_view.toPlainText()


# ---------------------------------------------------------------------------
# thinking-view output control: no overlap, always append at end
# ---------------------------------------------------------------------------

def test_sanitize_stream_text_normalizes_carriage_returns():
    import agent.ui.main_window as mw

    s = mw._sanitize_stream_text
    assert s("a\r\nb") == "a\nb"
    assert s("spin1\rspin2\rspin3") == "spin1\nspin2\nspin3"
    assert s("") == ""
    assert s("plain") == "plain"


def test_reasoning_appends_even_after_cursor_drift(qtbot, tmp_path, monkeypatch):
    """If the user clicks the thinking view (cursor moves off the end), new
    streaming chunks must still append at the end, not interleave/overlap."""
    from PyQt6.QtGui import QTextCursor

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    win._on_reasoning("abc")
    _flush_reason(win)
    # simulate the user clicking at the start of the thinking view
    cursor = win.think_view.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    win.think_view.setTextCursor(cursor)
    win._on_reasoning("def")
    _flush_reason(win)

    assert win.think_view.toPlainText() == "abcdef"


# ---------------------------------------------------------------------------
# skill suggestion flow: cached during run, dialog shown after _on_finished
# ---------------------------------------------------------------------------

def test_skill_suggestion_cached_during_run_shown_after_finish(qtbot, tmp_path, monkeypatch):
    """When the agent emits on_skill_suggested during a run, the UI must NOT
    show the modal dialog immediately (that would block streaming). Instead
    it caches the suggestion and shows the dialog only after _on_finished.

    This test monkeypatches QMessageBox.question and QInputDialog.getText so
    the dialog functions return scripted answers without blocking.
    """
    import agent.ui.main_window as mw
    from PyQt6.QtWidgets import QMessageBox, QInputDialog

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    # Patch dialogs to return scripted answers (Yes -> name "my-skill").
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("my-skill", True))

    # Pre-populate skill request history so the NEXT run triggers a suggestion.
    # Two identical requests -> 2nd triggers suggestion.
    win._agent.skills.record_request("deploy to staging")
    # Before running, no pending skill.
    assert win._pending_skill is None

    # Run a turn whose user text matches the recorded intent，且确实需要规划
    # （用到 task_plan）——简单问答/单次查询现在已经不提示固化技能了。
    tc_plan = [{"id": "c1", "function": {"name": "task_plan", "arguments": json.dumps(
        {"op": "create", "goal": "deploy", "steps": ["build", "publish"]})}}]
    llm = _ScriptedLLM([
        [_evt("done", tool_calls=tc_plan, usage={"total_tokens": 3})],
        [_evt("content", "done"), _evt("done", usage={"total_tokens": 3})],
    ])
    win._agent.llm = llm
    win._agent.max_iterations = 3
    win.send_editor.setPlainText("deploy to staging")
    win._on_send()

    # While busy: _pending_skill should be set, but NO dialog shown yet
    # (the patched QMessageBox.question would have been called if dialog was
    # shown during streaming — we verify it wasn't via the skill not being
    # saved until after finish).
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    # After _on_finished: the dialog must have been shown, and the skill
    # saved under the name "my-skill".
    saved = win._agent.skills.list()
    assert any(s.name == "my-skill" for s in saved), (
        f"expected skill 'my-skill' saved after finish, got {[s.name for s in saved]}"
    )
    # Pending skill must be cleared after the dialog.
    assert win._pending_skill is None


def test_skill_suggestion_does_not_block_streaming(qtbot, tmp_path, monkeypatch):
    """The skill-suggestion dialog must NOT appear during streaming. If a
    suggestion fires mid-stream, content chunks must still land in the chat
    view immediately. This is verified by checking that the chat view shows
    content before _on_finished completes (which is when the dialog would
    be shown under the new deferred-display behavior)."""
    import agent.ui.main_window as mw
    from PyQt6.QtWidgets import QMessageBox, QInputDialog

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    # Track whether the dialog was called during streaming.
    dialog_calls = {"count": 0}
    def _track_question(*a, **k):
        dialog_calls["count"] += 1
        return QMessageBox.StandardButton.No  # decline to save
    monkeypatch.setattr(QMessageBox, "question", _track_question)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("x", True))

    # Pre-populate so the next run triggers a suggestion.
    win._agent.skills.record_request("deploy to staging")

    # 先规划（task_plan）再输出：只有这种「需要计划/多步」的轮次才会提示固化。
    tc_plan = [{"id": "c1", "function": {"name": "task_plan", "arguments": json.dumps(
        {"op": "create", "goal": "deploy", "steps": ["build", "publish"]})}}]
    llm = _ScriptedLLM([
        [_evt("done", tool_calls=tc_plan, usage={"total_tokens": 3})],
        [_evt("content", "streamed-content"), _evt("done", usage={"total_tokens": 3})],
    ])
    win._agent.llm = llm
    win._agent.max_iterations = 3
    win.send_editor.setPlainText("deploy to staging")
    win._on_send()

    # Wait for the turn to finish.
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    # The dialog must have been called exactly once — AFTER streaming finished,
    # not during. The streamed content must have landed in the chat view.
    assert dialog_calls["count"] == 1, (
        f"expected dialog exactly once after finish, got {dialog_calls['count']}"
    )
    assert "streamed-content" in win.chat_view.toHtml()


def test_reasoning_with_carriage_returns_does_not_overlap(qtbot, tmp_path, monkeypatch):
    """Bare \\r (overstrike) must become a newline so later text doesn't print
    on top of the current line."""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    win._on_reasoning("line1\rline2")
    _flush_reason(win)

    assert win.think_view.toPlainText() == "line1\nline2"


def test_reasoning_empty_chunk_is_noop(qtbot, tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    win._on_reasoning("keep")
    win._on_reasoning("")  # empty / sanitizes to empty -> ignored
    _flush_reason(win)
    assert win.think_view.toPlainText() == "keep"
    assert win._agent._history == []


# ---------------------------------------------------------------------------
# auto-scroll-to-bottom: latest content must stay visible in every text view
# ---------------------------------------------------------------------------

def _force_layout(widget):
    """Make the widget actually size itself so its scrollbar has a real range.

    Under the offscreen platform a never-shown widget has viewport size 0 and a
    scrollbar range of 0..0, which makes "value == maximum" trivially true and
    hides real bugs. We resize the viewport, show the widget, and pump the event
    loop so the document layout settles before assertions.
    """
    widget.resize(400, 120)
    widget.show()
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()


def test_reasoning_scrolls_to_bottom_after_each_chunk(qtbot, tmp_path, monkeypatch):
    """Each reasoning chunk must leave the scrollbar pinned to the bottom so the
    latest line is always visible, even after the user scrolled up to read
    earlier output."""
    from PyQt6.QtGui import QTextCursor
    from PyQt6.QtWidgets import QApplication

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    _force_layout(win.think_view)

    # Fill enough lines to make the view scrollable, then verify bottom is shown.
    for i in range(40):
        win._on_reasoning(f"line {i}\n")
        _flush_reason(win)
        sb = win.think_view.verticalScrollBar()
        assert sb.value() == sb.maximum(), (
            f"think_view not at bottom after line {i}: "
            f"value={sb.value()} maximum={sb.maximum()}"
        )

    # User scrolls up to read earlier output.
    sb = win.think_view.verticalScrollBar()
    sb.setValue(sb.minimum())
    QApplication.processEvents()
    assert sb.value() < sb.maximum()

    # A new chunk arrives -> must snap back to the bottom.
    win._on_reasoning("new tail line\n")
    _flush_reason(win)
    sb = win.think_view.verticalScrollBar()
    assert sb.value() == sb.maximum()


def test_chat_view_scrolls_to_bottom_on_content(qtbot, tmp_path, monkeypatch):
    """Streaming content into the chat view must keep the scrollbar at the
    bottom, including after the user scrolled up."""
    from PyQt6.QtWidgets import QApplication

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    _force_layout(win.chat_view)

    for i in range(40):
        win._on_content(f"message chunk {i}\n")
        sb = win.chat_view.verticalScrollBar()
        assert sb.value() == sb.maximum(), (
            f"chat_view not at bottom after chunk {i}: "
            f"value={sb.value()} maximum={sb.maximum()}"
        )

    # After user scrolls up, the next chunk must still pin to bottom.
    sb = win.chat_view.verticalScrollBar()
    sb.setValue(sb.minimum())
    QApplication.processEvents()
    win._on_content("tail\n")
    sb = win.chat_view.verticalScrollBar()
    assert sb.value() == sb.maximum()


def test_log_view_scrolls_to_bottom_on_append(qtbot, tmp_path, monkeypatch):
    """Appending to the log view must keep it pinned to the bottom."""
    from PyQt6.QtWidgets import QApplication

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    _force_layout(win.log_view)

    for i in range(40):
        win._append_log(f"log line {i}")
        sb = win.log_view.verticalScrollBar()
        assert sb.value() == sb.maximum(), (
            f"log_view not at bottom after line {i}: "
            f"value={sb.value()} maximum={sb.maximum()}"
        )

    sb = win.log_view.verticalScrollBar()
    sb.setValue(sb.minimum())
    QApplication.processEvents()
    win._append_log("tail")
    sb = win.log_view.verticalScrollBar()
    assert sb.value() == sb.maximum()


def test_append_chat_scrolls_to_bottom(qtbot, tmp_path, monkeypatch):
    """Direct _append_chat calls (tool lines, errors) must also pin to bottom."""
    from PyQt6.QtWidgets import QApplication

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    _force_layout(win.chat_view)

    for i in range(40):
        win._append_chat(f"<p>block {i}</p>")
        sb = win.chat_view.verticalScrollBar()
        assert sb.value() == sb.maximum()

    sb = win.chat_view.verticalScrollBar()
    sb.setValue(sb.minimum())
    QApplication.processEvents()
    win._append_chat("<p>tail</p>")
    sb = win.chat_view.verticalScrollBar()
    assert sb.value() == sb.maximum()


def test_scratch_cache_released_when_idle(qtbot, tmp_path, monkeypatch):
    """一轮结束（空闲）时释放 MLX 临时缓冲；同步进行中不动它。

    运行期从不 clear_cache() 会让 prefill/生成攒下的 buffer 一直占着统一内存，
    用户实测顶到 13G 后整机换页卡顿。
    """
    import sys, types
    import agent.ui.main_window as mw

    calls = []

    class FakeMX:
        @staticmethod
        def clear_cache():
            calls.append(1)

    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.clear_cache = FakeMX.clear_cache
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)

    win._sync_busy = False
    win._release_mlx_scratch_cache()
    assert calls == [1]

    calls.clear()
    win._sync_busy = True           # 同步/检索进行中：不清理（那边正在复用 buffer）
    win._release_mlx_scratch_cache()
    assert calls == []


def _skill_dialog_probe(win, monkeypatch):
    """返回记录「是否弹过保存技能对话框」的计数器。"""
    from PyQt6.QtWidgets import QMessageBox, QInputDialog

    calls = {"n": 0}

    def _track_question(*a, **k):
        calls["n"] += 1
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", _track_question)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("x", True))
    return calls


def test_no_skill_suggestion_for_chitchat(qtbot, tmp_path, monkeypatch):
    """寒暄 / 单次问答：不提示保存为技能。"""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    calls = _skill_dialog_probe(win, monkeypatch)

    for _ in range(3):
        win._agent.skills.record_request("你好啊")
    llm = _ScriptedLLM([[_evt("content", "你好！"), _evt("done", usage={"total_tokens": 3})]])
    win._agent.llm = llm
    win.send_editor.setPlainText("你好啊")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    assert calls["n"] == 0
    assert "你好！" in win.chat_view.toHtml()


def test_no_skill_suggestion_for_single_query(qtbot, tmp_path, monkeypatch):
    """只调用一次工具的查询：不提示。"""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    calls = _skill_dialog_probe(win, monkeypatch)

    for _ in range(3):
        win._agent.skills.record_request("查一下知识库")
    tc = [{"id": "c1", "function": {"name": "rag_status", "arguments": "{}"}}]
    llm = _ScriptedLLM([
        [_evt("done", tool_calls=tc, usage={"total_tokens": 3})],
        [_evt("content", "查到了"), _evt("done", usage={"total_tokens": 3})],
    ])
    win._agent.llm = llm
    win.send_editor.setPlainText("查一下知识库")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    assert calls["n"] == 0


def test_skill_suggestion_after_multi_step_turn(qtbot, tmp_path, monkeypatch):
    """多步完成（≥3 次工具调用）：才提示保存为技能。"""
    win = _make_window(tmp_path, monkeypatch)
    qtbot.addWidget(win)
    calls = _skill_dialog_probe(win, monkeypatch)

    for _ in range(3):
        win._agent.skills.record_request("整理这份报告")

    def tc(i, name):
        return [{"id": f"c{i}", "function": {"name": name, "arguments": "{}"}}]

    llm = _ScriptedLLM([
        [_evt("done", tool_calls=tc(1, "rag_status"), usage={"total_tokens": 3})],
        [_evt("done", tool_calls=tc(2, "rag_status"), usage={"total_tokens": 3})],
        [_evt("done", tool_calls=tc(3, "rag_status"), usage={"total_tokens": 3})],
        [_evt("content", "整理好了"), _evt("done", usage={"total_tokens": 3})],
    ])
    win._agent.llm = llm
    win._agent.max_iterations = 6
    win.send_editor.setPlainText("整理这份报告")
    win._on_send()
    qtbot.waitUntil(lambda: not win._busy, timeout=5000)

    assert calls["n"] == 1
