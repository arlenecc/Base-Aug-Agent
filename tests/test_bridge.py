"""Tests for the UI bridge's blocking confirm/ask synchronization.

These tests exercise the threading.Event synchronization primitives directly,
so we disconnect the modal-dialog slots to avoid lingering queued events that
would otherwise pop a QMessageBox when a later test runs the event loop.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_window(qapp):
    import agent.ui.main_window as mw

    win = mw.MainWindow()
    # isolate the sync primitives from the modal-dialog slots
    try:
        win._bridge.confirm_request.disconnect()
    except TypeError:
        pass
    try:
        win._bridge.ask_request.disconnect()
    except TypeError:
        pass
    return win


def test_bridge_confirm_returns_ui_result(qapp):
    win = _make_window(qapp)
    cb = win._callbacks

    result_holder = {}

    def worker():
        result_holder["val"] = cb.confirm("proceed?")

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    cb.set_confirm_result(False)
    t.join(timeout=2.0)

    assert result_holder["val"] is False
    win.close()


def test_bridge_ask_user_returns_string(qapp):
    win = _make_window(qapp)
    cb = win._callbacks

    holder = {}

    def worker():
        holder["val"] = cb.ask_user("which option?")

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    cb.set_ask_result("option B")
    t.join(timeout=2.0)

    assert holder["val"] == "option B"
    win.close()


def test_bridge_cancel_short_circuits_confirm(qapp):
    win = _make_window(qapp)
    cb = win._callbacks
    cb.cancel()
    assert cb.confirm("anything") is False
    assert cb.ask_user("anything") is None
    win.close()
