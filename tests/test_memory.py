"""Tests for work memory and long-term memory."""
from __future__ import annotations

import os

from agent.memory import WorkMemory, LongTermMemory


def test_work_memory_set_get(tmp_path):
    wm = WorkMemory(path=str(tmp_path / "wm.json"))
    assert wm.get("x") is None
    wm.set("x", "1")
    assert wm.get("x") == "1"
    assert wm.list() == {"x": "1"}


def test_work_memory_clear(tmp_path):
    wm = WorkMemory(path=str(tmp_path / "wm.json"))
    wm.set("a", "1")
    wm.clear()
    assert wm.list() == {}


def test_work_memory_persists_to_disk(tmp_path):
    p = str(tmp_path / "wm.json")
    wm = WorkMemory(path=p)
    wm.set("k", "v")
    wm2 = WorkMemory(path=p)
    assert wm2.get("k") == "v"


def test_longterm_memory_add_and_search(tmp_path):
    mem = LongTermMemory(path=str(tmp_path / "lt.json"))
    mem.add("user prefers dark mode")
    mem.add("user codes in python")
    results = mem.search("python")
    assert any("python" in r for r in results)
    results = mem.search("dark")
    assert any("dark" in r for r in results)


def test_longterm_memory_persists(tmp_path):
    p = str(tmp_path / "lt.json")
    mem = LongTermMemory(path=p)
    mem.add("fact one")
    mem2 = LongTermMemory(path=p)
    assert "fact one" in mem2.all()


def test_longterm_memory_search_returns_empty_when_no_match(tmp_path):
    mem = LongTermMemory(path=str(tmp_path / "lt.json"))
    mem.add("hello world")
    assert mem.search("zzzz") == []


def test_longterm_memory_handles_corrupted_null_facts(tmp_path):
    """If the store file has {"facts": null} (corruption/partial write), the
    LongTermMemory must recover to an empty list, not raise TypeError."""
    p = str(tmp_path / "lt.json")
    with open(p, "w") as f:
        f.write('{"facts": null}')
    # Must not raise
    mem = LongTermMemory(path=p)
    assert mem.all() == []
    # add must work and produce a valid list
    mem.add("fact one")
    assert mem.all() == ["fact one"]
    # add_many must work too
    mem.add_many(["fact two", "fact three"])
    assert "fact two" in mem.all()
    assert "fact three" in mem.all()


def test_longterm_memory_add_many_handles_corrupted_store(tmp_path):
    """add_many on a store with null facts must not crash."""
    p = str(tmp_path / "lt.json")
    with open(p, "w") as f:
        f.write('{"facts": null}')
    mem = LongTermMemory(path=p)
    mem.add_many(["a", "b", "c"])
    assert mem.all() == ["a", "b", "c"]
