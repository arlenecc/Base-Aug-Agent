"""Tests for the skill manager (固化 frequently-run tasks)."""
from __future__ import annotations

import pytest

from agent.skills import SkillManager


def test_record_request_returns_none_until_threshold(tmp_path):
    sm = SkillManager(path=str(tmp_path / "skills.json"))
    # first time: no suggestion
    assert sm.record_request("deploy to staging") is None
    # second time: should suggest固化
    skill = sm.record_request("deploy to staging")
    assert skill is not None
    assert "deploy" in skill.keywords or "staging" in skill.keywords


def test_record_request_paraphrased_requests_share_key(tmp_path):
    """Paraphrased requests like 'deploy to staging' and 'please deploy staging'
    must map to the same request key so the counter accumulates. The old
    implementation joined keywords in original order, so dropping/adding
    stopwords like 'please' produced a different key and never reached the
    threshold."""
    sm = SkillManager(path=str(tmp_path / "skills.json"))
    # Two phrasings of the same intent (differ only by stopwords).
    assert sm.record_request("deploy to staging") is None  # 1st
    assert sm.record_request("please deploy staging") is not None  # 2nd -> suggest


def test_request_key_normalizes_stopwords(tmp_path):
    """_request_key must strip stopwords and sort so order-independent."""
    from agent.skills import SkillManager
    k1 = SkillManager._request_key("deploy to staging")
    k2 = SkillManager._request_key("staging deploy")
    k3 = SkillManager._request_key("please deploy the staging")
    assert k1 == k2 == k3
    # Empty/stopword-only text must yield empty key.
    assert SkillManager._request_key("the a an") == ""
    assert SkillManager._request_key("") == ""


def test_skills_reload_picks_up_external_writes(tmp_path):
    """If another process/instance writes to the skills file, reload() must
    pick up the new data so match() sees it — otherwise the worker thread's
    SkillManager would keep returning stale (pre-save) results."""
    p = str(tmp_path / "skills.json")
    sm = SkillManager(path=p)
    assert sm.list() == []
    # Another instance writes a skill.
    sm2 = SkillManager(path=p)
    sm2.create_skill(name="deploy", keywords=["deploy", "staging"], prompt="deploy it")
    # sm (old instance) doesn't see it yet.
    assert sm.match("deploy staging") is None
    # After reload, sm sees the new skill.
    sm.reload()
    matched = sm.match("deploy staging")
    assert matched is not None
    assert matched.name == "deploy"


def test_jsonstore_reload_picks_up_external_writes(tmp_path):
    """_JsonStore.reload() must re-read the file, discarding the in-memory
    cache, so a separate writer's changes become visible."""
    from agent.memory import _JsonStore
    p = str(tmp_path / "store.json")
    s1 = _JsonStore(p)
    s1.set("k", "v1")
    # Another store instance writes a new value.
    s2 = _JsonStore(p)
    s2.set("k", "v2")
    # s1 still sees the old value.
    assert s1.get("k") == "v1"
    # After reload, s1 sees the new value.
    s1.reload()
    assert s1.get("k") == "v2"


def test_create_and_match_skill(tmp_path):
    sm = SkillManager(path=str(tmp_path / "skills.json"))
    sm.create_skill(name="daily-report", keywords=["daily", "report"], prompt="Generate the daily metrics report")
    matched = sm.match("please run the daily report")
    assert matched is not None
    assert matched.name == "daily-report"
    assert sm.match("nothing relevant here") is None


def test_skills_persist(tmp_path):
    p = str(tmp_path / "skills.json")
    sm = SkillManager(path=p)
    sm.create_skill(name="greet", keywords=["hello"], prompt="say hi")
    sm2 = SkillManager(path=p)
    assert any(s.name == "greet" for s in sm2.list())


def test_match_returns_highest_score(tmp_path):
    sm = SkillManager(path=str(tmp_path / "skills.json"))
    sm.create_skill(name="a", keywords=["alpha"], prompt="a")
    sm.create_skill(name="b", keywords=["alpha", "beta"], prompt="b")
    matched = sm.match("alpha beta")
    assert matched.name == "b"


def test_record_request_different_requests_dont_trigger(tmp_path):
    sm = SkillManager(path=str(tmp_path / "skills.json"))
    assert sm.record_request("task A") is None
    assert sm.record_request("task B") is None


def test_skills_handles_corrupted_null_values(tmp_path):
    """If the store file has null skills/requests (corruption), SkillManager
    must recover to empty defaults, not raise TypeError on iteration."""
    p = str(tmp_path / "skills.json")
    with open(p, "w") as f:
        f.write('{"skills": null, "requests": null}')
    # Must not raise
    sm = SkillManager(path=p)
    assert sm.list() == []
    # create_skill must work
    sm.create_skill(name="s1", keywords=["k"], prompt="p")
    assert any(s.name == "s1" for s in sm.list())
    # match must not crash
    assert sm.match("k") is not None


def test_skills_delete_handles_corrupted_store(tmp_path):
    """delete on a store with null skills must not crash."""
    p = str(tmp_path / "skills.json")
    with open(p, "w") as f:
        f.write('{"skills": null, "requests": null}')
    sm = SkillManager(path=p)
    sm.delete("nonexistent")  # must not raise
    assert sm.list() == []
