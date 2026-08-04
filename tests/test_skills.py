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
