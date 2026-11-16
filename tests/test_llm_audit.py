import json

import pytest

from blue.audit import Audit, write_ai_usage
from blue.llm import LLMClient, ReplayMiss


class FakeProvider:
    name = "fake:test-model"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, *, temperature, max_tokens):
        self.calls += 1
        return f"answer-{self.calls}", {"input_tokens": 1, "output_tokens": 1}


def test_live_records_then_replay_returns_same_text_without_provider(tmp_path):
    audit = Audit(tmp_path / "a.jsonl")
    prov = FakeProvider()
    live = LLMClient("live", tmp_path / "cache", audit, provider=prov)
    first = live.complete("t", "sys", "user prompt")
    assert first.source == "live:fake:test-model" and prov.calls == 1

    replay = LLMClient("replay", tmp_path / "cache", audit)
    again = replay.complete("t", "sys", "user prompt")
    assert again.text == first.text and again.source == "cache" and prov.calls == 1


def test_replay_miss_is_an_error_not_a_silent_fallback(tmp_path):
    audit = Audit(tmp_path / "a.jsonl")
    with pytest.raises(ReplayMiss):
        LLMClient("replay", tmp_path / "cache", audit).complete("t", "sys", "never recorded")


def test_offline_mode_needs_a_standin_and_is_labelled(tmp_path):
    audit = Audit(tmp_path / "a.jsonl")
    c = LLMClient("offline", tmp_path / "cache", audit)
    with pytest.raises(RuntimeError):
        c.complete("t", "s", "u")
    assert c.complete("t", "s", "u", offline=lambda: "x").source == "offline-standin"


def test_every_call_is_audited_and_usage_declaration_reflects_it(tmp_path):
    audit = Audit(tmp_path / "a.jsonl")
    c = LLMClient("offline", tmp_path / "cache", audit)
    c.complete("docgen", "s", "u1", offline=lambda: "x")
    c.complete("translate", "s", "u2", offline=lambda: "y")
    lines = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in lines] == ["llm_call", "llm_call"]
    assert all(set(["prompt_sha", "response_sha", "source"]) <= set(row) for row in lines)
    text = write_ai_usage(audit.events, tmp_path / "AI_USAGE.md")
    assert "| docgen | offline-standin | 1 |" in text and "TEAM TO CONFIRM" in text


def test_approvals_are_never_recorded_as_a_human_unless_interactive(tmp_path):
    a = Audit(tmp_path / "a.jsonl")
    a.approve("risk tier", "m", "s")
    assert a.events[-1]["reviewer"] == "auto-demo"
    h = Audit(tmp_path / "b.jsonl", interactive=True, prompt=lambda q: "n")
    assert h.approve("risk tier", "m", "s") is False
    assert h.events[-1]["decision"] == "rejected" and h.events[-1]["reviewer"] == "interactive-user"
