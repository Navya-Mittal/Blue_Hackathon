import json
import re

import numpy as np
import pandas as pd
import pytest

from blue import docgen
from blue.audit import Audit
from blue.config import resolve
from blue.fixtures import load_reference
from blue.inventory import scan
from blue.llm import LLMClient
from blue.pipeline import run
from blue.redevelop import DRAFT2, DRAFT3, compare, repair_loop, run_candidate


@pytest.fixture(scope="module")
def ref_and_facts(cfg, splits):
    ref = load_reference(cfg, "credit_pd", splits)
    facts = next(f for f in scan(cfg) if f.model_id == "credit_pd")
    return ref, facts


def test_fixtures_are_from_real_r_and_match_current_inputs(cfg, splits):
    ref = load_reference(cfg, "credit_pd", splits)  # raises if splits or script changed
    assert ref["manifest"]["r_version"].startswith("Rscript (R) version")
    assert len(ref["coefficients"]) == 13


def test_repair_loop_converges_in_three_diagnosis_driven_steps(
    cfg, splits, ref_and_facts, tmp_path
):
    ref, facts = ref_and_facts
    audit = Audit(tmp_path / "a.jsonl")
    llm = LLMClient("offline", tmp_path / "c", audit)
    y = pd.read_csv(splits["test"])["default"].to_numpy()
    res = repair_loop(
        llm=llm,
        facts=facts,
        r_source=(resolve(cfg, "legacy_dir") / "credit_pd.R").read_text(),
        train_csv=splits["train"],
        score_csvs={"test": splits["test"], "edge": splits["edge"]},
        ref=ref,
        y_test=y,
        cfg=cfg,
        out_dir=tmp_path / "port",
        audit=audit,
    )
    assert res.passed and len(res.iterations) == 3
    assert "MISSING_COEFFICIENTS" in res.iterations[0].diagnosis
    assert "SYSTEMATIC_SHRINKAGE" in res.iterations[1].diagnosis
    assert res.iterations[2].diagnosis == [] and (tmp_path / "port" / "draft_1_to_2.diff").exists()


def _run(code, cfg, splits, tmp_path):
    p = tmp_path / "cand.py"
    p.write_text(code)
    return run_candidate(p, splits["train"], {"test": splits["test"], "edge": splits["edge"]}, 180)


def test_correct_port_matches_r_within_tolerance_including_nan_rows(
    cfg, splits, ref_and_facts, tmp_path
):
    ref, facts = ref_and_facts
    y = pd.read_csv(splits["test"])["default"].to_numpy()
    cand = _run(DRAFT3, cfg, splits, tmp_path)
    checks, diag = compare(ref, cand, y, facts.cutoff, cfg)
    assert diag == [] and all(c.passed for c in checks)
    edge = np.array([np.nan if v is None else v for v in cand["predictions"]["edge"]])
    assert (
        np.isnan(edge).sum() == 2
    )  # the two edge rows with missing predictors stay NaN, like R's NA


def test_penalised_port_is_rejected_and_diagnosed(cfg, splits, ref_and_facts, tmp_path):
    ref, facts = ref_and_facts
    y = pd.read_csv(splits["test"])["default"].to_numpy()
    checks, diag = compare(ref, _run(DRAFT2, cfg, splits, tmp_path), y, facts.cutoff, cfg)
    assert not all(c.passed for c in checks) and "SYSTEMATIC_SHRINKAGE" in diag


def test_tiny_numerical_drift_is_caught(cfg, splits, ref_and_facts, tmp_path):
    ref, facts = ref_and_facts
    y = pd.read_csv(splits["test"])["default"].to_numpy()
    cand = _run(DRAFT3, cfg, splits, tmp_path)
    cand["predictions"]["test"] = [
        None if v is None else v + 1e-3 for v in cand["predictions"]["test"]
    ]
    checks, diag = compare(ref, cand, y, facts.cutoff, cfg)
    assert "PRED_MISMATCH" in diag and not all(c.passed for c in checks)


def test_full_pipeline_offline_end_to_end(tmp_cfg):
    out = run(tmp_cfg, mode="offline", use_mlflow=False, say=lambda *_: None)
    art = out["artifacts"]
    for name in (
        "report.html",
        "metrics.json",
        "parity.json",
        "monitor.json",
        "impact.json",
        "audit.jsonl",
        "inventory.json",
        "revalidation_ticket.json",
    ):
        assert (art / name).exists(), name
    assert (
        out["parity"].passed and out["impact"]["claims_verified"] == out["impact"]["claims_drafted"]
    )
    assert out["impact"]["stale_doc_discrepancies"] == 5
    html = (art / "report.html").read_text()
    assert "auto-demo" in html and "Independent validation report" in html
    assert {e["reviewer"] for e in out["audit"].events if e["kind"] == "approval"} == {"auto-demo"}


class ScriptedLLM:
    """Plays the role of a real LLM: one hallucinated doc claim, one good port, one ungrounded narrative."""

    name = "scripted:fake-llm"

    def __init__(self, cfg):
        self.cfg = cfg
        self.facts = {f.path: f for f in scan(cfg)}

    def complete(self, system, user, *, temperature, max_tokens):
        if "model-documentation assistant" in system:
            path = re.search(r"FILE: (\S+)", user).group(1)
            claims = [
                json.loads(c.model_dump_json())
                for c in docgen.offline_draft(self.facts[path]).claims
            ]
            claims.append(
                {
                    "section": "Purpose",
                    "text": "Invented claim.",
                    "evidence": [
                        {"start_line": 1, "end_line": 1, "quote": "a quote that is not in the file"}
                    ],
                }
            )
            return json.dumps({"claims": claims}), {}
        if "translating a legacy R model" in system:
            return "```python\n" + DRAFT3 + "```", {}
        return "The Gini is 0.999 on 5 accounts.\n\nNothing else.\n\nDone.", {}


def test_live_mode_guards_catch_llm_mistakes_then_replay_reproduces(tmp_cfg):
    live = run(
        tmp_cfg, mode="live", use_mlflow=False, say=lambda *_: None, provider=ScriptedLLM(tmp_cfg)
    )
    art = live["artifacts"]
    # hallucinated doc claim is flagged, not silently kept
    assert live["impact"]["claims_verified"] == live["impact"]["claims_drafted"] - 3
    md = (art / "docs" / "credit_pd.md").read_text()
    assert "[UNVERIFIED] Invented claim." in md
    # a good LLM port passes parity first time
    assert live["parity"].passed and len(live["parity"].iterations) == 1
    # ungrounded narrative is rejected and logged
    assert any(e["kind"] == "narrative_rejected" for e in live["audit"].events)
    html = (art / "report.html").read_text()
    assert (
        "The Gini is 0.999 on 5 accounts" not in html
    )  # fabricated text never reaches the report body
    assert (
        "rejected by the numeric grounding check (ungrounded: 0.999)" in html
    )  # ...and the rejection is disclosed
    # replay reproduces the run with no provider at all
    replay = run(tmp_cfg, mode="replay", use_mlflow=False, say=lambda *_: None)
    assert (
        replay["parity"].passed
        and replay["metrics"]["performance"]["gini"] == live["metrics"]["performance"]["gini"]
    )
    assert {e["source"] for e in replay["audit"].events if e["kind"] == "llm_call"} == {"cache"}
