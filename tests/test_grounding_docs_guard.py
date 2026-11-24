import json

from blue import docgen
from blue.config import resolve
from blue.inventory import scan
from blue.redevelop import check_code
from blue.report import numeric_leaves, strip_ci_level, ungrounded_numbers


def test_grounding_accepts_computed_numbers_and_rejects_invented_ones():
    metrics = {"gini": 0.4371, "n_test": 7200, "rate": 0.2211, "nested": {"psi": 0.0011}}
    allowed = numeric_leaves(metrics)
    ok = "The Gini is 0.437 on 7,200 accounts; the default rate is 22.1% and PSI is 0.0011."
    assert ungrounded_numbers(ok, allowed) == []
    planted = "The Gini is 0.512 on 7,200 accounts."
    assert ungrounded_numbers(planted, allowed) == ["0.512"]


def test_grounding_allows_the_fixed_95_percent_interval_level_only_next_to_interval_wording():
    allowed = numeric_leaves({"gini": 0.437, "lo": 0.403, "hi": 0.469, "n": 7200})
    for ok in (
        "Gini is 0.437 (95% CI 0.403 to 0.469).",
        "Gini is 0.437, with a 95% confidence interval of 0.403 to 0.469.",
        "Gini is 0.437 (95 % bootstrap intervals 0.403 to 0.469).",
    ):
        assert ungrounded_numbers(strip_ci_level(ok), allowed) == [], ok
    # A 95 used any other way is still an ungrounded number, and invented numbers beside it are still caught.
    assert ungrounded_numbers(strip_ci_level("About 95% of the 7,200 accounts were approved."), allowed) == ["95%"]
    assert ungrounded_numbers(strip_ci_level("There were 95 accounts."), allowed) == ["95"]
    assert ungrounded_numbers(strip_ci_level("Gini is 0.512 (95% CI 0.403 to 0.469)."), allowed) == ["0.512"]


def test_grounding_ignores_booleans_and_nan():
    assert numeric_leaves({"a": True, "b": float("nan"), "c": 3}) == [3.0]


def test_scanner_extracts_facts_from_r_and_python(cfg):
    facts = {f.model_id: f for f in scan(cfg)}
    c = facts["credit_pd"]
    assert (c.language, c.family, c.target, c.n_predictors, c.cutoff, c.regularisation) == (
        "R",
        "binomial",
        "default",
        9,
        0.3,
        False,
    )
    assert c.algorithm_class == "logistic regression"
    g = facts["fraud_gbm"]
    assert (g.language, g.algorithm_class, g.n_predictors, g.cutoff) == (
        "Python",
        "gradient boosting",
        7,
        0.8,
    )
    assert facts["pricing_glm"].family == "gaussian"


def test_stale_doc_detector_finds_planted_discrepancies_and_no_false_alarm(cfg):
    facts = {f.model_id: f for f in scan(cfg)}
    root = resolve(cfg, "legacy_dir")
    d = docgen.stale_check(
        (root / "credit_pd_model_doc.md").read_text(), facts["credit_pd"], cfg["data"]["protected"]
    )
    assert {x.field for x in d} == {
        "Algorithm",
        "Number of predictors",
        "Regularisation",
        "Decision cut-off",
        "Protected attributes used as predictors",
    }
    assert all(x.evidence for x in d)  # every finding cites code lines
    ctrl = docgen.stale_check(
        (root / "pricing_glm_model_doc.md").read_text(),
        facts["pricing_glm"],
        cfg["data"]["protected"],
    )
    assert ctrl == []  # control case: no false alarms


def test_generated_claims_are_verified_and_tampering_is_caught(cfg):
    f = next(x for x in scan(cfg) if x.model_id == "credit_pd")
    src = (resolve(cfg, "legacy_dir").parent / f.path).read_text()
    doc = docgen.verify_claims(docgen.offline_draft(f), {f.path: src})
    assert all(c.verified for c in doc.claims)
    doc.claims[1].evidence[0].quote = "this text does not appear in the source"
    doc = docgen.verify_claims(doc, {f.path: src})
    assert doc.claims[1].verified is False and doc.claims[0].verified is True


def test_claim_without_evidence_is_never_verified(cfg):
    f = next(x for x in scan(cfg) if x.model_id == "credit_pd")
    raw = json.dumps({"claims": [{"section": "Purpose", "text": "Trust me.", "evidence": []}]})
    claims = docgen.parse_llm_claims(raw, f.path)
    doc = docgen.verify_claims(
        docgen.ModelDoc(model_id="credit_pd", generated_by="t", claims=claims), {f.path: "x"}
    )
    assert doc.claims[0].verified is False


def test_code_guard_blocks_dangerous_code():
    assert check_code("import numpy as np\nimport statsmodels.api as sm\n") == []
    assert check_code("import os\n")
    assert check_code("import subprocess\n")
    assert check_code("x = open('f').read()\n")
    assert check_code("import pandas as pd\npd.read_csv('secret.csv')\n")
    assert check_code("def broken(:\n")
