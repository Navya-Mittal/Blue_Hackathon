import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from blue import metrics as M
from blue.validate import fairness, rag_high_good, rag_low_good


def test_psi_identical_is_zero_and_shift_is_positive():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert M.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert M.psi(a, a + 1.0) > 0.25


def test_gini_ks_match_definitions():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.4, 0.5, 0.9])
    assert M.gini(y, s) == pytest.approx(2 * roc_auc_score(y, s) - 1)
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    grid = np.unique(s)
    ks = max(abs((pos <= t).mean() - (neg <= t).mean()) for t in grid)
    assert M.ks(y, s) == pytest.approx(ks)


def test_calibration_table_covers_all_rows():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    t = M.calibration_table(y, p, 10)
    assert t["n"].sum() == 1000 and len(t) == 10


def test_rag_helpers():
    assert (
        rag_high_good(0.5, 0.4, 0.3) == "Green"
        and rag_high_good(0.35, 0.4, 0.3) == "Amber"
        and rag_high_good(0.1, 0.4, 0.3) == "Red"
    )
    assert (
        rag_low_good(0.05, 0.1, 0.25) == "Green"
        and rag_low_good(0.2, 0.1, 0.25) == "Amber"
        and rag_low_good(0.9, 0.1, 0.25) == "Red"
    )


def test_fairness_approval_ratio_on_known_example(cfg):
    # Group A: 80% approved, group B: 40% approved -> ratio 0.5, by construction
    n = 2000
    sex = np.array([1] * n + [2] * n)
    pd_scores = np.concatenate(
        [np.where(np.arange(n) < 0.8 * n, 0.1, 0.9), np.where(np.arange(n) < 0.4 * n, 0.1, 0.9)]
    )
    test = pd.DataFrame(
        {"SEX": sex, "AGE": 35, "default": np.random.default_rng(0).integers(0, 2, 2 * n)}
    )
    out = fairness(test, pd_scores, cutoff=0.3, cfg=cfg)
    assert out["SEX"]["approval_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert {g["group"] for g in out["SEX"]["groups"]} == {"Male", "Female"}
