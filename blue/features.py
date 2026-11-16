"""Reference feature preparation mirroring the legacy credit_pd.R business rules.

Used by the challenger model and the monitor. The *ported* model carries its own copy of this logic
(that is exactly what the parity harness verifies).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["log_limit", "PAY_1", "PAY_2", "PAY_3", "bill1_k", "payamt1_k", "payamt2_k"]
FACTORS = {"EDUCATION": [1, 2, 3, 4], "MARRIAGE": [1, 2, 3]}


def prep_credit(d: pd.DataFrame) -> pd.DataFrame:
    """Apply the legacy data rules: recode odd categories, add log limit and scaled amounts."""
    d = d.copy()
    d["EDUCATION"] = d["EDUCATION"].where(~d["EDUCATION"].isin([0, 5, 6]), 4)
    d["MARRIAGE"] = d["MARRIAGE"].where(d["MARRIAGE"] != 0, 3)
    d["log_limit"] = np.log(d["LIMIT_BAL"])
    d["bill1_k"] = d["BILL_AMT1"] / 1000
    d["payamt1_k"] = d["PAY_AMT1"] / 1000
    d["payamt2_k"] = d["PAY_AMT2"] / 1000
    return d


def design_matrix(d: pd.DataFrame) -> pd.DataFrame:
    """Numeric design matrix (treatment-coded factors, reference = first level)."""
    p = prep_credit(d)
    X = p[NUMERIC].astype(float).copy()
    for col, levels in FACTORS.items():
        for lv in levels[1:]:
            X[f"{col}{lv}"] = (p[col] == lv).astype(float)
    return X
