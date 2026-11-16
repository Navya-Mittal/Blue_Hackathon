"""Validate: performance, stability, fairness, challenger. Every number is computed here, in code."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

from . import metrics as M
from .data import age_band
from .features import FACTORS, NUMERIC, design_matrix, prep_credit
from .schemas import Finding

SEX_LABELS = {1: "Male", 2: "Female"}


def rag_high_good(v: float, green_min: float, amber_min: float) -> str:
    """Green, Amber or Red for a metric where higher is better."""
    return "Green" if v >= green_min else ("Amber" if v >= amber_min else "Red")


def rag_low_good(v: float, green_max: float, amber_max: float) -> str:
    """Green, Amber or Red for a metric where lower is better."""
    return "Green" if v <= green_max else ("Amber" if v <= amber_max else "Red")


def performance(y: np.ndarray, p: np.ndarray, cfg: dict) -> dict[str, Any]:
    """Discrimination (AUC, Gini, KS with bootstrap intervals) and calibration numbers."""
    v = cfg["validation"]
    a = M.auc(y, p)
    g_lo, g_hi = M.bootstrap_ci(M.gini, y, p, v["bootstrap_samples"], seed=1)
    k_lo, k_hi = M.bootstrap_ci(M.ks, y, p, v["bootstrap_samples"], seed=2)
    cal = M.calibration_table(y, p, v["calibration_bins"])
    gaps = (cal["mean_pd"] - cal["observed"]).abs()
    worst = cal.loc[gaps.idxmax()]
    return {
        "worst_decile_gap": float(gaps.max()),
        "worst_decile_mean_pd": float(worst["mean_pd"]),
        "worst_decile_observed": float(worst["observed"]),
        "auc": a,
        "gini": 2 * a - 1,
        "gini_ci_low": g_lo,
        "gini_ci_high": g_hi,
        "ks": M.ks(y, p),
        "ks_ci_low": k_lo,
        "ks_ci_high": k_hi,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "mean_pd": float(np.mean(p)),
        "observed_default_rate": float(np.mean(y)),
        "calibration_gap": float(abs(np.mean(p) - np.mean(y))),
        "calibration_table": cal.round(6).to_dict(orient="records"),
    }


def stability(
    train: pd.DataFrame, test: pd.DataFrame, p_train: np.ndarray, p_test: np.ndarray, cfg: dict
) -> dict[str, Any]:
    """Score PSI and per-input CSI between the train and test samples."""
    tr, te = prep_credit(train), prep_credit(test)
    csi: dict[str, float] = {}
    for c in NUMERIC:
        csi[c] = M.psi(tr[c].to_numpy(float), te[c].to_numpy(float))
    for c in FACTORS:
        csi[c] = M.psi_categorical(tr[c], te[c])
    worst = max(csi, key=csi.get)
    return {
        "score_psi": M.psi(p_train, p_test),
        "csi": csi,
        "max_csi": csi[worst],
        "max_csi_feature": worst,
    }


def _groups(df: pd.DataFrame, attr: str, cfg: dict) -> pd.Series:
    if attr == "SEX":
        return df["SEX"].map(SEX_LABELS).fillna("Unknown")
    if attr == "AGE":
        return age_band(df["AGE"], cfg["data"]["age_bands"])
    raise ValueError(f"unsupported protected attribute {attr}")


def fairness(test: pd.DataFrame, p: np.ndarray, cutoff: float, cfg: dict) -> dict[str, Any]:
    """Rates per group for each protected attribute, plus approval ratio and equalised-odds gap."""
    y = test[cfg["data"]["target"]].to_numpy()
    pred = (p >= cutoff).astype(int)  # 1 = decline
    out: dict[str, Any] = {}
    rng = np.random.default_rng(7)
    for attr in cfg["data"]["protected"]:
        g = _groups(test, attr, cfg).to_numpy()
        mf = MetricFrame(
            metrics={
                "decline_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y,
            y_pred=pred,
            sensitive_features=g,
        )
        rows = []
        for grp, r in mf.by_group.iterrows():
            m = g == grp
            rows.append(
                {
                    "group": str(grp),
                    "n": int(m.sum()),
                    "approval_rate": float(1 - r["decline_rate"]),
                    "decline_rate": float(r["decline_rate"]),
                    "tpr_declined_defaulters": float(r["tpr"]),
                    "fpr_declined_good": float(r["fpr"]),
                    "mean_pd": float(np.mean(p[m])),
                    "observed_default_rate": float(np.mean(y[m])),
                }
            )
        appr = np.array([r["approval_rate"] for r in rows])
        ratio = float(appr.min() / appr.max())
        boots = []
        for _ in range(cfg["validation"]["bootstrap_samples"]):
            idx = rng.integers(0, len(y), len(y))
            a = [1 - pred[idx][g[idx] == grp].mean() for grp in np.unique(g)]
            boots.append(min(a) / max(a))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out[attr] = {
            "groups": rows,
            "approval_ratio": ratio,
            "approval_ratio_ci_low": float(lo),
            "approval_ratio_ci_high": float(hi),
            "equalized_odds_gap": float(equalized_odds_difference(y, pred, sensitive_features=g)),
            "approval_gap": float(appr.max() - appr.min()),
        }
    return out


def challenger(
    train: pd.DataFrame, test: pd.DataFrame, champion_auc: float, seed: int
) -> dict[str, Any]:
    """Fit a gradient-boosting challenger and compare its AUC with the champion's."""
    Xtr, Xte = design_matrix(train), design_matrix(test)
    m = HistGradientBoostingClassifier(
        random_state=seed, max_iter=200, learning_rate=0.05, max_depth=4
    )
    m.fit(Xtr, train["default"])
    p = m.predict_proba(Xte)[:, 1]
    y = test["default"].to_numpy()
    a = M.auc(y, p)
    return {
        "model": "histogram gradient boosting (200 iterations, depth 4)",
        "auc": a,
        "gini": 2 * a - 1,
        "ks": M.ks(y, p),
        "auc_gain_vs_champion": a - champion_auc,
    }


def make_findings(perf: dict, stab: dict, fair: dict, chal: dict, cfg: dict) -> list[Finding]:
    """Turn the numbers into Green/Amber/Red findings using the thresholds in config.yaml."""
    v = cfg["validation"]
    f: list[Finding] = []
    g = perf["gini"]
    f.append(
        Finding(
            area="Discrimination",
            metric="Gini (test)",
            value=f"{g:.3f} (95% CI {perf['gini_ci_low']:.3f} to {perf['gini_ci_high']:.3f})",
            criterion=f"Green >= {v['gini']['green_min']}, Amber >= {v['gini']['amber_min']}",
            rag=rag_high_good(g, v["gini"]["green_min"], v["gini"]["amber_min"]),
            message="Rank-ordering power of the PD score on held-out accounts.",
        )
    )
    cg = perf["calibration_gap"]
    f.append(
        Finding(
            area="Calibration",
            metric="|mean PD - observed default rate|",
            value=f"{cg:.4f}",
            criterion=f"Green <= {v['calibration_gap']['green_max']}, Amber <= {v['calibration_gap']['amber_max']}",
            rag=rag_low_good(
                cg, v["calibration_gap"]["green_max"], v["calibration_gap"]["amber_max"]
            ),
            message="Average predicted PD versus the observed default rate on the test sample.",
        )
    )
    wd = perf["worst_decile_gap"]
    f.append(
        Finding(
            area="Calibration",
            metric="Worst-decile |mean PD - observed|",
            value=f"{wd:.3f} (predicted {perf['worst_decile_mean_pd']:.3f} vs observed {perf['worst_decile_observed']:.3f})",
            criterion=f"Green <= {v['decile_gap']['green_max']}, Amber <= {v['decile_gap']['amber_max']}",
            rag=rag_low_good(wd, v["decile_gap"]["green_max"], v["decile_gap"]["amber_max"]),
            message="A good average can hide mis-calibration in the tails; the riskiest accounts drive decline decisions.",
        )
    )
    sp = stab["score_psi"]
    f.append(
        Finding(
            area="Stability",
            metric="Score PSI (train vs test)",
            value=f"{sp:.4f}",
            criterion=f"Green <= {v['psi']['green_max']}, Amber <= {v['psi']['amber_max']}",
            rag=rag_low_good(sp, v["psi"]["green_max"], v["psi"]["amber_max"]),
            message="Shift in the score distribution between development and test samples.",
        )
    )
    mc = stab["max_csi"]
    f.append(
        Finding(
            area="Stability",
            metric=f"Max CSI ({stab['max_csi_feature']})",
            value=f"{mc:.4f}",
            criterion=f"Green <= {v['csi']['green_max']}, Amber <= {v['csi']['amber_max']}",
            rag=rag_low_good(mc, v["csi"]["green_max"], v["csi"]["amber_max"]),
            message="Largest characteristic-level shift between train and test.",
        )
    )
    for attr, r in fair.items():
        ar = r["approval_ratio"]
        f.append(
            Finding(
                area=f"Fairness ({attr})",
                metric="Approval-rate ratio (min/max group)",
                value=f"{ar:.3f} (95% CI {r['approval_ratio_ci_low']:.3f} to {r['approval_ratio_ci_high']:.3f})",
                criterion=f"Green >= {v['approval_ratio']['green_min']}, Amber >= {v['approval_ratio']['amber_min']}",
                rag=rag_high_good(
                    ar, v["approval_ratio"]["green_min"], v["approval_ratio"]["amber_min"]
                ),
                message=f"Indicator only, not a legal test. {attr} is not a predictor in the model; this tests for residual disparity.",
            )
        )
        eo = r["equalized_odds_gap"]
        f.append(
            Finding(
                area=f"Fairness ({attr})",
                metric="Equalised-odds gap",
                value=f"{eo:.3f}",
                criterion=f"Green <= {v['equalized_odds_gap']['green_max']}, Amber <= {v['equalized_odds_gap']['amber_max']}",
                rag=rag_low_good(
                    eo, v["equalized_odds_gap"]["green_max"], v["equalized_odds_gap"]["amber_max"]
                ),
                message="Largest gap between groups in the rate of declining defaulters or good customers.",
            )
        )
    gain = chal["auc_gain_vs_champion"]
    f.append(
        Finding(
            area="Effective challenge",
            metric="Challenger AUC minus champion AUC",
            value=f"{gain:+.4f}",
            criterion=f"Green <= {v['challenger_auc_gain']['green_max']}, Amber <= {v['challenger_auc_gain']['amber_max']}",
            rag=rag_low_good(
                gain, v["challenger_auc_gain"]["green_max"], v["challenger_auc_gain"]["amber_max"]
            ),
            message="A materially better challenger suggests the champion's model form may be too simple.",
        )
    )
    return f


def outcome(findings: list[Finding]) -> str:
    """Draft outcome: any Red means escalate, any Amber approve with conditions, else approve."""
    rags = {x.rag for x in findings}
    if "Red" in rags:
        return "Escalate to the validation committee (draft)"
    if "Amber" in rags:
        return "Approve with conditions (draft)"
    return "Approve (draft)"


LIMITATIONS = [
    "The dataset has no time dimension, so there is no out-of-time sample. Back-testing uses one stratified random split.",
    "Data are a public Taiwanese credit-card portfolio from 2005; results say nothing about any real bank's portfolio.",
    "Fairness metrics use the two protected attributes present in the data (sex, age band) one at a time; intersectional groups are not tested.",
    "Fairness indicators are statistical diagnostics, not a legal or regulatory determination.",
    "RAG thresholds are illustrative and must be replaced by the model risk policy's thresholds.",
    "Drift monitoring uses a synthetic stress scenario with no simulated outcomes, so it demonstrates the mechanism, not real model decay.",
]


def run_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    p_train: np.ndarray,
    p_test: np.ndarray,
    cutoff: float,
    cfg: dict,
) -> dict[str, Any]:
    """Run all the validation tests and return one dictionary (saved as metrics.json)."""
    y = test[cfg["data"]["target"]].to_numpy()
    perf = performance(y, p_test, cfg)
    stab = stability(train, test, p_train, p_test, cfg)
    fair = fairness(test, p_test, cutoff, cfg)
    chal = challenger(train, test, perf["auc"], cfg["data"]["seed"])
    findings = make_findings(perf, stab, fair, chal, cfg)
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "cutoff": cutoff,
        "decline_rate_test": float(np.mean(p_test >= cutoff)),
        "performance": perf,
        "stability": stab,
        "fairness": fair,
        "challenger": chal,
        "findings": [x.model_dump() for x in findings],
        "outcome": outcome(findings),
        "limitations": LIMITATIONS,
    }
