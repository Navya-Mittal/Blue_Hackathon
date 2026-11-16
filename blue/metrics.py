"""Deterministic metric functions. All numbers in every report come from here, never from an LLM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def auc(y, s) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y, s))


def gini(y, s) -> float:
    """Gini coefficient: 2 x AUC - 1."""
    return 2 * auc(y, s) - 1


def ks(y, s) -> float:
    """KS statistic between the scores of defaulters and non-defaulters."""
    y = np.asarray(y)
    s = np.asarray(s)
    return float(ks_2samp(s[y == 1], s[y == 0]).statistic)


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with bins taken from the expected (reference) distribution."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant variable
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    """Population Stability Index for a categorical column."""
    cats = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    e = expected.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=eps)
    a = actual.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=eps)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Bin accounts by predicted PD; return mean predicted and observed default rate per bin."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    g = (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_pd=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    return g


def bootstrap_ci(fn, y, s, n: int, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap confidence interval for a metric `fn(y, s)`. Returns (low, high)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], s[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
