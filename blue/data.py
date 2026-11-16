"""Data loading, deterministic splits, hand-built edge cases and synthetic drift batches.

All data is public (UCI "Default of Credit Card Clients", Yeh & Lien 2009) or synthetic.
No personal data is used anywhere in this repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve


def file_sha256(path: Path) -> str:
    """SHA-256 hash of a file (used to check fixtures belong to the right data)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(cfg: dict) -> pd.DataFrame:
    """Read the credit CSV and add a 1-based `id` column."""
    df = pd.read_csv(resolve(cfg, "data_csv"))
    df.insert(0, "id", np.arange(1, len(df) + 1))
    return df


def age_band(age: pd.Series, cuts: list[int]) -> pd.Series:
    """Turn ages into three bands using the two cut points, e.g. <30, 30-44, 45+."""
    lo, hi = cuts
    return pd.cut(
        age,
        bins=[-np.inf, lo - 1e-9, hi - 1e-9, np.inf],
        labels=[f"<{lo}", f"{lo}-{hi - 1}", f"{hi}+"],
    ).astype(str)


def edge_cases(train: pd.DataFrame) -> pd.DataFrame:
    """Hand-built scoring rows that stress translation: extremes, rare categories, missing values."""
    base = train.drop(columns=["id", "default"]).median().round()
    rows: list[dict] = []

    def add(**over):
        r = base.to_dict()
        r.update(over)
        rows.append(r)

    add()  # typical applicant
    add(LIMIT_BAL=1_000_000)  # very high limit
    add(LIMIT_BAL=10_000)  # very low limit
    add(PAY_1=-2, PAY_2=-2, PAY_3=-2)  # no consumption
    add(PAY_1=8, PAY_2=8, PAY_3=7)  # severe delinquency
    add(EDUCATION=0)  # undocumented category (collapsed by legacy rule)
    add(EDUCATION=5)
    add(EDUCATION=6)
    add(MARRIAGE=0)
    add(BILL_AMT1=-5_000)  # credit balance
    add(BILL_AMT1=1_500_000)  # huge bill
    add(PAY_AMT1=0, PAY_AMT2=0)  # no repayments
    add(PAY_AMT1=1_000_000, PAY_AMT2=1_000_000)  # huge repayments
    add(PAY_2=np.nan)  # missing numeric predictor
    add(EDUCATION=np.nan)  # missing categorical predictor
    out = pd.DataFrame(rows)
    out.insert(0, "id", np.arange(900_001, 900_001 + len(out)))
    out["default"] = 0
    return out


def prepare_splits(cfg: dict, force: bool = False) -> dict[str, Path]:
    """Create the train/test/edge CSVs deterministically.

    The committed R fixtures were generated from these exact files.
    """
    d = resolve(cfg, "splits_dir")
    d.mkdir(parents=True, exist_ok=True)
    paths = {k: d / f"{k}.csv" for k in ("train", "test", "edge")}
    if not force and all(p.exists() for p in paths.values()):
        return paths
    df = load_raw(cfg)
    y = df[cfg["data"]["target"]]
    train, test = train_test_split(
        df, test_size=cfg["data"]["test_size"], stratify=y, random_state=cfg["data"]["seed"]
    )
    train, test = train.sort_values("id"), test.sort_values("id")
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    edge_cases(train).to_csv(paths["edge"], index=False)
    manifest = {k: file_sha256(p) for k, p in paths.items()}
    (d / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return paths


def drift_batch(
    df: pd.DataFrame, severity: float, rng: np.random.Generator, n: int
) -> pd.DataFrame:
    """SYNTHETIC stress scenario: worse repayment behaviour and shrinking limits.

    Labels are not simulated.
    """
    b = df.sample(n=n, replace=False, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    if severity > 0:
        for c in ("PAY_1", "PAY_2", "PAY_3"):
            bump = rng.binomial(1, min(0.6 * severity, 1.0), size=len(b))
            b[c] = np.minimum(b[c] + bump * 2, 8)
        b["LIMIT_BAL"] = (b["LIMIT_BAL"] * (1 - 0.35 * severity)).round().clip(lower=10_000)
        b["BILL_AMT1"] = (b["BILL_AMT1"] * (1 + 0.5 * severity)).round()
    return b
