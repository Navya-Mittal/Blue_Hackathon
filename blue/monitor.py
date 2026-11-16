"""Monitor: drift detection on incoming batches and a re-validation trigger.

The batches come from a SYNTHETIC stress scenario (worsening repayment behaviour, shrinking limits). Outcomes are
not simulated, so this demonstrates the mechanism (detect, flag, raise a ticket), not real model decay.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M
from .data import drift_batch
from .features import FACTORS, NUMERIC, prep_credit
from .redevelop import run_candidate


def _status(psi_v: float, shift: float, mc: dict) -> str:
    if psi_v >= mc["psi_revalidate"] or abs(shift) >= mc["decline_rate_shift_revalidate"]:
        return "Re-validate"
    if psi_v >= mc["psi_watch"] or abs(shift) >= mc["decline_rate_shift_watch"]:
        return "Watch"
    return "OK"


def run_monitor(
    cfg: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    port_path: Path,
    train_csv: Path,
    p_train: np.ndarray,
    p_ref: np.ndarray,
    cutoff: float,
) -> dict[str, Any]:
    """Score synthetic drifted batches with the port, measure drift, and raise a ticket if needed."""
    mc = cfg["monitor"]
    rng = np.random.default_rng(cfg["data"]["seed"])
    batches, paths = [], {}
    with tempfile.TemporaryDirectory() as td:
        for i, sev in enumerate(mc["severities"][: mc["batches"]], 1):
            b = drift_batch(test, sev, rng, mc["batch_size"])
            p = Path(td) / f"batch_{i}.csv"
            b.to_csv(p, index=False)
            batches.append((i, sev, b))
            paths[f"batch_{i}"] = p
        res = run_candidate(port_path, train_csv, paths, cfg["parity"]["timeout_seconds"])
    if res["errors"]:
        raise RuntimeError(f"monitor scoring failed: {res['errors']}")

    ref_feat = prep_credit(train)
    ref_decline = float(np.mean(p_ref >= cutoff))
    rows = []
    for i, sev, b in batches:
        p = np.array(res["predictions"][f"batch_{i}"], dtype=float)
        feat = prep_credit(b)
        csi = {c: M.psi(ref_feat[c].to_numpy(float), feat[c].to_numpy(float)) for c in NUMERIC}
        csi.update({c: M.psi_categorical(ref_feat[c], feat[c]) for c in FACTORS})
        worst = max(csi, key=csi.get)
        dr = float(np.mean(p >= cutoff))
        score_psi = M.psi(p_train, p)
        rows.append(
            {
                "batch": i,
                "severity": sev,
                "score_psi": score_psi,
                "decline_rate": dr,
                "decline_rate_shift": dr - ref_decline,
                "max_csi": csi[worst],
                "max_csi_feature": worst,
                "status": _status(score_psi, dr - ref_decline, mc),
            }
        )
    order = {"OK": 0, "Watch": 1, "Re-validate": 2}
    overall = max((r["status"] for r in rows), key=order.get)
    psis = [r["score_psi"] for r in rows]
    slope = float(np.polyfit(np.arange(len(psis)), psis, 1)[0]) if len(psis) > 2 else 0.0
    breach = next((r["batch"] for r in rows if r["score_psi"] >= mc["psi_revalidate"]), None)
    first_flag = next((r for r in rows if r["status"] == "Re-validate"), None)
    trend = f"Trend (descriptive, not a forecast): score PSI rose about {slope:.3f} per batch. "
    if breach and first_flag:
        trend += (
            f"PSI alone first crossed the re-validation threshold in batch {breach}; "
            f"the first re-validation flag came in batch {first_flag['batch']}."
        )
    else:
        trend += "No re-validation flag in this window."
    ticket = None
    if overall == "Re-validate":
        first = first_flag
        ticket = {
            "model_id": "credit_pd",
            "reason": f"batch {first['batch']}: score PSI {first['score_psi']:.3f}, "
            f"decline-rate shift {first['decline_rate_shift']:+.3f}, largest characteristic shift in {first['max_csi_feature']}",
            "action": "Re-run independent validation on fresh labelled data before continued use.",
            "status": "OPEN (raised by the monitor; resolved by a human validator)",
        }
    return {
        "scenario": "SYNTHETIC stress scenario: repayment behaviour worsens and limits shrink across batches; "
        "outcomes are not simulated, so this demonstrates the monitoring mechanism, not real decay.",
        "thresholds": {
            k: mc[k]
            for k in (
                "psi_watch",
                "psi_revalidate",
                "decline_rate_shift_watch",
                "decline_rate_shift_revalidate",
            )
        },
        "reference_decline_rate": ref_decline,
        "batches": rows,
        "overall_status": overall,
        "psi_trend_per_batch": slope,
        "forecast": trend,
        "ticket": ticket,
    }
