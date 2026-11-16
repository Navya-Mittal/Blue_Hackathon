"""Orchestrator: runs the stages in order and times them, with an audit log and human gates.

Order in the code: Discover, Document, Redevelop, Validate, Monitor, Report.
"""

from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import docgen, fixtures
from .audit import Audit, write_ai_usage
from .config import resolve
from .data import prepare_splits
from .inventory import build_records, scan
from .llm import LLMClient, make_provider
from .monitor import run_monitor
from .redevelop import repair_loop, run_candidate
from .report import build_report, make_narrative, numeric_leaves
from .validate import run_validation


class PipelineError(RuntimeError):
    """Raised when a stage cannot continue (for example, parity not reached)."""


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def _log_to_mlflow(
    art: Path, model_id: str, mode: str, rec, parity, metrics: dict, n_disc: int, total: float
):
    """Record this run in a local MLflow store (sqlite file next to the artifacts folder)."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(art.parent / 'mlflow.db').as_posix()}")
    # keep MLflow's artifact root next to the db; its default is ./mlruns in the working directory
    if mlflow.get_experiment_by_name("blue") is None:
        mlflow.create_experiment(
            "blue", artifact_location=(art.parent / "mlartifacts").resolve().as_uri()
        )
    mlflow.set_experiment("blue")
    with mlflow.start_run(run_name=f"{model_id}-{mode}"):
        mlflow.set_tags(
            {
                "parity": "passed",
                "legacy_model": model_id,
                "risk_tier": str(rec.risk_tier),
                "llm_mode": mode,
                "draft_outcome": metrics["outcome"],
            }
        )
        mlflow.log_params(
            {
                "parity_iterations": len(parity.iterations),
                **{f"tol_{k}": v for k, v in parity.tolerances.items()},
            }
        )
        p = metrics["performance"]
        mlflow.log_metrics(
            {
                "auc": p["auc"],
                "gini": p["gini"],
                "ks": p["ks"],
                "score_psi": metrics["stability"]["score_psi"],
                "challenger_auc_gain": metrics["challenger"]["auc_gain_vs_champion"],
                "stale_doc_discrepancies": n_disc,
                "total_seconds": total,
            }
        )
        for name in (
            "report.html",
            "metrics.json",
            "parity.json",
            "monitor.json",
            "impact.json",
            "audit.jsonl",
        ):
            mlflow.log_artifact(str(art / name))
        mlflow.log_artifact(parity.final_code_path, artifact_path="port")


def run(
    cfg: dict,
    mode: str | None = None,
    interactive: bool = False,
    regenerate: bool = False,
    use_mlflow: bool = True,
    say=print,
    provider=None,
) -> dict[str, Any]:
    """Run the whole pipeline and return the results. Everything is written to artifacts/latest/."""
    t_all = time.perf_counter()
    timers: dict[str, float] = {}
    art = resolve(cfg, "artifacts_dir") / "latest"
    if art.exists():
        shutil.rmtree(art)
    (art / "docs").mkdir(parents=True)
    audit = Audit(art / "audit.jsonl", interactive=interactive)
    mode = mode or cfg["llm"]["mode"]
    if mode == "live" and provider is None:
        provider = make_provider(cfg["llm"])
    llm = LLMClient(
        mode,
        resolve(cfg, "cache_dir"),
        audit,
        provider,
        cfg["llm"]["temperature"],
        cfg["llm"]["max_tokens"],
    )
    audit.event("run_start", mode=mode, interactive=interactive)
    say(f"Blue pipeline starting (llm mode: {mode})")

    @contextmanager
    def stage(name: str):
        t0 = time.perf_counter()
        yield
        timers[name] = round(time.perf_counter() - t0, 2)
        say(f"  [{name}] {timers[name]:.1f}s")

    splits = prepare_splits(cfg)
    if regenerate:
        say("Regenerating reference fixtures by running the legacy R model...")
        fixtures.regenerate(cfg, "credit_pd", splits)

    legacy = resolve(cfg, "legacy_dir")

    # ---------------------------------------------------------------- 1. Discover
    with stage("discover"):
        facts = scan(cfg)
        records = build_records(facts, cfg)
        for r in records:
            audit.approve("risk tier", r.model_id, f"tier {r.risk_tier}: {r.tier_rationale[:100]}")
        _dump(art / "inventory.json", [r.model_dump() for r in records])
    say(
        f"  inventory: {len(records)} models, tiers "
        + ", ".join(f"{r.model_id}=T{r.risk_tier}" for r in records)
    )

    # ---------------------------------------------------------------- 2. Document
    docs, discrepancies = {}, {}
    with stage("document"):
        for f, rec in zip(facts, records):
            src = (legacy.parent / f.path).read_text(encoding="utf-8")
            doc = docgen.draft_doc(f, src, llm)
            docs[f.model_id] = doc
            (art / "docs" / f"{f.model_id}.md").write_text(
                docgen.render_markdown(doc), encoding="utf-8"
            )
            existing = legacy / f"{f.model_id}_model_doc.md"
            if existing.exists():
                d = docgen.stale_check(
                    existing.read_text(encoding="utf-8"), f, cfg["data"]["protected"]
                )
                discrepancies[f.model_id] = d
                _dump(
                    art / "docs" / f"{f.model_id}_discrepancies.json", [x.model_dump() for x in d]
                )
                rec.doc_status = (
                    f"outdated ({len(d)} discrepancies)" if d else "current (0 discrepancies)"
                )
            else:
                rec.doc_status = "missing (regenerated)"
            audit.approve(
                "documentation",
                f.model_id,
                f"{sum(1 for c in doc.claims if c.verified)}/{len(doc.claims)} claims verified; {rec.doc_status}",
            )
    n_disc = sum(len(v) for v in discrepancies.values())
    say(
        f"  documentation: {sum(len(d.claims) for d in docs.values())} claims drafted, "
        f"{sum(sum(1 for c in d.claims if c.verified) for d in docs.values())} verified; stale-doc discrepancies: {n_disc}"
    )

    # ---------------------------------------------------------------- 3. Redevelop
    model_id = cfg["redevelop"]["models"][0]
    f = next(x for x in facts if x.model_id == model_id)
    rec = next(r for r in records if r.model_id == model_id)
    train, test = pd.read_csv(splits["train"]), pd.read_csv(splits["test"])
    with stage("redevelop"):
        ref = fixtures.load_reference(cfg, model_id, splits)
        parity = repair_loop(
            llm=llm,
            facts=f,
            r_source=(legacy / f"{model_id}.R").read_text(encoding="utf-8"),
            train_csv=splits["train"],
            score_csvs={"test": splits["test"], "edge": splits["edge"], "train": splits["train"]},
            ref=ref,
            y_test=test[cfg["data"]["target"]].to_numpy(),
            cfg=cfg,
            out_dir=art / "port",
            audit=audit,
        )
        _dump(art / "parity.json", parity.model_dump())
        rec.redevelopment = f"ported to Python; parity {'PASSED' if parity.passed else 'FAILED'} in {len(parity.iterations)} iterations"
        audit.approve("redevelopment", model_id, rec.redevelopment)
    for it in parity.iterations:
        say(f"  draft {it.n}: {'PASS' if it.passed else 'FAIL'} {it.diagnosis or ''}")
    if not parity.passed:
        _dump(art / "inventory.json", [r.model_dump() for r in records])
        raise PipelineError(
            f"Parity not reached after {len(parity.iterations)} iterations; see {art / 'parity.json'}"
        )

    # ---------------------------------------------------------------- 4. Validate
    with stage("validate"):
        port = Path(parity.final_code_path)
        out = run_candidate(
            port,
            splits["train"],
            {"train": splits["train"], "test": splits["test"]},
            cfg["parity"]["timeout_seconds"],
        )
        if out["errors"]:
            raise PipelineError(f"port failed during validation scoring: {out['errors']}")
        p_train, p_test = np.array(out["predictions"]["train"]), np.array(
            out["predictions"]["test"]
        )
        metrics = run_validation(train, test, p_train, p_test, f.cutoff, cfg)
        _dump(art / "metrics.json", metrics)
        rec.validation_status = metrics["outcome"]
    reds = sum(1 for x in metrics["findings"] if x["rag"] == "Red")
    ambers = sum(1 for x in metrics["findings"] if x["rag"] == "Amber")
    say(
        f"  validation: Gini {metrics['performance']['gini']:.3f}; {reds} red, {ambers} amber findings; outcome: {metrics['outcome']}"
    )

    # ---------------------------------------------------------------- 5. Monitor
    with stage("monitor"):
        mon = run_monitor(cfg, train, test, port, splits["train"], p_train, p_test, f.cutoff)
        _dump(art / "monitor.json", mon)
        if mon["ticket"]:
            _dump(art / "revalidation_ticket.json", mon["ticket"])
            audit.event("revalidation_ticket", **mon["ticket"])
    say(f"  monitor: overall status {mon['overall_status']}")

    # ---------------------------------------------------------------- 6. Report
    with stage("report"):
        last = parity.iterations[-1]
        ctx_nums = {
            "parity_iterations": len(parity.iterations),
            "parity_checks": len(last.checks),
            "parity_checks_passed": sum(1 for c in last.checks if c.passed),
            "n_discrepancies": len(discrepancies.get(model_id, [])),
        }
        allowed = numeric_leaves(metrics) + numeric_leaves(ctx_nums)
        narrative, narrative_status = make_narrative(llm, metrics, ctx_nums, audit, allowed)
        arrays = {"y": test[cfg["data"]["target"]].to_numpy(), "p_train": p_train, "p_test": p_test}
        ctx = {
            "cfg": cfg,
            "record": rec.model_dump(),
            "records": [r.model_dump() for r in records],
            "discrepancies": [d.model_dump() for d in discrepancies.get(model_id, [])],
            "doc_verified": sum(1 for c in docs[model_id].claims if c.verified),
            "doc_total": len(docs[model_id].claims),
            "parity": parity,
            "r_version": ref["manifest"]["r_version"],
            "monitor": mon,
            "approvals": [e for e in audit.events if e["kind"] == "approval"],
        }
        html = build_report(ctx, metrics, arrays, narrative, narrative_status)
        (art / "report.html").write_text(html, encoding="utf-8")
        audit.approve("validation report", model_id, f"draft outcome: {metrics['outcome']}")
        # approvals list in the report was captured before this last gate; rebuild once so it is complete
        ctx["approvals"] = [e for e in audit.events if e["kind"] == "approval"]
        (art / "report.html").write_text(
            build_report(ctx, metrics, arrays, narrative, narrative_status), encoding="utf-8"
        )
    _dump(art / "inventory.json", [r.model_dump() for r in records])

    # ---------------------------------------------------------------- 7. Impact, usage declaration, MLflow
    total = round(time.perf_counter() - t_all, 2)
    impact = {
        "llm_mode": mode,
        "stage_seconds": timers,
        "total_seconds": total,
        "models_inventoried": len(records),
        "claims_drafted": sum(len(d.claims) for d in docs.values()),
        "claims_verified": sum(sum(1 for c in d.claims if c.verified) for d in docs.values()),
        "stale_doc_discrepancies": n_disc,
        "parity_iterations": len(parity.iterations),
        "parity_passed": parity.passed,
        "validation_findings": {
            "red": reds,
            "amber": ambers,
            "green": sum(1 for x in metrics["findings"] if x["rag"] == "Green"),
        },
        "monitor_status": mon["overall_status"],
        "manual_baseline_from_brief": "8 to 12 weeks per model validation (client figure in the hackathon brief)",
        "caveat": (
            "Measured on ONE small public-data model with automated steps only. Human review, governance and "
            "real-bank complexity are not included, so do not extrapolate this to the ~250-model estate."
        ),
    }
    _dump(art / "impact.json", impact)
    write_ai_usage(audit.events, art / "AI_USAGE.generated.md")

    if use_mlflow:
        try:
            _log_to_mlflow(art, model_id, mode, rec, parity, metrics, n_disc, total)
            audit.event("mlflow", status="logged", store="artifacts/mlflow.db")
        except Exception as e:  # tracking is optional; never fail the pipeline because of it
            audit.event("mlflow", status="skipped", reason=str(e)[:200])
            say(f"  (mlflow skipped: {str(e)[:80]})")

    say(f"Done in {total:.1f}s. Open {art / 'report.html'}")
    return {
        "artifacts": art,
        "impact": impact,
        "metrics": metrics,
        "parity": parity,
        "monitor": mon,
        "discrepancies": discrepancies,
        "records": records,
        "audit": audit,
    }
