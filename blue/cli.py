from __future__ import annotations

import argparse
import json
import os
import sys

from .audit import write_ai_usage
from .config import ROOT, load_config, resolve
from .inventory import build_records, scan
from .pipeline import PipelineError, run


def main(argv: list[str] | None = None) -> int:
    # keep the demo quiet and offline-friendly: no MLflow telemetry or upgrade hints
    """Command-line entry point: run, scan or ai-usage. Returns the exit code."""
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("DO_NOT_TRACK", "true")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["run"] + argv
    ap = argparse.ArgumentParser(prog="blue", description="Blue-chip Legacy Upgrade Engine")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument(
        "--mode",
        choices=["offline", "replay", "live"],
        default=None,
        help="offline: no LLM (rule-based stand-ins); replay: cached LLM responses; live: call the LLM API",
    )
    r.add_argument("--interactive", action="store_true", help="ask a human to approve each gate")
    r.add_argument(
        "--regenerate-fixtures",
        action="store_true",
        help="re-run the legacy R model (needs Rscript)",
    )
    r.add_argument("--no-mlflow", action="store_true")
    sub.add_parser("scan", help="print the model inventory only")
    sub.add_parser(
        "ai-usage",
        help="write AI_USAGE.md from the last run's audit log (then edit the TEAM TO CONFIRM parts)",
    )
    # allow --config before or after the subcommand
    args, rest = ap.parse_known_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "run":
        try:
            run(
                cfg,
                mode=args.mode,
                interactive=args.interactive,
                regenerate=args.regenerate_fixtures,
                use_mlflow=not args.no_mlflow,
            )
        except PipelineError as e:
            print(f"PIPELINE STOPPED: {e}", file=sys.stderr)
            return 2
        return 0
    if args.cmd == "scan":
        for rec in build_records(scan(cfg), cfg):
            print(f"{rec.model_id:12s} {rec.language:7s} tier {rec.risk_tier}  {rec.algorithm}")
        return 0
    if args.cmd == "ai-usage":
        log = resolve(cfg, "artifacts_dir") / "latest" / "audit.jsonl"
        events = (
            [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if log.exists()
            else []
        )
        write_ai_usage(events, ROOT / "AI_USAGE.md")
        print("Wrote AI_USAGE.md. Edit the sections marked TEAM TO CONFIRM before submitting.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
