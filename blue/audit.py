"""Audit log, human-approval gates and the auto-generated AI-usage declaration.

Honesty rule: in demo runs approvals are recorded as reviewer="auto-demo", never as a human.
Run with --interactive to record real human decisions.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional


class Audit:
    """Writes every event to a JSONL log and handles the human-approval gates."""

    def __init__(
        self,
        path: Path,
        interactive: bool = False,
        reviewer: str = "auto-demo",
        prompt: Optional[Callable[[str], str]] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")  # fresh log per run
        self.events: list[dict[str, Any]] = []
        self.interactive = interactive
        self.reviewer = reviewer
        self._prompt = prompt or input

    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Add a timestamped event to the log (in memory and on disk)."""
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **fields}
        self.events.append(rec)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def approve(self, stage: str, model_id: str, summary: str) -> bool:
        """Human-approval gate. Non-interactive runs auto-approve and say so in the log."""
        if self.interactive:
            ans = (
                self._prompt(f"[gate] {stage} / {model_id}: {summary}\n  Approve? [y/N] ")
                .strip()
                .lower()
            )
            decision = ans in ("y", "yes")
            reviewer = self.reviewer if self.reviewer != "auto-demo" else "interactive-user"
        else:
            decision, reviewer = True, "auto-demo"
        self.event(
            "approval",
            stage=stage,
            model_id=model_id,
            decision="approved" if decision else "rejected",
            reviewer=reviewer,
            summary=summary,
        )
        return decision


AI_USAGE_TEMPLATE = """# AI usage declaration

The hackathon rules require teams to declare AI tools and any existing code used.
This file is generated from the pipeline audit log (`artifacts/latest/audit.jsonl`), plus the
static statements below. **The team must review and edit the sections marked TEAM TO CONFIRM.**

## 1. AI inside the pipeline (generated from the audit log of the last run)

{pipeline_table}

Notes
- `offline-standin` means NO LLM was called: a deterministic, rule-based stand-in produced that step.
  It exists so the demo runs without an API key and so tests are reproducible.
- `cache` means a previously recorded LLM response was replayed (replay mode).
- `live:<model>` means a real LLM API call was made and its response recorded in `llm_cache/`.
- Every number in generated reports comes from code (`metrics.json`); a grounding check rejects any
  LLM-written number that is not in that file.
- Legacy source code is passed to LLM prompts as quoted data, never as instructions.

## 2. AI coding assistants used to build this repository  (TEAM TO CONFIRM)

- The initial repository scaffold (module layout, pipeline code, tests, README) was drafted with
  Claude (Anthropic) acting as a coding assistant. Team members then reviewed, ran, and modified it.
  Describe here what your team changed or wrote yourselves: ____________________

## 3. Existing code and third-party material

- Open-source libraries: pandas, numpy, scipy, scikit-learn, statsmodels, fairlearn, pydantic,
  jinja2, matplotlib, PyYAML, mlflow (optional), pytest. See `requirements.txt`.
- Data: "Default of Credit Card Clients" (Yeh & Lien, 2009; UCI Machine Learning Repository), public
  data, no personal data. See `data/README.md`.
- Legacy sample models in `legacy_models/` were written for this project as synthetic examples.
- Prior team code reused: ____________________  (TEAM TO CONFIRM; write "none" if none)
"""


def write_ai_usage(events: list[dict[str, Any]], path: Path) -> str:
    """Fill in the AI usage declaration from the audit events and write it to `path`."""
    counts: Counter = Counter()
    for e in events:
        if e.get("kind") == "llm_call":
            counts[(e.get("task", "?"), e.get("source", "?"))] += 1
    if counts:
        rows = ["| Pipeline task | Source | Calls |", "|---|---|---|"]
        rows += [f"| {t} | {s} | {n} |" for (t, s), n in sorted(counts.items())]
        table = "\n".join(rows)
    else:
        table = "_No pipeline steps ran in the last run._"
    text = AI_USAGE_TEMPLATE.format(pipeline_table=table)
    Path(path).write_text(text, encoding="utf-8")
    return text
