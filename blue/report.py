"""Report: one self-contained HTML file. Prose may be LLM-written, but every number must come from metrics.json."""

from __future__ import annotations

import base64
import io
import re
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jinja2 import Environment, select_autoescape  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .llm import LLMClient  # noqa: E402

NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")


# ------------------------------------------------------------------ numeric grounding
def numeric_leaves(obj: Any) -> list[float]:
    """Collect every finite number inside nested dicts and lists."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if np.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out += numeric_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += numeric_leaves(v)
    return out


def ungrounded_numbers(
    text: str, allowed: Iterable[float], extra_allowed: Iterable[float] = ()
) -> list[str]:
    """Numbers in `text` that cannot be matched (after rounding to the written precision) to any allowed value.
    A written percentage such as 81.5% matches 0.815 x 100."""
    vals = list(allowed)
    extra = set(float(x) for x in extra_allowed)
    bad: list[str] = []
    for tok in NUM_RE.findall(text):
        raw = tok.rstrip("%").replace(",", "")
        try:
            t = float(raw)
        except ValueError:
            continue
        if t in extra:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        ok = any(abs(round(abs(c), decimals) - abs(t)) < 1e-9 for v in vals for c in (v, v * 100.0))
        if not ok:
            bad.append(tok)
    return bad


# ------------------------------------------------------------------ narrative
def template_narrative(m: dict, ctx: dict) -> list[str]:
    """Fixed-template summary (three paragraphs) filled with the computed numbers; used with no LLM."""
    p, s, ch = m["performance"], m["stability"], m["challenger"]
    reds = [f for f in m["findings"] if f["rag"] == "Red"]
    ambers = [f for f in m["findings"] if f["rag"] == "Amber"]
    para1 = (
        f"The Python port reproduced the legacy R model after {ctx['parity_iterations']} iterations, passing "
        f"{ctx['parity_checks_passed']} of {ctx['parity_checks']} parity checks. "
        f"The documentation check found {ctx['n_discrepancies']} discrepancies between the existing "
        f"documentation and the code for this model."
    )
    para2 = (
        f"On {m['n_test']:,} held-out accounts the model has a Gini of {p['gini']:.3f} and a KS of {p['ks']:.3f}. "
        f"Its average predicted PD is {p['mean_pd']:.4f} against an observed default rate of {p['observed_default_rate']:.4f}, "
        f"but in the worst-calibrated decile it predicts {p['worst_decile_mean_pd']:.3f} against an observed {p['worst_decile_observed']:.3f}. "
        f"Score PSI between the development and test samples is {s['score_psi']:.4f}."
    )
    para3 = (
        f"A gradient-boosting challenger reaches an AUC of {ch['auc']:.3f}, which is {ch['auc_gain_vs_champion']:.3f} "
        f"above the legacy model's {p['auc']:.3f}. "
        f"The automated review raised {len(reds)} red and {len(ambers)} amber findings; "
        f"the draft outcome is: {m['outcome']}."
    )
    return [para1, para2, para3]


NARRATIVE_SYSTEM = (
    "You write the executive summary of a model validation report for a bank. You are given a JSON object of "
    "computed results. Write exactly three short paragraphs. Use ONLY numbers that appear in the JSON, copied to "
    "the same precision (a proportion may be written as a percentage). Do not state legal or regulatory "
    "conclusions, do not claim compliance, and do not introduce any other numbers."
)


def make_narrative(
    llm: LLMClient, m: dict, ctx: dict, audit, allowed: list[float]
) -> tuple[list[str], str]:
    """Get the summary. LLM text with numbers not in the metrics is rejected; the template is used."""
    import json

    tmpl = template_narrative(m, ctx)
    payload = json.dumps(
        {
            "results": {
                k: m[k] for k in ("n_test", "performance", "stability", "challenger", "outcome")
            },
            "context": ctx,
            "findings": m["findings"],
        },
        sort_keys=True,
        default=str,
    )
    res = llm.complete(
        "report_narrative", NARRATIVE_SYSTEM, payload, offline=lambda: "\n\n".join(tmpl)
    )
    paras = [x.strip() for x in res.text.split("\n\n") if x.strip()]
    bad = ungrounded_numbers(" ".join(paras), allowed)
    if bad:
        audit.event("narrative_rejected", source=res.source, ungrounded=bad)
        return (
            tmpl,
            f"LLM text rejected by the numeric grounding check (ungrounded: {', '.join(bad)}); template used",
        )
    return paras, (
        "rule-based template (no LLM)"
        if res.source == "offline-standin"
        else f"{res.source}; numeric grounding check passed"
    )


# ------------------------------------------------------------------ charts
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def charts(m: dict, arrays: dict) -> dict[str, str]:
    """Draw the report charts (ROC, calibration, scores, fairness) as base64 PNG strings."""
    out = {}
    y, pt, ptr = arrays["y"], arrays["p_test"], arrays["p_train"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    fpr, tpr, _ = roc_curve(y, pt)
    ax.plot(fpr, tpr, color="#1f4e9c", lw=2, label=f"AUC {m['performance']['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    ax.legend(loc="lower right")
    out["roc"] = _png(fig)

    cal = m["performance"]["calibration_table"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([r["mean_pd"] for r in cal], [r["observed"] for r in cal], "o-", color="#1f4e9c")
    top = max(max(r["mean_pd"], r["observed"]) for r in cal) * 1.05
    ax.plot([0, top], [0, top], "--", color="#999")
    ax.set(
        xlabel="Mean predicted PD (decile)",
        ylabel="Observed default rate",
        title="Calibration (test)",
    )
    out["calibration"] = _png(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    bins = np.linspace(0, 1, 31)
    ax.hist(ptr, bins=bins, alpha=0.55, density=True, label="train", color="#1f4e9c")
    ax.hist(pt, bins=bins, alpha=0.55, density=True, label="test", color="#e08a1e")
    ax.set(
        xlabel="Predicted PD",
        ylabel="Density",
        title=f"Score distribution (PSI {m['stability']['score_psi']:.4f})",
    )
    ax.legend()
    out["scores"] = _png(fig)

    attrs = list(m["fairness"])
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.2 * len(attrs) / 1.3, 3.6), squeeze=False)
    for ax, a in zip(axes[0], attrs):
        rows = m["fairness"][a]["groups"]
        ax.bar([r["group"] for r in rows], [r["approval_rate"] for r in rows], color="#2f855a")
        ax.set(ylim=(0, 1), title=f"Approval rate by {a}")
        ax.axhline(
            max(r["approval_rate"] for r in rows) * 0.8,
            ls="--",
            color="#c53030",
            lw=1,
            label="80% of highest group",
        )
        ax.legend(fontsize=7, loc="lower right")
    out["fairness"] = _png(fig)
    return out


# ------------------------------------------------------------------ HTML
TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
 body{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:1040px;margin:24px auto;padding:0 18px;color:#1d2433;line-height:1.45}
 h1{margin-bottom:2px} h2{border-bottom:2px solid #1f4e9c;padding-bottom:4px;margin-top:34px} h3{margin-bottom:4px}
 .sub{color:#5b6475;margin-top:0} .banner{background:#fff8e1;border:1px solid #f0d58a;padding:10px 14px;border-radius:6px;font-size:.92em}
 table{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:.92em} th,td{border:1px solid #d9dee8;padding:5px 8px;text-align:left;vertical-align:top}
 th{background:#eef2f9} .chip{display:inline-block;padding:1px 9px;border-radius:10px;color:#fff;font-weight:600;font-size:.85em}
 .Green{background:#1a7f37}.Amber{background:#b7791f}.Red{background:#c53030}.pass{color:#1a7f37;font-weight:600}.fail{color:#c53030;font-weight:600}
 .grid{display:flex;flex-wrap:wrap;gap:10px}.grid img{max-width:49%} .small{font-size:.85em;color:#5b6475} code{background:#f2f4f8;padding:1px 4px;border-radius:3px}
 .outcome{font-size:1.1em;padding:10px 14px;border-left:5px solid #1f4e9c;background:#eef2f9;margin:12px 0}
</style></head><body>
<h1>{{ title }}</h1>
<p class="sub">Team {{ team }} &middot; model <b>{{ record.name }}</b> ({{ record.model_id }}) &middot; generated {{ generated }}</p>
<p class="banner"><b>Draft produced by an automated pipeline.</b> It requires review and sign-off by an independent human validator.
RAG thresholds are illustrative. Data are public or synthetic; no personal data.</p>

<h2>1. Executive summary</h2>
{% for p in narrative %}<p>{{ p }}</p>{% endfor %}
<div class="outcome"><b>Draft outcome:</b> {{ m.outcome }}</div>
<p class="small">Summary text source: {{ narrative_status }}.</p>

<h2>2. Model identification and risk tier <span class="small">(SS1/23 Principle 1)</span></h2>
<table><tr><th>Model</th><th>Language</th><th>Algorithm</th><th>Predictors</th><th>Materiality</th><th>Tier</th><th>Documentation</th><th>Redevelopment</th></tr>
{% for r in records %}<tr><td>{{ r.model_id }}</td><td>{{ r.language }}</td><td>{{ r.algorithm }}</td><td>{{ r.n_predictors }}</td><td>{{ r.materiality }}</td><td>{{ r.risk_tier }}</td><td>{{ r.doc_status }}</td><td>{{ r.redevelopment }}</td></tr>{% endfor %}</table>
<p class="small"><b>Tier rationale ({{ record.model_id }}):</b> {{ record.tier_rationale }} Materiality is a business assumption supplied in config.yaml; it cannot be inferred from code.</p>

<h2>3. Development, documentation and redevelopment <span class="small">(SS1/23 Principle 3)</span></h2>
<h3>Documentation currency</h3>
{% if discrepancies %}<p>The existing documentation disagrees with the code on <b>{{ discrepancies|length }}</b> points:</p>
<table><tr><th>Field</th><th>Documentation says</th><th>Code does</th><th>Severity</th><th>Code evidence</th></tr>
{% for d in discrepancies %}<tr><td>{{ d.field }}</td><td>{{ d.documented }}</td><td>{{ d.actual }}</td><td>{{ d.severity }}</td>
<td>{% for e in d.evidence %}<code>{{ e.file }}:L{{ e.start_line }}{% if e.end_line != e.start_line %}-{{ e.end_line }}{% endif %}</code> {% endfor %}</td></tr>{% endfor %}</table>
{% else %}<p>No discrepancies found between the existing documentation and the code.</p>{% endif %}
<p>Regenerated documentation: {{ doc_verified }} of {{ doc_total }} claims verified against quoted source lines (<code>docs/{{ record.model_id }}.md</code>).</p>
<h3>Parity proof: legacy R vs Python port</h3>
<table><tr><th>Draft</th><th>Produced by</th><th>Result</th><th>Diagnosis codes</th><th>Checks passed</th></tr>
{% for it in parity.iterations %}<tr><td>{{ it.n }}</td><td>{{ it.source }}</td><td class="{{ 'pass' if it.passed else 'fail' }}">{{ 'PASS' if it.passed else 'FAIL' }}</td>
<td>{{ it.diagnosis|join(', ') or '-' }}</td><td>{{ it.checks|selectattr('passed')|list|length }} / {{ it.checks|length }}</td></tr>{% endfor %}</table>
<table><tr><th>Final-draft check</th><th>Result</th><th>Value</th><th>Tolerance</th><th>Detail</th></tr>
{% for c in parity.iterations[-1].checks %}<tr><td>{{ c.name }}</td><td class="{{ 'pass' if c.passed else 'fail' }}">{{ 'PASS' if c.passed else 'FAIL' }}</td>
<td>{{ '%.3g'|format(c.value) if c.value is not none else '' }}</td><td>{{ '%.0e'|format(c.tolerance) if c.tolerance is not none else '' }}</td><td>{{ c.detail }}</td></tr>{% endfor %}</table>
<p class="small">Reference outputs come from a real run of the legacy R script ({{ r_version }}). Edge cases include extreme values, undocumented categories and missing values.</p>

<h2>4. Independent validation <span class="small">(SS1/23 Principle 4)</span></h2>
<table><tr><th>Area</th><th>Metric</th><th>Value</th><th>Criterion (illustrative)</th><th>Rating</th></tr>
{% for f in m.findings %}<tr><td>{{ f.area }}</td><td>{{ f.metric }}</td><td>{{ f.value }}</td><td>{{ f.criterion }}</td><td><span class="chip {{ f.rag }}">{{ f.rag }}</span></td></tr>{% endfor %}</table>
<div class="grid"><img src="{{ ch.roc }}"><img src="{{ ch.calibration }}"><img src="{{ ch.scores }}"><img src="{{ ch.fairness }}"></div>
<h3>Fairness by group <span class="small">(decision: decline if PD &ge; {{ '%.2f'|format(m.cutoff) }})</span></h3>
{% for attr, r in m.fairness.items() %}<p><b>{{ attr }}</b> &middot; approval ratio {{ '%.3f'|format(r.approval_ratio) }} &middot; equalised-odds gap {{ '%.3f'|format(r.equalized_odds_gap) }}</p>
<table><tr><th>Group</th><th>n</th><th>Approval rate</th><th>Declined defaulters (TPR)</th><th>Declined good customers (FPR)</th><th>Mean PD</th><th>Observed default rate</th></tr>
{% for g in r.groups %}<tr><td>{{ g.group }}</td><td>{{ g.n }}</td><td>{{ '%.3f'|format(g.approval_rate) }}</td><td>{{ '%.3f'|format(g.tpr_declined_defaulters) }}</td><td>{{ '%.3f'|format(g.fpr_declined_good) }}</td><td>{{ '%.3f'|format(g.mean_pd) }}</td><td>{{ '%.3f'|format(g.observed_default_rate) }}</td></tr>{% endfor %}</table>{% endfor %}
<h3>Challenger</h3>
<p>{{ m.challenger.model }}: AUC {{ '%.3f'|format(m.challenger.auc) }}, Gini {{ '%.3f'|format(m.challenger.gini) }}, KS {{ '%.3f'|format(m.challenger.ks) }} (champion AUC {{ '%.3f'|format(m.performance.auc) }}).</p>

{% if monitor %}<h2>5. Ongoing monitoring <span class="small">(SS1/23 Principle 4, monitoring)</span></h2>
<p class="small">{{ monitor.scenario }}</p>
<table><tr><th>Batch</th><th>Drift severity</th><th>Score PSI</th><th>Decline rate</th><th>Shift vs reference</th><th>Max CSI</th><th>Status</th></tr>
{% for b in monitor.batches %}<tr><td>{{ b.batch }}</td><td>{{ b.severity }}</td><td>{{ '%.4f'|format(b.score_psi) }}</td><td>{{ '%.3f'|format(b.decline_rate) }}</td><td>{{ '%+.3f'|format(b.decline_rate_shift) }}</td><td>{{ '%.3f'|format(b.max_csi) }} ({{ b.max_csi_feature }})</td><td>{{ b.status }}</td></tr>{% endfor %}</table>
<p><b>Overall:</b> {{ monitor.overall_status }}. {{ monitor.forecast }}</p>{% endif %}

<h2>6. Limitations</h2><ul>{% for l in m.limitations %}<li>{{ l }}</li>{% endfor %}</ul>

<h2>7. Human sign-off and audit</h2>
<table><tr><th>Stage</th><th>Model</th><th>Decision</th><th>Recorded reviewer</th></tr>
{% for a in approvals %}<tr><td>{{ a.stage }}</td><td>{{ a.model_id }}</td><td>{{ a.decision }}</td><td>{{ a.reviewer }}</td></tr>{% endfor %}</table>
<p class="small">"auto-demo" means the demo auto-approved the gate; it is not a human decision. Run with <code>--interactive</code> to record human approvals.</p>
<table><tr><th>Independent validator</th><td style="width:60%">Name / date / signature: ______________________</td></tr></table>
</body></html>"""


def build_report(
    ctx: dict, m: dict, arrays: dict, narrative: list[str], narrative_status: str
) -> str:
    """Fill the HTML template and return the finished report as one string."""
    env = Environment(autoescape=select_autoescape(default=True))
    return env.from_string(TEMPLATE).render(
        title=ctx["cfg"]["report"]["title"],
        team=ctx["cfg"]["report"]["team"],
        generated=time.strftime("%Y-%m-%d %H:%M"),
        record=ctx["record"],
        records=ctx["records"],
        m=m,
        discrepancies=ctx["discrepancies"],
        doc_verified=ctx["doc_verified"],
        doc_total=ctx["doc_total"],
        parity=ctx["parity"],
        r_version=ctx["r_version"],
        ch=charts(m, arrays),
        monitor=ctx.get("monitor"),
        approvals=ctx["approvals"],
        narrative=narrative,
        narrative_status=narrative_status,
    )
