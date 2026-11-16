"""Discover: scan legacy models, extract facts deterministically, assign a transparent risk tier."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .config import resolve
from .schemas import Evidence, ModelFacts, ModelRecord

PY_MODELS = {
    "GradientBoostingClassifier": "gradient boosting",
    "HistGradientBoostingClassifier": "gradient boosting",
    "XGBClassifier": "gradient boosting",
    "LGBMClassifier": "gradient boosting",
    "RandomForestClassifier": "random forest",
    "LogisticRegression": "logistic regression",
    "LinearRegression": "linear regression",
}
FAMILY_CLASS = {
    "binomial": "logistic regression",
    "gaussian": "linear regression",
    "poisson": "poisson regression",
}


# ---------------------------------------------------------------- helpers
def evidence_for(rel: str, lines: list[str], start: int, end: int) -> Evidence:
    """Build an Evidence record quoting source lines start..end (quote cut to 240 characters)."""
    quote = " ".join(line.strip() for line in lines[start - 1 : end]).strip()
    return Evidence(file=rel, start_line=start, end_line=end, quote=quote[:240])


def count_code_lines(lines: list[str]) -> int:
    """Lines of code: not blank and not a comment."""
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def split_top_level(s: str, sep: str) -> list[str]:
    """Split `s` on `sep`, ignoring separators that sit inside brackets."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return [x for x in out if x]


def balanced_end(text: str, open_idx: int) -> int:
    """Index of the ')' that closes the '(' at `open_idx`."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced parentheses")


def call_args(text: str, fname: str) -> list[list[str]]:
    """Top-level argument lists of every call to `fname(...)` in R source."""
    out = []
    for m in re.finditer(re.escape(fname) + r"\s*\(", text):
        o = text.index("(", m.start())
        out.append(split_top_level(text[o + 1 : balanced_end(text, o)], ","))
    return out


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


# ---------------------------------------------------------------- R parser
def parse_r(rel: str, text: str) -> ModelFacts:
    """Read an R model file (glm or lm) and pull out its facts and evidence lines."""
    lines = text.splitlines()
    model_id = Path(rel).stem
    ev: dict[str, list[Evidence]] = {}
    m = re.search(r"\b(glm|lm)\s*\(", text)
    if not m:
        raise ValueError(f"{rel}: no glm()/lm() call found")
    open_idx = text.index("(", m.start())
    end_idx = balanced_end(text, open_idx)
    call = text[m.start() : end_idx + 1]
    s_line, e_line = _line_of(text, m.start()), _line_of(text, end_idx)
    ev["algorithm"] = [evidence_for(rel, lines, s_line, e_line)]
    ev["variables"] = ev["algorithm"]
    inner = text[open_idx + 1 : end_idx]
    formula = split_top_level(inner, ",")[0]
    lhs, rhs = [x.strip() for x in formula.split("~", 1)]
    predictors = split_top_level(rhs, "+")

    fm = re.search(r"family\s*=\s*(\w+)\s*(?:\(\s*(?:link\s*=\s*)?\"?(\w+)\"?\s*\))?", call)
    family = fm.group(1).lower() if fm else "gaussian"
    link = fm.group(2) if fm and fm.group(2) else ("identity" if family == "gaussian" else None)
    algo_class = FAMILY_CLASS.get(family, f"glm ({family})")
    desc = f"{algo_class.capitalize()} (GLM, {family} family" + (f", {link} link)" if link else ")")

    cutoff = None
    for i, line in enumerate(lines, 1):
        cm = re.match(r"\s*(CUTOFF|THRESHOLD|cutoff|threshold)\s*<-\s*([0-9.]+)", line)
        if cm:
            cutoff = float(cm.group(2))
            ev["cutoff"] = [evidence_for(rel, lines, i, i)]
            break
    treat = [evidence_for(rel, lines, i, i) for i, line in enumerate(lines, 1) if "ifelse(" in line]
    if treat:
        ev["data_treatment"] = treat
    reg = bool(re.search(r"glmnet|penalized|lasso|ridge", text, re.I))
    inputs = [re.sub(r"\s+", " ", a[0]).strip() for a in call_args(text, "read.csv") if a]
    outputs = [
        re.sub(r"\s+", " ", a[1]).strip() for a in call_args(text, "write.csv") if len(a) > 1
    ]
    purpose_line = lines[0].lstrip("# ").strip() if lines else ""
    ev["purpose"] = [evidence_for(rel, lines, 1, 1)]
    ev["outputs"] = [
        evidence_for(rel, lines, _line_of(text, mm.start()), _line_of(text, mm.start()))
        for mm in re.finditer(r"write\.csv\(", text)
    ][:2]
    return ModelFacts(
        model_id=model_id,
        path=rel,
        language="R",
        algorithm_class=algo_class,
        algorithm_desc=desc,
        family=family,
        link=link,
        target=lhs,
        predictors=predictors,
        n_predictors=len(predictors),
        cutoff=cutoff,
        regularisation=reg,
        data_inputs=inputs,
        outputs=outputs,
        loc=count_code_lines(lines),
        purpose_line=purpose_line,
        evidence=ev,
    )


# ---------------------------------------------------------------- Python parser
def parse_py(rel: str, text: str) -> ModelFacts:
    """Read a Python model file (scikit-learn style) and extract the same facts."""
    lines = text.splitlines()
    tree = ast.parse(text)
    model_id = Path(rel).stem
    consts: dict[str, tuple[object, int, int]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                consts[node.targets[0].id] = (
                    ast.literal_eval(node.value),
                    node.lineno,
                    node.end_lineno or node.lineno,
                )
            except (ValueError, SyntaxError):
                pass
    ev: dict[str, list[Evidence]] = {}
    algo_class, call_node = None, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name in PY_MODELS:
                algo_class, call_node = PY_MODELS[name], node
                ev["algorithm"] = [
                    evidence_for(rel, lines, node.lineno, node.end_lineno or node.lineno)
                ]
                break
    if algo_class is None:
        raise ValueError(f"{rel}: no known model class found")
    kw = {k.arg: ast.unparse(k.value) for k in call_node.keywords if k.arg}
    desc = f"{algo_class.capitalize()} (" + ", ".join(f"{k}={v}" for k, v in kw.items()) + ")"
    feats = consts.get("FEATURES", ([], 0, 0))
    target = consts.get("TARGET", (None, 0, 0))[0]
    if feats[1]:
        ev["variables"] = [evidence_for(rel, lines, feats[1], feats[2])]
    cutoff = None
    for key in ("THRESHOLD", "CUTOFF"):
        if key in consts:
            cutoff = float(consts[key][0])
            ev["cutoff"] = [evidence_for(rel, lines, consts[key][1], consts[key][2])]
    reads, writes = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "read_csv" and node.args:
                reads.append(ast.unparse(node.args[0]))
            if node.func.attr == "to_csv" and node.args:
                writes.append(ast.unparse(node.args[0]))
    doc = ast.get_docstring(tree) or ""
    purpose = doc.splitlines()[0] if doc else ""
    ev["purpose"] = [evidence_for(rel, lines, 1, 2)] if doc else []
    reg = None
    if algo_class == "logistic regression":
        reg = kw.get("penalty", "'l2'") not in ("None", "'none'")
    return ModelFacts(
        model_id=model_id,
        path=rel,
        language="Python",
        algorithm_class=algo_class,
        algorithm_desc=desc,
        target=target,
        predictors=list(feats[0]),
        n_predictors=len(feats[0]),
        cutoff=cutoff,
        regularisation=reg,
        data_inputs=reads,
        outputs=writes,
        loc=count_code_lines(lines),
        purpose_line=purpose,
        evidence=ev,
    )


def scan(cfg: dict) -> list[ModelFacts]:
    """Parse every .R and .py file in the legacy models folder."""
    root = resolve(cfg, "legacy_dir")
    facts: list[ModelFacts] = []
    for p in sorted(root.iterdir()):
        rel = f"{root.name}/{p.name}"
        if p.suffix == ".R":
            facts.append(parse_r(rel, p.read_text(encoding="utf-8")))
        elif p.suffix == ".py":
            facts.append(parse_py(rel, p.read_text(encoding="utf-8")))
    return facts


# ---------------------------------------------------------------- tiering
def tier(f: ModelFacts, cfg: dict) -> tuple[int, int, str, str]:
    """Work out a model's risk tier. Returns (tier, complexity, materiality, rationale)."""
    t = cfg["tiering"]
    materiality = t["materiality"].get(f.model_id, "medium")
    cx = t["complexity_weights"].get(f.algorithm_class, 2)
    reasons = [f"algorithm weight {cx} ({f.algorithm_class})"]
    if f.n_predictors >= t["many_predictors"]:
        cx += 1
        reasons.append(f"{f.n_predictors} predictors (>= {t['many_predictors']})")
    if f.loc >= t["long_code_loc"]:
        cx += 1
        reasons.append(f"{f.loc} lines of code (>= {t['long_code_loc']})")
    if materiality == "high":
        risk_tier = 1
    elif materiality == "medium" or cx >= 3:
        risk_tier = 2
    else:
        risk_tier = 3
    rationale = (
        f"Materiality '{materiality}' (assumption from config) and complexity score {cx} "
        f"[{'; '.join(reasons)}] give tier {risk_tier}. Rule: high materiality -> tier 1; "
        f"medium materiality or complexity >= 3 -> tier 2; otherwise tier 3."
    )
    return risk_tier, cx, materiality, rationale


def build_records(facts: list[ModelFacts], cfg: dict) -> list[ModelRecord]:
    """Turn the extracted facts into inventory records, one per model."""
    recs = []
    for f in facts:
        risk_tier, cx, materiality, why = tier(f, cfg)
        recs.append(
            ModelRecord(
                model_id=f.model_id,
                name=f.model_id.replace("_", " ").title(),
                path=f.path,
                language=f.language,
                algorithm=f.algorithm_desc,
                target=f.target,
                n_predictors=f.n_predictors,
                loc=f.loc,
                materiality=materiality,
                complexity_score=cx,
                risk_tier=risk_tier,
                tier_rationale=why,
                redevelopment=(
                    "in scope"
                    if f.model_id in cfg["redevelop"]["models"]
                    else "inventory + documentation only"
                ),
            )
        )
    return recs
