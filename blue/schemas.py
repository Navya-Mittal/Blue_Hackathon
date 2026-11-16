"""Typed contracts passed between pipeline stages (pydantic v2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    # fields named model_id would otherwise warn on pydantic < 2.10 ("model_" is a protected namespace there)
    model_config = ConfigDict(protected_namespaces=())


class Evidence(_Base):
    """A quoted piece of source code with its file and line numbers."""

    file: str
    start_line: int
    end_line: int
    quote: str


class ModelFacts(_Base):
    """Facts extracted deterministically from legacy source code (no LLM involved)."""

    model_id: str
    path: str
    language: str  # "R" | "Python"
    algorithm_class: str  # e.g. "logistic regression"
    algorithm_desc: str
    family: Optional[str] = None
    link: Optional[str] = None
    target: Optional[str] = None
    predictors: list[str] = Field(default_factory=list)
    n_predictors: int = 0
    cutoff: Optional[float] = None
    regularisation: Optional[bool] = None  # None = not applicable / unknown
    data_inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    loc: int = 0
    purpose_line: str = ""
    evidence: dict[str, list[Evidence]] = Field(default_factory=dict)


class ModelRecord(_Base):
    """One row of the model inventory."""

    model_id: str
    name: str
    path: str
    language: str
    algorithm: str
    target: Optional[str]
    n_predictors: int
    loc: int
    materiality: str
    materiality_source: str = (
        "assumption supplied in config.yaml (business input, not inferable from code)"
    )
    complexity_score: int
    risk_tier: int
    tier_rationale: str
    doc_status: str = "unknown"  # missing | current | outdated | generated
    redevelopment: str = "not in scope"
    validation_status: str = "not run"


class Claim(_Base):
    """One statement in the documentation, with the source lines that support it."""

    section: str
    text: str
    evidence: list[Evidence] = Field(default_factory=list)
    verified: Optional[bool] = None


class ModelDoc(_Base):
    """The drafted documentation for one model."""

    model_id: str
    generated_by: str
    claims: list[Claim] = Field(default_factory=list)

    @property
    def verified_share(self) -> float:
        """Share of claims that were verified (0 if there are none)."""
        if not self.claims:
            return 0.0
        return sum(1 for c in self.claims if c.verified) / len(self.claims)


class Discrepancy(_Base):
    """A difference between what the old document says and what the code does."""

    field: str
    documented: str
    actual: str
    severity: str  # High | Medium | Low
    evidence: list[Evidence] = Field(default_factory=list)


class ParityCheck(_Base):
    """One check of the Python port against the R output."""

    name: str
    passed: bool
    value: Optional[float] = None
    tolerance: Optional[float] = None
    detail: str = ""


class ParityIteration(_Base):
    """One draft of the port and how it did on the checks."""

    n: int
    source: str  # offline-standin | cache | live:<model>
    code_path: str
    passed: bool
    diagnosis: list[str] = Field(default_factory=list)
    checks: list[ParityCheck] = Field(default_factory=list)
    diff_path: Optional[str] = None


class ParityResult(_Base):
    """The outcome of the whole repair loop."""

    model_id: str
    passed: bool
    iterations: list[ParityIteration]
    final_code_path: Optional[str] = None
    tolerances: dict[str, float] = Field(default_factory=dict)
    n_checks_final: int = 0
    n_checks_passed_final: int = 0


class Finding(_Base):
    """One validation finding with a Green, Amber or Red rating."""

    area: str
    metric: str
    value: str
    criterion: str
    rag: str  # Green | Amber | Red
    message: str
