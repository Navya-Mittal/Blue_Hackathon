# Blue: Blue-chip Legacy Upgrade Engine

*Keeping your models out of the red.*

EXL x Trinity Data & AI Hackathon 2026 | Challenge 04: Agentic model risk factory | Team **Blue**

## What it does

A bank may have hundreds of legacy models. Validating one by hand takes weeks, its documentation is often out of
date, and rewriting it in a new language is risky. Blue automates the evidence, and a human makes the decision.

Given a folder of legacy credit-risk models (R and Python), Blue runs six stages:

| Stage | What happens | Main output |
|---|---|---|
| 1. Discover | Scans the code, extracts facts and gives each model a risk tier using a rule you can read | `inventory.json` |
| 2. Document | Drafts documentation where every claim quotes its source lines, and a checker confirms the quote. Compares the old documentation with the code to find what is out of date | `docs/` |
| 3. Redevelop | Ports the R model to Python, runs the port in a guarded subprocess and compares it with the output of the real R model. If it does not match, it diagnoses why and repairs it | `port/`, `parity.json` |
| 4. Validate | Discrimination, calibration, stability, fairness on sex and age, and a gradient-boosting challenger | `metrics.json` |
| 5. Monitor | Drift checks (PSI, CSI) on synthetic batches and a re-validation ticket when thresholds are crossed | `monitor.json` |
| 6. Report | One HTML validation report. Any number in the text that was not computed is rejected | `report.html` |

Four steps need a human approval (risk tier, documentation, redevelopment, report), and every model call is written to
an audit log. The stages follow the PRA's SS1/23 model risk principles. In the code, Monitor runs before Report so the
report can include its results.

## Quick start

No API key and no R installation are needed. Python 3.10 or newer (developed on 3.13).

```bash
# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py          # about 30 s
python -m pytest -q         # about 1.5 min, 25 tests
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_demo.py
python -m pytest -q
```

Then open `artifacts/latest/report.html` in a browser. On macOS and Linux, `make setup` creates the virtual environment
and `make demo` and `make test` are shortcuts for the two run commands (activate the environment first).
`requirements-tested.txt` lists the exact versions the project was tested with.

Other commands:

```bash
python run_demo.py scan                 # print the model inventory only
python run_demo.py --interactive        # ask for y/n at each approval gate (recorded as interactive-user)
python run_demo.py --no-mlflow          # skip the optional MLflow tracking
python run_demo.py ai-usage             # write AI_USAGE.md from the last run's audit log
```

### Guided walkthrough (the demo notebook)

`notebooks/blue_demo.ipynb` runs the pipeline and then shows each stage's real output. Its last section plants mistakes
(an invented number, forged evidence, unsafe code, a port that is off by 0.001) and shows each one being caught.

- **Colab:** click the Open in Colab badge at the top of the notebook. The first cell clones this repo from GitHub, so
  there is nothing to upload (if cloning fails it asks for `blue-hackathon.zip` instead). With no API key it runs offline
  and takes about a minute. For live mode add a Colab secret named `GEMINI_API_KEY` (a free key from Google AI Studio)
  with notebook access switched on, or paste the key when asked. Live takes longer, because the LLM writes the
  documentation and every port draft.
- **VS Code or Jupyter:** open it from the `notebooks/` folder and run the cells in order. Set `GEMINI_API_KEY` in the
  environment for live mode.
- **Settings (section 0b):** `MODE` is `auto` (the default: live if a key is found, otherwise offline), `live`, `replay`
  or `offline`. `PROVIDER` is `gemini` (the default), `xai` (Grok) or `anthropic` (Claude). Leave `MODEL` blank to use
  that provider's default.
- **Repeat runs:** after a live run the last cell saves the recorded answers as `llm_cache.zip`. Unzip it into
  `llm_cache/` and set `MODE = "replay"` to re-run with no key and no cost.

## What the demo run finds

All of this is real output from the bundled public data (credit-card default, Taiwan, 2005), from the default offline run, where
the documentation wording, the port drafts and the report text are scripted stand-ins. The live LLM run is below.

| Result | Value |
|---|---|
| Models inventoried | 3: `credit_pd` is Tier 1, `fraud_gbm` Tier 2, `pricing_glm` Tier 3 |
| Documentation claims checked against the code | 19 of 19 verified (offline stand-in wording) |
| Out-of-date documentation | 5 planted errors found in `credit_pd`; 0 on the control model `pricing_glm` |
| R to Python port | Scripted drafts: draft 1 fails (5 of 13 terms missing), draft 2 fails (coefficients shrunk by a default penalty), draft 3 passes |
| Difference from the real R output | 1.4e-7 for coefficients and 5.0e-7 for predictions, against a tolerance of 1e-5 |
| Validation | Gini 0.437. 2 Red, 1 Amber, 7 Green. Draft outcome: escalate to the validation committee |
| The two Red findings | Worst risk decile is off by 0.137 (predicted 0.574, observed 0.711). A challenger model scores about +0.05 AUC higher (+0.0496 in our local run, +0.0492 in Colab) |
| Fairness | Sex is Green. For age, the equalised-odds gap is 0.060 (Amber) |
| Monitoring | A re-validation ticket is raised at batch 4 of 6 (synthetic drift) |
| Run time | About 25 s on Linux for this one model. The brief quotes 8 to 12 weeks for a manual validation |

The time figure is for one small model with automated steps only. It leaves out human review and governance, so it
must not be scaled up to a whole estate of about 250 models.

### One live run: Google Gemini 3.5 Flash-Lite (free tier)

The demo notebook was run live twice (23 and 24 November 2026), with a real LLM writing the documentation, the port and the
report summary. The checks were the same as offline.

| Result | Value |
|---|---|
| R to Python port | Draft 1 passed in both runs, with 9 of 9 parity checks. Coefficients within 1.4e-7 and predictions within 5.0e-7 of the real R output, against a tolerance of 1e-5 |
| Regenerated documentation for `credit_pd` | 5 of 5 claims verified against their quoted source lines in the first run, 4 of 4 in the second. The LLM words it differently each run |
| Out-of-date documentation | The same 5 errors found. This check is rule-based and does not use the LLM |
| Report summary | First run: Gemini's text was rejected by the numeric check, because it wrote a "95%" that is the confidence level of the intervals and not a computed result, so the template was used. The check now allows that one fixed level, and only beside "CI" or "confidence interval" wording. Second run: Gemini's summary passed the check |
| What this does not show | Two runs of one small model are too few to give a pass rate. The first draft passed, so the repair loop was not exercised live |

## What is, and is not, an LLM

| Mode | What happens | Status |
|---|---|---|
| `offline` (default) | No LLM. The documentation wording, the three port drafts and the report text come from rule-based stand-ins, labelled `offline-standin` in the report and the audit log. The drafts are scripted, but the repair loop really does diagnose each failure from the parity checks and pick the next draft from that diagnosis | Run and tested |
| `live` | Calls an LLM over the internet. The demo notebook has built-in adapters for Google Gemini (the default, with a free tier), xAI Grok and Anthropic Claude. The command line (`--mode live`) has adapters for Anthropic and OpenAI (`pip install anthropic`, set the API key). Responses are cached in `llm_cache/` | Notebook: run on Gemini 3.5 Flash-Lite, twice (see above). Grok and Claude in the notebook: tested only against a local stand-in server. Command line: never run against a real API, and its adapters send `temperature`, which current Claude models reject, so they need updating before use. The safeguards are also tested with a scripted fake provider that misbehaves on purpose |
| `replay` | Re-runs from `llm_cache/` with no network. A missing entry is an error, not a silent fallback | Tested with the fake provider, including in the notebook. No real recorded cache is committed yet; the notebook's last cell saves one after a live run |

So the checking and proof machinery is what is demonstrated here. The LLM is a replaceable part that this machinery
makes safe to use: the notebook changes provider with one setting, and the checks do not change.

## Responsible AI

The full note is in `docs/RESPONSIBLE_AI_NOTE.md`. In short:

- Blue assists validators and never approves a model. Reports are marked as drafts with a blank signature line.
- Only public or synthetic data is used. The default run sends nothing anywhere. In live mode, prompts hold source code
  and aggregate numbers, never data rows.
- The provider's terms matter. The live runs used Google's free Gemini tier, where Google's terms say prompts and replies may
  be used to improve its products (they apply different terms in the EEA, UK and Switzerland; check the current terms), so
  only public or synthetic code was sent. A bank would use a private endpoint that does not train on its data.
- Sex and age are not model inputs. They are only used to test outcomes. The age gap is reported as Amber, not hidden.
- Safeguards: quoted evidence for every claim, a parity test against real R output, an import and call allow-list with an
  isolated process for generated code, a check on every number in the text (the only fixed number it allows is the 95% interval level, written beside "CI" or
  "confidence interval"), and a log of every model call.

## Repository layout

```
blue/               the Python package (inventory, docgen, redevelop, validate, monitor, report, pipeline, llm, audit)
legacy_models/      the 3 sample "legacy" models, plus an out-of-date credit_pd_model_doc.md
fixtures/credit_pd/ output of the real R model, used as the parity reference (hash-checked)
data/               public dataset and fixed train/test splits (see data/README.md)
config.yaml         every threshold in one place (illustrative, not bank policy)
tests/              25 tests
notebooks/          blue_demo.ipynb, the guided demo
docs/               architecture diagram (PNG, SVG, PDF), Responsible AI note, submission guide
artifacts/latest/   created by a run: report.html, metrics.json, parity.json, audit.jsonl and more
```

## What has and has not been tested

| Area | State |
|---|---|
| Clean install | A fresh virtual environment from `requirements.txt` (Python 3.13, Linux): the demo runs and all 25 tests pass |
| Older libraries, no MLflow | Passed earlier with pandas 2.2.3, numpy 2.1.3, scikit-learn 1.6.1, statsmodels 0.14.4, fairlearn 0.11.0 and MLflow not installed (it is skipped with a message) |
| Demo notebook | Runs end to end locally, and in Colab (cloned from GitHub), offline and live on Gemini |
| Windows | The demo has been run in PowerShell. The test suite has not |
| macOS, Python 3.10 to 3.12 | Not tested |
| Live LLM path | Notebook live mode: run on Gemini 3.5 Flash-Lite, twice. Grok and Claude in the notebook: local stand-in only. Command-line `--mode live`: not tested against a real API |
| `.github/workflows/ci.yml` | Written but never run |

The code guard for generated code is an import and call allow-list plus an isolated subprocess with a timeout. It is
not an operating-system sandbox, so it is fine for a demo but not for running untrusted code.

## Limitations

- One public dataset and one model rebuilt end to end. The other two models are inventoried and documented only.
- The data has no time dimension, so there is no out-of-time test and the drift monitoring is a synthetic scenario.
- Fairness is tested one attribute at a time (sex, age band). The metrics are statistical indicators, not a legal finding.
- Thresholds and risk-tier rules are illustrative and would be replaced by a bank's model risk policy. Materiality is a
  business input in `config.yaml` because it cannot be read from code.
- Only R and Python models are read. SAS is not supported.
- In offline mode the port drafts are scripted, so a pass shows that the checks work, not how well an LLM ports code.
- The live evidence is two runs of one small free-tier model, which both passed at draft 1. That is too few to give a pass
  rate, and the repair loop was not exercised live. Live wording differs from run to run (for example 5 of 5 against 4 of 4
  documentation claims verified).

## Declarations

- **AI tools:** this repository was drafted with the help of Claude (Anthropic) as a coding assistant. Each run also writes
  `artifacts/latest/AI_USAGE.generated.md`, which lists every model call the pipeline made. The live demo runs also called
  Google Gemini (`gemini-3.5-flash-lite`) through the Gemini API.
- **Existing code:** no third-party code is copied. Libraries: numpy, pandas, scipy, scikit-learn, statsmodels, fairlearn,
  matplotlib, jinja2, pydantic, PyYAML, MLflow (optional), pytest. Optional, for live mode in the notebook: openai (used
  to reach Gemini and Grok) and anthropic.
- **Data:** public only. The dataset is Yeh and Lien (2009), "Default of Credit Card Clients", from the UCI Machine
  Learning Repository. Check its licence before redistributing the file (see `data/README.md`).

## Regenerating the R reference (optional)

The R output is already committed, so R is not needed to run or test. To rebuild it you need R with no extra packages:
`RSCRIPT=/path/to/Rscript python run_demo.py --regenerate-fixtures` (or `make fixtures`).

More on the submission steps and the 3-minute demo script is in `docs/SUBMISSION.md`.
