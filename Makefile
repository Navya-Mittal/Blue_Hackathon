PY ?= python3
VENV ?= .venv

.PHONY: setup demo test fixtures clean

setup:            ## create a virtualenv and install dependencies
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

demo:             ## full pipeline, offline mode (no API key, ~25 s) -> artifacts/latest/report.html
	$(PY) run_demo.py

test:             ## run the test suite (~75 s)
	$(PY) -m pytest -q

fixtures:         ## regenerate the R reference outputs (needs Rscript on PATH, or set RSCRIPT)
	$(PY) run_demo.py --regenerate-fixtures

clean:
	rm -rf artifacts mlruns mlartifacts .pytest_cache
