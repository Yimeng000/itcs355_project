PYTHON ?= python

.PHONY: test train portability-audit scan-secrets check

test:
	$(PYTHON) -m pytest -q tests/

train:
	$(PYTHON) -m dvc repro

portability-audit:
	$(PYTHON) scripts/portability_audit.py

scan-secrets:
	$(PYTHON) scripts/scan_secrets.py

check: portability-audit scan-secrets test
