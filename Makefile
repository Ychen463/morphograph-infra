.PHONY: install test lint audit

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

audit:
	python scripts/audit_splits.py --manifest data/manifests/all.csv --protocol configs/protocols/lodo.example.yaml
