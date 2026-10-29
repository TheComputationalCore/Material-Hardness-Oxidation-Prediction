# Makefile for data audit, training, evaluation, and the Flask app

install:
	pip install -r requirements.txt -r requirements-dev.txt

audit:
	python -m src.data.audit

train:
	python -m src.models.train_hardness
	python -m src.models.train_oxidation

calibrate:
	python -m src.models.calibration

benchmark:
	python -m src.models.benchmark

reproduce:
	python -m src.models.reproducibility

external:
	python -m src.models.external_validation --write

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

check: audit reproduce test lint

run:
	python -m src.app.app

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -name '*.pyc' -delete
