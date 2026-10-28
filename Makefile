# Makefile for ML Training + Flask App

install:
	pip install -r requirements.txt

train:
	python -m src.models.train_hardness
	python -m src.models.train_oxidation

evaluate:
	python -m src.models.evaluate

run:
	python -m src.app.app


clean:
	rm -rf __pycache__
	find . -name '*.pyc' -delete
