PYTHON ?= python3

.PHONY: check reproduce test-legacy test-multimodel test-minor bundles release paper

check:
	$(PYTHON) tools/check_repository.py

reproduce:
	$(PYTHON) tools/reproduce_paper.py

test-legacy:
	$(PYTHON) tools/run_tests.py legacy

test-multimodel:
	$(PYTHON) tools/run_tests.py multimodel

test-minor:
	$(PYTHON) tools/run_tests.py minor_revision

bundles:
	$(PYTHON) tools/build_kaggle_bundle.py all

release:
	$(PYTHON) tools/build_release.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
