.PHONY: check assets
PYTHON ?= python

check:
	$(PYTHON) tools/verify_sources.py
	$(PYTHON) -m unittest discover -s tests -v

assets:
	$(PYTHON) assets/scripts/build_assets.py all
