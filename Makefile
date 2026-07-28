# Ergonomic entry points for the thesis codebase.
#
# Why not just `uv run`? On macOS, `uv run` re-syncs the venv every call and
# uv's install process sets the `UF_HIDDEN` file flag on editable-install .pth
# files, which makes CPython's site.py silently skip them. `make fix-pth`
# strips that flag once, and the `PY` invocation below uses the venv's Python
# directly so no re-sync (and no re-hiding) happens.

PY_BIN := .venv/bin/python
PY := PYTHONPATH=src $(PY_BIN)
SITE_PACKAGES := .venv/lib/python3.12/site-packages

.PHONY: help sync fix-pth qa smoke baseline test lint format clean stew-check wauc-check

help:
	@echo "Common targets:"
	@echo "  make sync        # uv sync + fix-pth  (first-time or after deps change)"
	@echo "  make fix-pth     # strip macOS UF_HIDDEN from .pth files"
	@echo "  make qa          # data-loader smoke test on subject 1"
	@echo "  make smoke       # FBCSP smoke run on subject 1 (both datasets)"
	@echo "  make baseline    # full FBCSP baseline — all 9 subjects, both protocols"
	@echo "  make test        # run pytest"
	@echo "  make lint        # ruff check"
	@echo "  make format      # ruff format + ruff check --fix"
	@echo "  make stew-check  # validate the manually-placed STEW data under data/STEW/"
	@echo "  make wauc-check  # validate the manually-placed WAUC data under data/WAUC/"

sync:
	uv sync --all-extras
	$(MAKE) fix-pth

fix-pth:
	@chflags nohidden $(SITE_PACKAGES)/*.pth 2>/dev/null || true

qa: fix-pth
	$(PY) scripts/qa_load.py --subjects 1

smoke: fix-pth
	$(PY) scripts/run_fbcsp_baseline.py --subjects 1 --output results/fbcsp_smoke.csv

baseline: fix-pth
	$(PY) scripts/run_fbcsp_baseline.py --output results/fbcsp_baseline.csv

test: fix-pth
	$(PY) -m pytest

lint: fix-pth
	$(PY_BIN) -m ruff check src scripts tests

format: fix-pth
	$(PY_BIN) -m ruff format src scripts tests
	$(PY_BIN) -m ruff check --fix src scripts tests

clean:
	rm -rf results/*.csv .pytest_cache .ruff_cache .mypy_cache

stew-check: fix-pth
	$(PY) scripts/check_stew_data.py

wauc-check: fix-pth
	$(PY) scripts/check_wauc_data.py
