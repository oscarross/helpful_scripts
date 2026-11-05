# Video Duplicate Finder - install & run helpers
#
#   make install              create .venv and install dependencies
#   make tui [DIR=~/Movies]   terminal UI
#   make run [DIR=~/Movies]   classic CLI (interactive if DIR is empty)
#   make help                 list all targets

SHELL := /bin/bash

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
STAMP  := $(VENV)/.installed

# Scan options (override on the command line, e.g. make run DIR=~/Movies METHOD=combined)
DIR       ?=
METHOD    ?= phash
THRESHOLD ?= 0.85
WORKERS   ?= 4
ARGS      ?=

.DEFAULT_GOAL := help
.PHONY: help install reinstall check tui run scan lint format clean

help: ## Show this help
	@echo "Video Duplicate Finder"
	@echo
	@echo "Usage: make <target> [DIR=path] [METHOD=phash|dhash|average|combined] [THRESHOLD=0.85] [WORKERS=4]"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Examples:"
	@echo "  make install"
	@echo "  make tui DIR=~/Movies"
	@echo "  make scan DIR=~/Movies METHOD=combined THRESHOLD=0.9 ARGS='--json dups.json'"

$(STAMP): requirements.txt
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' || \
		{ echo "❌ Python 3.8+ is required (found: $$($(PYTHON) --version 2>&1))"; exit 1; }
	@if test -d $(VENV) && ! $(PY) -c 'import sys, sysconfig, os; sys.exit(0 if os.path.isdir(sysconfig.get_paths()["purelib"]) and len(os.listdir(os.path.join(sys.prefix, "lib"))) == 1 else 1)' 2>/dev/null; then \
		echo "♻️  $(VENV) is broken or was made with another Python version - recreating"; rm -rf $(VENV); \
	fi
	@test -x $(PY) || { echo "🐍 Creating virtualenv in $(VENV)…"; \
		if command -v uv >/dev/null; then uv venv --quiet --python $(PYTHON) $(VENV); \
		else $(PYTHON) -m venv $(VENV); fi; }
	@echo "📦 Installing dependencies (opencv, librosa, scikit-image… first run can take a few minutes)…"
	@if command -v uv >/dev/null; then \
		uv pip install --python $(PY) -r requirements.txt; \
	else \
		$(PY) -m pip --version >/dev/null 2>&1 || $(PY) -m ensurepip --upgrade >/dev/null || \
			{ echo "❌ pip is missing in $(VENV) and ensurepip failed - try: make reinstall"; exit 1; }; \
		$(PY) -m pip install --quiet --upgrade pip && \
		$(PY) -m pip install -r requirements.txt; \
	fi
	@touch $(STAMP)
	@echo "✅ Installed. Next: make tui"
	@command -v ffmpeg >/dev/null || \
		echo "⚠️  ffmpeg not found - audio fingerprinting will be skipped (brew install ffmpeg)"

install: $(STAMP) ## Create .venv and install dependencies (re-runs when requirements.txt changes)

reinstall: ## Recreate .venv from scratch
	rm -rf $(VENV)
	@$(MAKE) --no-print-directory install

check: $(STAMP) ## Verify Python packages and ffmpeg are available
	@$(PY) -c 'import cv2, imagehash, librosa, skimage, textual, send2trash; print("✅ Python packages OK (OpenCV", cv2.__version__ + ")")'
	@command -v ffmpeg >/dev/null && echo "✅ ffmpeg OK" || \
		echo "⚠️  ffmpeg not found - audio fingerprinting will be skipped (brew install ffmpeg)"

tui: $(STAMP) ## Launch the terminal UI (optional DIR=path)
	@echo "🚀 Starting TUI ($$($(PY) --version))…"
	@$(PY) tui.py "$(DIR)" || { code=$$?; echo "❌ TUI exited with code $$code"; exit $$code; }

run: $(STAMP) ## Run the CLI; asks interactively when DIR is not given
	@$(PY) video_duplicate_finder.py $(if $(DIR),"$(DIR)" -m $(METHOD) -t $(THRESHOLD) -w $(WORKERS)) $(ARGS)

scan: $(STAMP) ## Non-interactive scan (requires DIR=path)
	@test -n "$(DIR)" || { echo "❌ Usage: make scan DIR=path/to/videos"; exit 2; }
	@$(PY) video_duplicate_finder.py "$(DIR)" -m $(METHOD) -t $(THRESHOLD) -w $(WORKERS) $(ARGS)

lint: ## Lint & check formatting with ruff (same rules as CI)
	@command -v ruff >/dev/null || { echo "❌ ruff not found (pip install ruff)"; exit 1; }
	ruff check video_duplicate_finder.py tui.py
	ruff format --check video_duplicate_finder.py tui.py

format: ## Auto-format with ruff
	ruff format video_duplicate_finder.py tui.py

clean: ## Remove .venv and caches
	rm -rf $(VENV) __pycache__
