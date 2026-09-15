#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk0/home/tysearch/ngram-embedding
.venv/bin/python -u scripts/fetch_wheel_ranges.py
bash scripts/bootstrap.sh
.venv/bin/python -m pytest -q tests
touch environment.ready
