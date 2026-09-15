#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk0/home/tysearch/ngram-embedding
export HF_HUB_DISABLE_XET=1
exec /mnt/disk0/home/tysearch/conda_envs/engram-v41/bin/python -u scripts/download_assets.py --root .
