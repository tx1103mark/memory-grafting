#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk0/home/tysearch/ngram-embedding
# A one-off dependency wait for this authorized preparation job, not a recurring automation.
while [ ! -f environment.ready ] || [ ! -f assets.complete ]; do
  sleep 30
done
.venv/bin/python -m scripts.prepare_data --root .
.venv/bin/python -m scripts.prepare_harness_tasks --root .
echo 'CPU preparation complete. GPU training requires an allocated device.'
