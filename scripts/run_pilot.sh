#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk0/home/tysearch/ngram-embedding
PY=.venv/bin/python
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
# CUDA_VISIBLE_DEVICES must be explicitly set by the operator after resource allocation.
: "${CUDA_VISIBLE_DEVICES:?Set an allocated GPU first}"
$PY -m pytest -q tests
if [ ! -f memory/table.pt ] || [ ! -f memory/manifest.json ]; then
  $PY -m scripts.build_memory --root .
fi
$PY -m scripts.evaluate --root . --tasks ceval-valid --output runs/B0_ceval_val.json
for group in L R G; do
  run="runs/${group}_seed42_2000000tokens"
  if [ ! -f "$run/complete.json" ]; then
    resume=()
    if [ -f "$run/last.pt" ]; then resume=(--resume "$run/last.pt"); fi
    $PY -m scripts.train --root . --group "$group" --tokens 2000000 --seed 42 "${resume[@]}"
  fi
  $PY -m scripts.evaluate --root . --checkpoint "runs/${group}_seed42_2000000tokens/last.pt" \
    --tasks ceval-valid --output "runs/${group}_ceval_val.json"
done
$PY -m scripts.compare_results --baseline runs/L_ceval_val.json --grafted runs/G_ceval_val.json --output runs/G_vs_L_pilot.json
$PY -m scripts.compare_results --baseline runs/R_ceval_val.json --grafted runs/G_ceval_val.json --output runs/G_vs_R_pilot.json
echo 'Pilot completed; compare controls before deciding on main runs.'
