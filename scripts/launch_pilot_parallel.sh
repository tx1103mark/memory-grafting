#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/disk0/home/tysearch/ngram-embedding
cd "$ROOT"
PY=.venv/bin/python
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
run_train() {
  local group=$1 gpu=$2 run="runs/${1}_seed42_2000000tokens"
  local resume=()
  if [ -f "$run/last.pt" ] && [ ! -f "$run/complete.json" ]; then resume=(--resume "$run/last.pt"); fi
  if [ ! -f "$run/complete.json" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" $PY -u -m scripts.train --root . --group "$group" --tokens 2000000 --seed 42 "${resume[@]}" > "logs/train_${group}.log" 2>&1
  fi
}
while [ ! -f data/processed/manifest.json ]; do sleep 30; done
$PY -m scripts.prepare_harness_tasks --root .
run_train L 0 & p_l=$!
while [ ! -f assets.complete ]; do sleep 30; done
if [ ! -f memory/table.pt ] || [ ! -f memory/manifest.json ]; then
  CUDA_VISIBLE_DEVICES=3 $PY -u -m scripts.build_memory --root . > logs/build_memory.log 2>&1
fi
run_train R 7 & p_r=$!
run_train G 3 & p_g=$!
wait "$p_l"; wait "$p_r"; wait "$p_g"
run_eval() {
  local group=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" $PY -u -m scripts.evaluate --root . \
    --checkpoint "runs/${group}_seed42_2000000tokens/last.pt" --tasks ceval-valid \
    --output "runs/${group}_ceval_val.json" > "logs/eval_${group}.log" 2>&1
}
run_eval L 0 & e_l=$!
run_eval R 7 & e_r=$!
run_eval G 3 & e_g=$!
wait "$e_l"; wait "$e_r"; wait "$e_g"
$PY -m scripts.compare_results --baseline runs/L_ceval_val.json --grafted runs/G_ceval_val.json --output runs/G_vs_L_pilot.json
$PY -m scripts.compare_results --baseline runs/R_ceval_val.json --grafted runs/G_ceval_val.json --output runs/G_vs_R_pilot.json
date -Iseconds > runs/pilot.complete
