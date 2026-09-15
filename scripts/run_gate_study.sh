#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk0/home/tysearch/ngram-embedding
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs/gate_study runs/gate_study
run_one() {
  local gpu=$1 group=$2 alpha=$3 name=$4
  export CUDA_VISIBLE_DEVICES=$gpu
  .venv/bin/python -u -m scripts.train --group "$group" --alpha-init "$alpha" --tokens 2000000 --micro-batch 2 --accumulation 2 --run-name "$name" > "logs/gate_study/$name.train.log" 2>&1
  .venv/bin/python -u -m scripts.evaluate --checkpoint "runs/$name/last.pt" --tasks ceval-valid --output "runs/gate_study/$name.json" > "logs/gate_study/$name.eval.log" 2>&1
  if [ "$group" != L ]; then
    .venv/bin/python -u -m scripts.diagnose_memory --checkpoint "runs/$name/last.pt" --output "runs/gate_study/$name.diagnostic.json" > "logs/gate_study/$name.diagnostic.log" 2>&1
  fi
}
# First inspect the existing checkpoint on the same validation prefix and disable memory in harness.
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u -m scripts.diagnose_memory --checkpoint runs/G_seed42_2000000tokens/last.pt --output runs/gate_study/original_G.diagnostic.json > logs/gate_study/original_G.diagnostic.log 2>&1
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u -m scripts.evaluate --checkpoint runs/G_seed42_2000000tokens/last.pt --memory-mode zero --output runs/gate_study/original_G.off.json > logs/gate_study/original_G.off.log 2>&1
run_one 0 L .001 gate_L & p0=$!
run_one 2 R .001 gate_R_a001 & p1=$!
run_one 3 G .001 gate_G_a001 & p2=$!
run_one 4 R .01 gate_R_a01 & p3=$!
run_one 5 G .01 gate_G_a01 & p4=$!
run_one 6 R .05 gate_R_a05 & p5=$!
run_one 7 G .05 gate_G_a05 & p6=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5" "$p6"; do wait "$pid" || failed=1; done
if [ "$failed" != 0 ]; then exit 1; fi
date -Iseconds > runs/gate_study/complete
