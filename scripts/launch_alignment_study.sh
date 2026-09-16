#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/alignment_study
nohup env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u -m scripts.run_alignment_study > logs/alignment_study/run.log 2>&1 </dev/null &
echo "$!" > logs/alignment_study/run.pid
echo "Submitted alignment study PID $(cat logs/alignment_study/run.pid)"
