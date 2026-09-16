#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/..";mkdir -p logs/aligned_downstream
nohup .venv/bin/python -u -m scripts.run_aligned_downstream > logs/aligned_downstream/runner.log 2>&1 </dev/null &
echo "$!" > logs/aligned_downstream/runner.pid
echo "Submitted aligned downstream PID $(cat logs/aligned_downstream/runner.pid)"
