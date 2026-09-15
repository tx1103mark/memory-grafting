#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/mechanism_study
nohup .venv/bin/python -u -m scripts.run_mechanism_study > logs/mechanism_study/runner.log 2>&1 </dev/null &
echo "$!" > logs/mechanism_study/runner.pid
echo "Submitted mechanism study PID $(cat logs/mechanism_study/runner.pid)"
