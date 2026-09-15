#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/confirmation_study
nohup .venv/bin/python -u -m scripts.run_confirmation_study > logs/confirmation_study/runner.log 2>&1 </dev/null &
echo "$!" > logs/confirmation_study/runner.pid
echo "Submitted confirmation study PID $(cat logs/confirmation_study/runner.pid)"
