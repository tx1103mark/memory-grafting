#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/long_source_study
nohup .venv/bin/python -u -m scripts.run_long_source_study > logs/long_source_study/runner.log 2>&1 </dev/null &
echo "$!" > logs/long_source_study/runner.pid
echo "Submitted runner PID $(cat logs/long_source_study/runner.pid); inspect runner.log for actual startup."
