#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/long_study
# Runner holds a nonblocking flock, so a duplicate exits without launching jobs.
nohup .venv/bin/python -u -m scripts.run_long_study > logs/long_study/runner.log 2>&1 </dev/null &
echo "$!" > logs/long_study/runner.pid
echo "Submitted runner PID $(cat logs/long_study/runner.pid); inspect runner.log for actual startup."
