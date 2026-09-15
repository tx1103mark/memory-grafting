#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
host=${1:?usage: launch_semantic_study.sh 203-or-238}
mkdir -p logs/semantic_study
nohup .venv/bin/python -u -m scripts.run_semantic_study --host "$host" > "logs/semantic_study/runner-$host.log" 2>&1 </dev/null &
echo "$!" > "logs/semantic_study/runner-$host.pid"
echo "Submitted semantic-study host $host PID $(cat logs/semantic_study/runner-$host.pid)"
