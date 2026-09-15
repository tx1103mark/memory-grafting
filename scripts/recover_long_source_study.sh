#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
old_runner=$(cat logs/long_source_study/runner.pid)
manual_g=$(cat logs/long_source_study/G_T12_S1.manual.pid)
manual_r=$(cat logs/long_source_study/R_T12_S1.manual.pid)
while kill -0 "$old_runner" 2>/dev/null || kill -0 "$manual_g" 2>/dev/null || kill -0 "$manual_r" 2>/dev/null; do
  sleep 30
done
bash scripts/launch_long_source_study.sh
