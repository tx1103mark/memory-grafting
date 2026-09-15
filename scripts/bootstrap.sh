#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/disk0/home/tysearch/ngram-embedding
BASE=/mnt/disk0/home/tysearch/conda_envs/engram-v41/bin/python
mkdir -p "$ROOT"/{data,models,memory,runs,logs}
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$BASE" -m venv --system-site-packages "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --find-links "$ROOT/wheels" --timeout 120 --retries 5 -i https://pypi.tuna.tsinghua.edu.cn/simple -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/python" -m pip freeze > "$ROOT/environment.freeze.txt"
cd "$ROOT"
# Asset download is launched separately so package installation does not block it.
