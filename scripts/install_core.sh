#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk0/home/tysearch/ngram-embedding
.venv/bin/python -m pip install --no-deps --timeout 60 -i https://pypi.tuna.tsinghua.edu.cn/simple \
 transformers==4.57.6 datasets==3.6.0 peft==0.18.0 accelerate==1.12.0 \
 huggingface-hub==0.36.0 tokenizers==0.22.2 dill==0.3.8 multiprocess==0.70.16 \
 fsspec==2025.3.0 datasketch==1.6.5 pytest==8.4.2
.venv/bin/python -m pytest -q tests/test_core.py
