#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/disk0/home/tysearch/ngram-embedding
cd "$ROOT"
# ModelScope and hf-mirror are transport mirrors. Pinned Hugging Face LFS
# SHA-256 values below are the source of truth.
TEACHER_URL=https://www.modelscope.cn/models/Qwen/Qwen3-8B-Base/resolve/master
TEACHER_META_URL=https://huggingface.co/Qwen/Qwen3-8B-Base/resolve/49e3418fbbbca6ecbdf9608b4d22e5a407081db4
CORPUS_URL=https://hf-mirror.com/datasets/openbmb/Ultra-FineWeb/resolve/02c85641e3d19a854be2e09139c25adaa9518063
S1_CACHE='models/teacher/.cache/huggingface/download/ucOYR0sM0bukMpWiwV3qkLpQbMA=.9983f1b9ef2f60e7c3730d9bc11ada914e6ec630639b5b020a89bd158cd0446b.256aeecc.incomplete'
S2_CACHE='models/teacher/.cache/huggingface/download/ewHVStNi_qTCwOvZyUK4XvcG7h0=.9aa12339835bf7a093d3d0b0a0d2d77f8538301cfc7f8e7ec7a585858ebf7a1f.d10da547.incomplete'
CORPUS_CACHE='data/raw/corpus/.cache/huggingface/download/data/ultrafineweb_zh/6oE8nbN1q-4g_S3UidQdHQuiDEA=.5a68d59ae39c48d067e1b6a5b7013cb92c43265ed619391beeca4c28b4f42e59.3ad67daf.incomplete'
S1='models/teacher/model-00001-of-00005.safetensors'
S2='models/teacher/model-00002-of-00005.safetensors'
CORPUS='data/raw/corpus/data/ultrafineweb_zh/ultrafineweb-zh-part-235-of-256.parquet'
mkdir -p "$(dirname "$CORPUS")"
if [ ! -f "$S1" ] && [ -f "$S1_CACHE" ]; then mv "$S1_CACHE" "$S1"; fi
if [ ! -f "$S2" ] && [ -f "$S2_CACHE" ]; then mv "$S2_CACHE" "$S2"; fi
if [ ! -f "$CORPUS" ] && [ -f "$CORPUS_CACHE" ]; then mv "$CORPUS_CACHE" "$CORPUS"; fi
fetch() {
  local url=$1 out=$2 log=$3
  aria2c -c --allow-overwrite=true --auto-file-renaming=false --file-allocation=none \
    --max-tries=0 --retry-wait=3 --timeout=60 -x 16 -s 16 -k 1M -d "$(dirname "$out")" \
    -o "$(basename "$out")" "$url" > "$log" 2>&1
}
fetch "$TEACHER_URL/model-00001-of-00005.safetensors" "$S1" logs/aria2_teacher1.log & p1=$!
fetch "$TEACHER_URL/model-00002-of-00005.safetensors" "$S2" logs/aria2_teacher2.log & p2=$!
fetch "$CORPUS_URL/data/ultrafineweb_zh/ultrafineweb-zh-part-235-of-256.parquet" "$CORPUS" logs/aria2_corpus.log & p3=$!
wait "$p1"; wait "$p2"; wait "$p3"
for file in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do
  wget -q --timeout=60 --tries=20 -O "models/teacher/$file.tmp" "$TEACHER_META_URL/$file"
  mv "models/teacher/$file.tmp" "models/teacher/$file"
done
printf '%s  %s\n' \
  '9983f1b9ef2f60e7c3730d9bc11ada914e6ec630639b5b020a89bd158cd0446b' "$S1" \
  '9aa12339835bf7a093d3d0b0a0d2d77f8538301cfc7f8e7ec7a585858ebf7a1f' "$S2" \
  '5a68d59ae39c48d067e1b6a5b7013cb92c43265ed619391beeca4c28b4f42e59' "$CORPUS" | sha256sum -c -
"$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path
from safetensors import safe_open
root=Path('/mnt/disk0/home/tysearch/ngram-embedding')
with safe_open(root/'models/student/model.safetensors',framework='pt') as f:
    assert len(list(f.keys())) > 100
assert len(list((root/'data/raw/ceval').rglob('*.parquet'))) == 156
assert len(list((root/'data/raw/cmmlu').rglob('*.zip'))) == 1
(root/'assets.complete').write_text('sha256-verified required assets complete\n')
PY
echo 'Required assets verified.'
