"""Pin source revisions and download only selected Chinese shards + Base models."""
import argparse
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.pop('HF_XET_HIGH_PERFORMANCE', None)
os.environ.setdefault('HF_XET_NUM_CONCURRENT_RANGE_GETS', '4')
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def retry(fn, *args, **kwargs):
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception as error:
            print('Transfer retry', attempt+1, type(error).__name__, flush=True)
            if attempt == 5:
                raise
            time.sleep(min(30, 2**attempt))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    if (a.root/'assets.complete').exists():
        print('Locked assets already complete.',flush=True)
        return
    api = HfApi()
    lock_path = a.root / 'assets.lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
    else:
        lock = {}
        for key, repo, kind in [
            ('student', 'Qwen/Qwen3-0.6B-Base', 'model'),
            ('teacher', 'Qwen/Qwen3-8B-Base', 'model'),
            ('corpus', 'openbmb/Ultra-FineWeb', 'dataset'),
            ('ceval', 'ceval/ceval-exam', 'dataset'),
            ('cmmlu', 'haonan-li/cmmlu', 'dataset')]:
            info = retry(api.repo_info, repo, repo_type=kind)
            files = [s.rfilename for s in info.siblings]
            lock[key] = {'repo': repo, 'revision': info.sha, 'kind': kind, 'files': files}
            print(key, info.sha, files[:8], flush=True)
        candidates = [f for f in lock['corpus']['files'] if f.endswith('.parquet') and
                      any(x in f.lower() for x in ['/zh/', '/zh-', '_zh/', 'zh/', 'chinese'])]
        if not candidates:
            raise RuntimeError('Chinese parquet paths not identified: ' + str(lock['corpus']['files'][:40]))
        random.Random(42).shuffle(candidates)
        lock['corpus']['selected_files'] = candidates[:a.shards]
        lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False))
    def download_model(key):
        v = lock[key]
        patterns = ['*.json', '*.txt', '*.model', '*.jinja']
        if key == 'student':
            patterns += ['*.safetensors']
        else:
            # Teacher values use embedding + blocks 0..11 only. The pinned index
            # maps those tensors exclusively to shards 1 and 2.
            patterns += ['model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors']
        retry(snapshot_download, v['repo'], revision=v['revision'], local_dir=a.root/'models'/key,
                          allow_patterns=patterns, max_workers=2)
        print('COMPLETE', key, flush=True)
    def download_dataset(key):
        v = lock[key]
        files = v.get('selected_files') or [f for f in v['files'] if f.endswith(('.parquet', '.zip', '.csv', '.json'))]
        if not files:
            raise RuntimeError('No data files: ' + key)
        for f in files:
            print('Downloading', key, f, flush=True)
            retry(hf_hub_download, v['repo'], f, repo_type='dataset', revision=v['revision'], local_dir=a.root/'data'/'raw'/key)
        print('COMPLETE', key, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download_model, k) for k in ['student', 'teacher']]
        futures += [pool.submit(download_dataset, k) for k in ['ceval', 'cmmlu', 'corpus']]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    (a.root/'assets.complete').write_text('complete\n')


if __name__ == '__main__':
    from filelock import FileLock
    with FileLock('.assets-download.lock'):
        main()
