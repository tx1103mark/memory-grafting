import json
import urllib.request
for repo in ['openbmb/Ultra-FineWeb', 'ceval/ceval-exam', 'haonan-li/cmmlu']:
    with urllib.request.urlopen('https://hf-mirror.com/api/datasets/' + repo, timeout=45) as f:
        data = json.load(f)
    files = [s['rfilename'] for s in data['siblings']]
    print(repo, data['sha'], len(files), files[:25], flush=True)
