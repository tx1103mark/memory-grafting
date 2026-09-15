"""Resume a PyPI wheel in independently verified HTTP ranges on unstable links."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

root=Path('/mnt/disk0/home/tysearch/ngram-embedding/wheels')
root.mkdir(exist_ok=True)
if (root/'scipy-source.json').exists():
    asset=json.loads((root/'scipy-source.json').read_text(encoding='utf-8-sig'))
else:
    with urllib.request.urlopen('https://pypi.org/pypi/scipy/1.18.1/json',timeout=60) as f:
        metadata=json.load(f)
    asset=next(v for v in metadata['urls'] if 'cp312-cp312-manylinux_2_27_x86_64' in v['filename'])
parts=root/(asset['filename']+'.parts')
parts.mkdir(exist_ok=True)
size=asset['size']
chunk=1_048_576
def fetch(start):
    end=min(start+chunk,size)-1
    part=parts/str(start)
    if part.exists() and part.stat().st_size==end-start+1:return part
    for attempt in range(8):
        try:
            req=urllib.request.Request(asset['url'],headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=30) as response:
                if response.status!=206:raise RuntimeError('Server ignored Range')
                content=response.read()
            if len(content)!=end-start+1:raise RuntimeError('Incomplete range')
            part.write_bytes(content)
            print('Downloaded range',start,end,flush=True)
            return part
        except Exception as e:
            print('Retry',start,type(e).__name__,flush=True)
            if attempt==7:raise
            time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    files=list(pool.map(fetch,range(0,size,chunk)))
target=root/asset['filename']
with target.open('wb') as out:
    for path in files:out.write(path.read_bytes())
if hashlib.sha256(target.read_bytes()).hexdigest()!=asset['digests']['sha256']:
    raise RuntimeError('PyPI wheel SHA256 mismatch')
print('VERIFIED',target,flush=True)
