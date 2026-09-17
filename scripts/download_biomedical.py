"""Download and verify the pinned Biomed-Enriched shard used by TinyEngram."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO='almanach/Biomed-Enriched'
REVISION='03a84da5ae6d75bd784b678972e46d880967f8e8'
FILE='data/commercial-00000-of-00026.parquet'
SHA256='537979ade2adc124fe386b274c765c26c6e93da60235fad964e6ba009bfbe33f'
SIZE=1_861_721_014

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args()
    folder=a.root/'data/raw/biomed';folder.mkdir(parents=True,exist_ok=True)
    path=Path(hf_hub_download(REPO,FILE,repo_type='dataset',revision=REVISION,local_dir=folder))
    actual=digest(path)
    if path.stat().st_size!=SIZE or actual!=SHA256:
        raise RuntimeError(f'Biomedical shard integrity mismatch: size={path.stat().st_size}, sha256={actual}')
    manifest={'repo':REPO,'revision':REVISION,'file':FILE,'size':SIZE,'sha256':SHA256}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(path)

if __name__=='__main__':main()
