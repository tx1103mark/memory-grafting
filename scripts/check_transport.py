import concurrent.futures
import subprocess
urls={
 'modelscope':'https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base/resolve/master/model.safetensors',
 'hf-mirror':'https://hf-mirror.com/Qwen/Qwen3-0.6B-Base/resolve/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/model.safetensors',
 'hf':'https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/model.safetensors'}
def check(item):
    k,url=item
    r=subprocess.run(['curl','-L','--range','0-1048575','--max-time','15','-s','-o','/dev/null','-w',
                      '%{http_code} bytes=%{size_download} speed=%{speed_download}',url],capture_output=True,text=True)
    return k,r.returncode,r.stdout,r.stderr
with concurrent.futures.ThreadPoolExecutor(3) as pool:
    for r in pool.map(check,urls.items()):print(r,flush=True)
