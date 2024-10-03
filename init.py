import argparse
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from threading import Thread
from time import time, sleep
from typing import Callable, Union

PUBLIC_IP = os.getenv('RUNPOD_PUBLIC_IP', '<pod public ip>')
SSH_PORT = os.getenv('RUNPOD_TCP_PORT_22', '<ssh connect port>')
SIMULATE = False


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Runpod LLM model serving'
    )

    model_dir = os.getenv('MODEL_DIR', '/models')
    ctx = int(os.getenv('CTX', 8192))
    port = int(os.getenv('PORT', 5002))
    args = os.getenv('ARGS', '--usecublas mmq --gpulayers 999 --flashattention --ignoremissing --skiplauncher')
    quantkv = os.getenv('QUANTKV')
    dry_run = bool(os.getenv('DRYRUN', False))
    model = os.getenv('MODEL')

    parser.add_argument('--model-dir', default=model_dir, metavar='path', help='local model directory [env: MODEL_DIR]')
    parser.add_argument('--ctx', default=ctx, metavar='n', help='max inference context length [env: CTX]')
    parser.add_argument('--args', default=args, metavar='s',
                        help='additional koboldcpp arguments and options [env: ARGS]')
    parser.add_argument('--port', default=port, metavar='n', help='port to bind koboldcpp to [env: PORT]')
    parser.add_argument('--quantkv', choices=(0,1,2), default=quantkv, type=int, metavar='n',
                        help='KV cache quantization to use. 0=fp16, 1=q8, 2=q4 [env: QUANTKV]')
    parser.add_argument('--dry-run', action='store_true', default=dry_run, help='simulate bootstrapping [env: DRYRUN]')
    parser.add_argument('--model', default=model, metavar='url/path', required=not model,
                        help='single HF file, repo or directory (copy it from HF repo web UI or point it to a local '
                             'volume mounted model) [env: MODEL]')

    if not (model or '--model' in sys.argv):
        sys.argv.append('-h')
        Thread(target=lambda: sleep(0.1) or print(
            '\n[!] ERROR No model specified. Set the MODEL environment variable to point at one.\n')).start()

    return parser.parse_args()


def main():
    global SIMULATE

    try:
        opts = parse_args()
        SIMULATE = opts.dry_run

        if is_url(opts.model):
            repo, rev, file = parse_hf_url(opts.model)
            download(repo, rev, file, opts.model_dir)
            model_path = find_model_path(opts.model_dir)
        else:
            model_path = opts.model
        launch(model_path, opts.args, opts.ctx, opts.port, opts.quantkv)
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):  # allow user to manually terminate
            sys.exit(1)
        print('[*] FATAL startup error encountered. Entering idle sleep mode instead of crash-looping, '
              'to allow manual trouble shooting in Runpod admin UI and logs. Error message:', str(e))


def download(repo, rev, file, models_dir):
    cmd = f"huggingface-cli download --local-dir '{models_dir}' --revision '{rev}' '{repo}'"
    if file:
        files = derive_files(file)
        cmd += ''.join(f" '{f}'" for f in files)
    run(cmd)


def derive_files(fn):
    m = re.match(r'(.*?)(\d+)-of-(\d+).gguf$', fn)
    if not m:
        return [fn]
    prefix, _, nfiles = m.groups()
    nfiles = int(nfiles)
    return [f'{prefix}{i:05d}-of-{nfiles:05d}.gguf' for i in range(1,nfiles+1)]


def parse_hf_url(url):
    prefix = 'https://huggingface.co/'
    assert url.startswith(prefix), f'Invalid Huggingface URL, must be absolute path and start with {prefix},got "{url}"'

    parts = url[len(prefix):].split('/')
    assert len(parts) > 1, "Invalid Huggingface URL, the URL must point to a valid repository"

    repo = '/'.join(parts[:2])
    rev = 'main'
    file = ''

    if len(parts) >= 4:
        rev = parts[3]
    if len(parts) > 4:
        file = '/'.join(parts[4:]).split("?")[0]

    return repo, rev, file


def find_model_path(model_dir):
    for d, _, files in os.walk(model_dir):
        for f in sorted(files):
            if f.lower().endswith('.gguf'):
                return os.path.join(d, f)
    if SIMULATE:
        return f"{model_dir}/tinyllama.gguf"
    raise ValueError(f'Failed to find a gguf in downloaded model assets: {model_dir}'
                     ' . Check the repo on huggingface')


def launch(model_path, args:str, context_length:int, port:int, quantkv:int):
    log("For increased security, create a direct ssh tunnel manually with command:")
    log(f"ssh root@{PUBLIC_IP} -p {SSH_PORT} -L {port}:localhost:{port}")
    log(f"instead of relying on runpod or cloudflare MITM proxies, and point the browser to http://localhost:{port}/")
    print()
    log("(verify that one of the SSH host keys above match the one shown with the -v [verbose] ssh connection option, "
        "to ensure no obvious MITM attempt)")
    print('-' * 78)

    if quantkv is not None:
        args += f" --flashattention --quantkv {quantkv}"
    cmd = (f"koboldcpp --contextsize {context_length} --host 0.0.0.0 --port {port} --model '{model_path}' {args}")
    with autotune_ctx_fallback(cmd, model_path, context_length, quantkv):
        run(cmd)


@contextmanager
def autotune_ctx_fallback(koboldcmd:str, model:str, context_length:int, quantkv:int):
    quant_map = {0: 'f16', 1: 'q8_0', 2: 'q4_0'}

    started = time()
    try:
        yield
    except subprocess.CalledProcessError:
        if time() - started > 120:  # if it's been running for a while, then it is probably a fatal problem and not OOM
            raise
        print('[!] Failed to launch kobold model serving with a context size of', context_length)
        print('[*] Starting search optimization for the largest context size that will fit in the GPU VRAM')

        llamacmd = f"llama-cli -m '{model}' --predict 5 -ngl 999 --log-disable"

        if quantkv in quant_map:
            v =  quant_map[quantkv]
            llamacmd += f" --flash-attn --cache-type-k {v} --cache-type-v {v}"

        largest_ctxlen = search(128, context_length, assessor(llamacmd))

        if largest_ctxlen is None:
            print('[!] FATAL - Failed to find any context size that fits into the GPU VRAM for the model')
            return

        # make an new attempt at launching kobold with the discovered ctx-len, and if that fails, enter post mortem mode
        new_ctx = (largest_ctxlen // 128) * 128
        if new_ctx != largest_ctxlen:
            print(f'[*] Rounding down discovert ctx size to nearest 128 multiple, from {largest_ctxlen} -> {new_ctx}')

        print('\n[*] Launching kobold a second time with context size:', new_ctx)
        new_cmd = replace_context_size_arg(koboldcmd, new_ctx)
        run(new_cmd)


def assessor(llamacmd:str):
    def assess(ctxlen:int):
        print(f'\n[*] Trying: {ctxlen} context size')
        try:
            run(f'{llamacmd} --ctx-size {ctxlen}', capture=True, show=False)
            print('[*] SUCCESS - increasing size')
            return True
        except subprocess.CalledProcessError:
            print('[!] FAILURE - decreasing size')
        return False
    return assess


def search(low:int, high:int, assess: Callable[[int], bool]) -> Union[None, int]:
    """Search for the longest context length that does not lead to VRAM OOM crash"""

    # Start by finding the actual bounds that do not lead to crashes. Begin with the smallest, and if that
    # doesn't work, no point continuing. Then the largest. If that works, then no need to continue searching.
    # Otherwise, do binary search to find the largest value that does not lead to a crash.

    if not assess(low):
        return

    if assess(high):
        return high

    last_success = None

    while low < high:
        mid = (high + low) // 2
        print(f'[*] Next search range: {low} - {high}')
        if assess(mid):
            last_success, low = mid, mid + 1
        else:
            high = mid - 1
    return last_success


def replace_context_size_arg(koboldcmd:str, new_context_size:int) -> str:
    args = shlex.split(koboldcmd)
    opt = '--contextsize'
    if opt in args:
        i = args.index(opt)
        args = args[:i] + args[i+2:]
    args += [opt, str(new_context_size)]
    return shlex.join(args)


def is_url(path:str):
    return path.startswith('http')


def run(cmd, capture=False, show=True):
    if show:
        log(cmd)
    if not SIMULATE:
        subprocess.run(shlex.split(cmd), check=True, capture_output=capture)


def log(s):
    print('[*]', s)


if __name__ == '__main__':
    main()
