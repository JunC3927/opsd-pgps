#!/usr/bin/env python3
"""Accept the historical local wheel or explicitly download the official fixed release."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate(path, spec):
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        raise ValueError('FlashAttention wheel missing or has an unexpected byte count')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != spec['sha256']:
        raise ValueError('FlashAttention wheel SHA256 mismatch; refusing installation')


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--wheel', type=Path)
    source.add_argument('--download', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--no-proxy', action='store_true')
    args = p.parse_args()
    spec = json.loads((ROOT / 'flash_attn_wheel.json').read_text())
    destination = args.output_dir.expanduser().resolve() / spec['filename']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate(destination, spec)
    else:
        partial = destination.with_name(destination.name + '.partial')
        if args.wheel:
            original = args.wheel.expanduser().resolve()
            validate(original, spec)
            shutil.copyfile(original, partial)
        else:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.no_proxy else urllib.request.build_opener()
            request = urllib.request.Request(spec['url'], headers={'User-Agent': 'geometry-opsd-environment/1'})
            print('Downloading historical FlashAttention wheel (about 244 MiB)...', file=sys.stderr, flush=True)
            with opener.open(request, timeout=120) as response, partial.open('wb') as output:
                downloaded = 0
                next_report = 32 * 1024 * 1024
                for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                    output.write(block)
                    downloaded += len(block)
                    if downloaded >= next_report:
                        print('FlashAttention: %.0f / %.0f MiB' %
                              (downloaded / 1048576, spec['bytes'] / 1048576), file=sys.stderr, flush=True)
                        next_report = downloaded + 32 * 1024 * 1024
        print('Verifying FlashAttention wheel SHA256...', file=sys.stderr, flush=True)
        validate(partial, spec)
        partial.replace(destination)
    # Only the local wheel pathname is printed; no proxy/auth environment is logged.
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
