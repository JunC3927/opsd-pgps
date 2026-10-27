"""Explicit one-command workflow: official data -> checks -> smoke -> formal training."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

from .common import load_settings, repo_root
from .runtime import prepared, read, fingerprint, finished


def execute(script, config, *extra):
    command = [sys.executable, '-u', str(repo_root() / 'scripts' / script), '--config', str(config), *extra]
    print('\nRunning: ' + ' '.join(command), flush=True)
    subprocess.run(command, cwd=repo_root(), check=True)


def run(config, prepare_only=False, smoke_gpu=None):
    settings = load_settings(config)
    output = Path(settings['paths']['output'])
    output.mkdir(parents=True, exist_ok=True)
    exp = output / settings['experiment']['name']
    with (output / ('.' + exp.name + '.workflow.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not exp.exists():
            execute('prepare.py', config)
        assert (exp / 'PREPARED.json').is_file(), 'Previous preparation did not complete; inspect PREPARATION_FAILED.json'
        value = prepared(config)
        formal = Path(value['run_dir'])
        if any((formal / name).exists() for name in ('process.json', 'train.log', 'SUCCESS')):
            finished(formal)  # Reject active/failed jobs; never automatically retrain.
            print('Formal training is already complete: ' + str(formal), flush=True)
            return
        cpu = exp / 'CPU_PREFLIGHT_PASSED.json'
        if not cpu.exists():
            execute('preflight.py', config)
        sig = fingerprint(value)
        assert read(cpu)['passed'] and read(cpu)['fingerprint'] == sig, 'Prepared code/data changed; use a new experiment name'
        assert read(exp / 'CODE_FROZEN.json')['fingerprint'] == sig
        if prepare_only:
            print('CPU preparation complete; no GPU job started: ' + str(exp), flush=True)
            return
        marker = exp / 'SMOKE_PASSED.json'
        if not marker.exists():
            smoke = Path(value['smoke_dir'])
            if any((smoke / name).exists() for name in ('process.json', 'train.log', 'SUCCESS')):
                finished(smoke, True)  # Only a successful existing smoke can continue to export.
            else:
                execute('launch.py', config, '--mode', 'smoke')
            gpu = value['gpus'][0] if smoke_gpu is None else smoke_gpu
            execute('smoke_check.py', config, '--gpu', str(gpu))
        assert read(marker)['passed'] and read(marker)['fingerprint'] == sig
        print('Smoke passed. Starting 669 updates from the original base model.', flush=True)
        execute('launch.py', config, '--mode', 'train')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Read official data and run CPU checks only')
    parser.add_argument('--smoke-gpu', type=int, help='Defaults to the first training GPU after smoke training releases it')
    args = parser.parse_args()
    if args.smoke_gpu is not None and args.smoke_gpu < 0:
        parser.error('--smoke-gpu must be nonnegative')
    run(args.config, args.prepare_only, args.smoke_gpu)


if __name__ == '__main__':
    main()
