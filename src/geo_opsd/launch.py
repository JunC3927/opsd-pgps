"""Run one explicitly requested smoke or training job on exactly six free GPUs."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import time

from .common import repo_root
from .runtime import prepared, read, put, fingerprint, process_identity, free_gpus


def environment(value, run, mode):
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy') or key == 'DASHSCOPE_API_KEY' or key.startswith('PGPS_SHARED_EVAL_') or key in ('PGPS_TRAIN_GPUS', 'VLLM_ATTENTION_BACKEND', 'WANDB_ENTITY', 'WANDB_RUN_ID'):
            env.pop(key)
    runtime = run / 'runtime'
    runtime.mkdir(exist_ok=True)
    short = Path('/tmp') / ('geoopsd_' + hashlib.sha256(str(run).encode()).hexdigest()[:12])
    if short.is_symlink():
        assert short.resolve() == runtime.resolve()
    elif short.exists():
        raise RuntimeError('Temporary runtime path belongs to another experiment')
    else:
        short.symlink_to(runtime, target_is_directory=True)
    for name in ('tmp', 'ray', 'wandb', 'wandb_cache', 'wandb_data', 'vllm_cache', 'triton'):
        (runtime / name).mkdir(exist_ok=True)
    wb = value.get('wandb', {})
    wbmode = (wb.get('mode') or 'offline') if mode == 'train' else 'disabled'
    assert wbmode in ('offline', 'online', 'disabled')
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, value['gpus'])), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONPATH=os.pathsep.join([str(Path(repo_root()) / 'src'), value['framework']]),
        RAY_ADDRESS='local', RAY_TMPDIR=str(short / 'ray'), TMPDIR=str(short / 'tmp'),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        VLLM_USE_V1='1', VLLM_CACHE_ROOT=str(runtime / 'vllm_cache'), TRITON_CACHE_DIR=str(runtime / 'triton'),
        TENSORBOARD_DIR=str(run / 'tensorboard'), WANDB_DIR=str(runtime / 'wandb'),
        WANDB_CACHE_DIR=str(runtime / 'wandb_cache'), WANDB_DATA_DIR=str(runtime / 'wandb_data'),
        WANDB_MODE=wbmode, WANDB_PROJECT=wb.get('project') or 'geometry-opsd',
        WANDB_RUN_ID=wb.get('run_id') or 'geo-' + hashlib.sha256(value['experiment_dir'].encode()).hexdigest()[:12],
        WANDB_RESUME='never', WANDB_INIT_TIMEOUT='300', NO_PROXY='*', no_proxy='*', HYDRA_FULL_ERROR='1',
        GEO_OPSD_LAUNCHED=value['experiment_dir'])
    if wb.get('entity'):
        env['WANDB_ENTITY'] = wb['entity']
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    args = parser.parse_args()
    value = prepared(args.config)
    e = Path(value['experiment_dir'])
    with (e / 'training_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sig = fingerprint(value)
        assert read(e / 'CODE_FROZEN.json')['fingerprint'] == sig
        cpu = read(e / 'CPU_PREFLIGHT_PASSED.json')
        assert cpu['passed'] and cpu['fingerprint'] == sig and cpu['gpu_count'] == 6
        if args.mode == 'train':
            smoke = read(e / 'SMOKE_PASSED.json')
            assert smoke['passed'] and smoke['fingerprint'] == sig and smoke['gpu_count'] == 6
            assert smoke['real_exported_model_inference'] is True
        run = Path(value['run_dir' if args.mode == 'train' else 'smoke_dir'])
        assert not any((run / name).exists() for name in ('process.json', 'train.log', 'SUCCESS')), 'Existing run is preserved; inspect before recovery'
        assert not list((run / 'checkpoints').glob('global_step_*')), 'Existing checkpoints are preserved'
        snapshot = free_gpus(value['gpus'])
        run.mkdir(parents=True, exist_ok=True)
        info = {'mode': args.mode, 'time': time.time(), 'launcher': process_identity(os.getpid()),
                'gpus': value['gpus'], 'gpu_snapshot': snapshot, 'fingerprint': sig}
        try:
            with (run / 'train.log').open('xb') as output:
                child = subprocess.Popen([value['python'], '-u', '-m', 'geo_opsd.train',
                    '--config', value['config_path'], '--mode', args.mode], cwd=value['framework'],
                    env=environment(value, run, args.mode), stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT)
                info.update(driver_pid=child.pid, driver_identity=process_identity(child.pid))
                assert info['driver_identity'], 'Training process exited before its identity could be recorded'
                put(run / 'process.json', info)
                code = child.wait()
            info.update(exit_code=code, success=code == 0 and (run / 'SUCCESS').is_file(), finished_at=time.time())
        except BaseException as exc:
            info.update(success=False, error=type(exc).__name__ + ': ' + str(exc), finished_at=time.time())
            put(run / 'experiment_status.json', info)
            raise
        put(run / 'experiment_status.json', info)
        raise SystemExit(code or (0 if info['success'] else 1))


if __name__ == '__main__':
    main()
