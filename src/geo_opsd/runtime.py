"""Prepared experiment identity and CPU-only safety helpers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import load_settings, repo_root

STEPS = [100, 200, 300, 400, 500, 600, 669]


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    for name in ('DASHSCOPE_API_KEY', 'WANDB_API_KEY'):
        secret = os.environ.get(name, '')
        assert not secret or secret not in content, 'Refusing to persist a credential'
    temp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(content)
    try:
        if immutable:
            os.link(temp, path)
        else:
            os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepared(config):
    settings = load_settings(config)
    e = Path(settings['paths']['output']) / settings['experiment']['name']
    value = read(e / 'runtime.json')
    assert value['schema_version'] == 1
    expected = {
        'experiment_dir': e, 'run_dir': e / 'run', 'smoke_dir': e / 'smoke',
        'framework': Path(repo_root()) / 'framework', 'config_yaml': e / 'config.yaml',
        'model': e / 'model', 'source_model': Path(settings['paths']['model']),
        'train_parquet': e / 'data/train.parquet', 'smoke_parquet': e / 'data/smoke.parquet',
        'config_path': Path(settings['config_path']),
    }
    for key, path in expected.items():
        assert Path(value[key]).absolute() == path.absolute(), f'Prepared experiment identity changed: {key}'
    gpus = value['gpus']
    assert gpus == settings['experiment']['gpus'], 'GPU allocation changed; prepare a new experiment'
    assert len(gpus) == len(set(gpus)) == 6, 'This recipe uses exactly six GPUs, even on an eight-GPU host'
    assert all(isinstance(g, int) and not isinstance(g, bool) and g >= 0 for g in gpus)
    if 'recipe' in value:
        assert value['recipe'] == settings['experiment'].get('recipe', 'hidden_guidance'), 'Teacher recipe changed after prepare'
    if 'source_data' in value:
        assert Path(value['source_data']).resolve() == Path(settings['paths']['data']).resolve(), 'Dataset root changed after prepare'
    if 'wandb' in settings:
        for key in ('project', 'entity', 'mode'):
            assert value.get('wandb', {}).get(key) == settings['wandb'].get(key), 'W&B settings changed after prepare'
        if settings['wandb'].get('run_id'):
            assert value['wandb']['run_id'] == settings['wandb']['run_id']
    assert Path(value['python']).absolute() == Path(sys.executable).absolute(), 'Use the Python environment recorded during prepare'
    value['_settings'] = settings
    return value


def fingerprint(value):
    e = Path(value['experiment_dir'])
    root = Path(repo_root())
    files = [e / 'runtime.json', e / 'config.yaml', e / 'audit/original_resolved_config.yaml',
             e / 'audit/framework_alignment.json', e / 'DATA_PREPARED.json',
             e / 'data/train.parquet', e / 'data/smoke.parquet', Path(value['config_path'])]
    if (e / 'prompt_suffix.txt').exists():
        files.append(e / 'prompt_suffix.txt')
    for folder in (root / 'src', root / 'scripts', root / 'framework'):
        files += sorted(p for p in folder.rglob('*') if p.suffix in ('.py', '.yaml', '.yml'))
    files += sorted(p for p in (e / 'model').iterdir() if p.suffix in ('.json', '.jinja', '.txt', '.model', '.tiktoken'))
    entries = [str(path.absolute()) + ':' + sha(path) for path in files]
    data = read(e / 'DATA_PREPARED.json')
    if data.get('data_mode') == 'official':
        for key in ('source_train', 'source_test'):
            if data.get(key):
                assert sha(data[key]) == data[key + '_sha256'], 'Official source JSON changed after prepare'
        # Full hashes are checked during CPU preflight; metadata detects later normal edits.
        for record in data['images'].values():
            stat = Path(record['path']).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (record['bytes'], record['mtime_ns']), 'Official image changed after prepare'
    for path in sorted((e / 'model').glob('*.safetensors')):
        assert path.is_symlink(), 'Training overlay must link the original base weights'
        assert path.resolve() == (Path(value['source_model']) / path.name).resolve(), 'Overlay points outside the original model'
        stat = path.stat()
        entries.append(str(path) + ':' + str(path.resolve()) + ':' + str(stat.st_size) + ':' + str(stat.st_mtime_ns))
    assert entries
    return hashlib.sha256('\n'.join(entries).encode()).hexdigest()


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0], 'uid': path.stat().st_uid}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(value):
    current = process_identity(value['pid'])
    return bool(current and current['state'] != 'Z' and current['start_ticks'] == value['start_ticks'])


def free_gpus(gpus):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                   '--format=csv,noheader,nounits'], text=True, timeout=30)
    rows = {}
    for line in raw.splitlines():
        index, uuid, memory = [part.strip() for part in line.split(',')]
        rows[int(index)] = {'uuid': uuid, 'used_mib': int(memory)}
    assert all(g in rows and rows[g]['used_mib'] < 512 for g in gpus), 'Selected GPUs are not free; no process will be stopped'
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=30)
    selected = {rows[g]['uuid'] for g in gpus}
    assert not any(line.split(',')[0].strip() in selected for line in apps.splitlines()), 'Selected GPU has an active process'
    return {g: rows[g] for g in gpus}


def checkpoint(run, step, smoke=False):
    cp = Path(run) / 'checkpoints' / f'global_step_{step}'
    assert (cp / 'data.pt').is_file() and (cp / 'data.pt').stat().st_size > 0
    assert read(cp / 'actor/fsdp_config.json')['world_size'] == 6
    sizes = {}
    kinds = ('model', 'extra_state') if smoke else ('model', 'optim', 'extra_state')
    for kind in kinds:
        for rank in range(6):
            path = cp / 'actor' / f'{kind}_world_size_6_rank_{rank}.pt'
            assert path.is_file() and path.stat().st_size > 0, f'Missing six-rank shard: {path}'
            sizes[str(path.relative_to(cp))] = path.stat().st_size
    assert read(cp / 'actor/huggingface/config.json')['model_type'] == 'qwen3_vl'
    return cp, sizes


def finished(run, smoke=False):
    run = Path(run)
    status = read(run / 'experiment_status.json')
    assert status['success'] is True and status['exit_code'] == 0 and (run / 'SUCCESS').is_file()
    assert not same_process(read(run / 'process.json')['driver_identity']), 'Training process is still alive'
    assert (run / 'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == str(2 if smoke else 669)
