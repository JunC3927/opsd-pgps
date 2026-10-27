"""CPU merge of the six FSDP ranks, with original base inference metadata."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from .runtime import prepared, read, put, sha, checkpoint, finished, STEPS

METADATA = ('config.json', 'generation_config.json', 'preprocessor_config.json', 'processor_config.json',
            'video_preprocessor_config.json', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json')


def weights_audit(model):
    model = Path(model)
    files = sorted(model.glob('*.safetensors'))
    assert files and all(path.stat().st_size > 0 for path in files), 'Exported weights missing'
    index = model / 'model.safetensors.index.json'
    if index.exists():
        names = set(read(index)['weight_map'].values())
        assert names and names == {path.name for path in files}, 'Incomplete exported shard index'
    else:
        assert len(files) == 1 and files[0].name == 'model.safetensors'
    sizes = {path.name: path.stat().st_size for path in files}
    assert sum(sizes.values()) > 10_000_000_000, 'Export is too small for the BF16 8B model'
    return sizes


def restore_metadata(original, target, stage):
    original, target = Path(original), Path(target)
    assert read(original / 'config.json')['model_type'] == 'qwen3_vl'
    audit = {}
    for name in METADATA:
        source, dest = original / name, target / name
        if dest.exists() and (not source.exists() or dest.read_bytes() != source.read_bytes()):
            archive = stage / 'exporter_metadata' / name
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                shutil.copy2(dest, archive)
        if source.exists():
            shutil.copy2(source, dest)
            audit[name] = {'present': True, 'sha256': sha(dest)}
            assert dest.read_bytes() == source.read_bytes()
        else:
            if dest.exists():
                dest.unlink()
            audit[name] = {'present': False}
    return audit


def final_link(run, step, target, smoke):
    if smoke or step != 669:
        return
    link = run / 'final_model'
    if link.exists() or link.is_symlink():
        assert link.is_symlink() and link.resolve() == target.resolve()
    else:
        link.symlink_to(target, target_is_directory=True)


def export_checkpoint(value, mode, step):
    smoke = mode == 'smoke'
    assert (step == 2 if smoke else step in STEPS), 'Unexpected checkpoint step'
    run = Path(value['smoke_dir' if smoke else 'run_dir'])
    finished(run, smoke)
    cp, shards = checkpoint(run, step, smoke)
    stage = run / 'exports' / f'step_{step:04d}'
    stage.mkdir(parents=True, exist_ok=True)
    target = stage / 'model'
    with (stage / 'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = stage / 'EXPORT_COMPLETE.json'
        if marker.exists():
            result = read(marker)
            assert result['source'] == str(cp) and result['model'] == str(target)
            assert result['world_size'] == 6 and result['source_shards'] == shards
            assert weights_audit(target) == result['weight_sizes']
            for name, entry in result['metadata'].items():
                assert (target / name).exists() == entry['present']
                if entry['present']:
                    assert sha(target / name) == entry['sha256'] == sha(Path(value['source_model']) / name)
            final_link(run, step, target, smoke)
            return target
        assert not target.exists(), 'Unclaimed export exists; preserve it and inspect before recovery'
        partial = stage / f'model.partial.{time.time_ns()}'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=value['framework'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for key in ('DASHSCOPE_API_KEY', 'WANDB_API_KEY'):
            env.pop(key, None)
        with (stage / 'export.log').open('ab') as output:
            subprocess.run([value['python'], '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                '--local_dir', str(cp / 'actor'), '--target_dir', str(partial)], cwd=value['framework'],
                env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        sizes = weights_audit(partial)
        metadata = restore_metadata(value['source_model'], partial, stage)
        partial.rename(target)
        put(marker, {'step': step, 'source': str(cp), 'model': str(target), 'world_size': 6,
                     'source_shards': shards, 'weight_sizes': sizes, 'metadata': metadata, 'time': time.time()}, True)
    final_link(run, step, target, smoke)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    parser.add_argument('--step', required=True, type=int)
    args = parser.parse_args()
    model = export_checkpoint(prepared(args.config), args.mode, args.step)
    print(str(model), flush=True)


if __name__ == '__main__':
    main()
