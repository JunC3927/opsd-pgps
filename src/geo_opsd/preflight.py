"""CPU-only alignment audit; importing FA2 is not a substitute for the GPU smoke."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from .runtime import prepared, read, put, sha, fingerprint

PATH_AND_LOG_KEYS = {
    'data.train_files', 'data.custom_cls.path', 'actor_rollout_ref.model.path',
    'actor_rollout_ref.rollout.prometheus.served_model_name', 'critic.model.tokenizer_path',
    'reward_model.model.input_tokenizer', 'trainer.experiment_name', 'trainer.default_local_dir',
    'trainer.rollout_data_dir', 'trainer.group_name', 'trainer.project_name', 'trainer.logger',
    'trainer.save_freq', 'trainer.max_actor_ckpt_to_keep',
}


def flatten(value, prefix=''):
    if isinstance(value, dict):
        return {key: child for name, item in value.items()
                for key, child in flatten(item, prefix + '.' + name if prefix else name).items()}
    return {prefix: value}


def compare_recipe(old, current):
    before, after = flatten(old), flatten(current)
    changes = [{'key': key, 'old': before.get(key), 'new': after.get(key)}
               for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
    forbidden = [change for change in changes if change['key'] not in PATH_AND_LOG_KEYS]
    assert not forbidden, 'Scientific recipe differs from the original six-rank run: ' + json.dumps(forbidden)
    return changes


def prompt_audit(value):
    import pyarrow.parquet as pq
    from transformers import AutoProcessor
    from PIL import Image
    e = Path(value['experiment_dir'])
    table = pq.read_table(value['train_parquet'])
    smoke = pq.read_table(value['smoke_parquet'])
    assert len(table) == 8022 and len(smoke) == 18
    assert table.slice(0, 18).equals(smoke, check_metadata=True)
    rows = table.to_pylist()
    assert len({row['extra_info']['index'] for row in rows}) == 8021
    assert sum(bool(row['extra_info']['sampler_padding']) for row in rows) == 1
    source = read(e / 'DATA_PREPARED.json')
    assert source['passed'] and source['rows'] == 8022 and source['unique_questions'] == 8021
    assert source['smoke_rows'] == 18
    if source.get('data_mode') == 'official':
        from .official_data import verify_sources
        assert source['original_prompt_and_parsing_verified']
        verify_sources(source)
    else:
        assert source['only_teacher_suffix_changed']
    processor = AutoProcessor.from_pretrained(value['model'], trust_remote_code=True)
    pre = read(Path(value['model']) / 'preprocessor_config.json')
    assert pre['min_pixels'] == 1024 and pre['max_pixels'] == 262144
    assert processor.image_processor.patch_size == 16 and processor.image_processor.merge_size == 2
    suffix = (e / 'prompt_suffix.txt').read_text().rstrip('\n')
    recipe = value['_settings']['experiment'].get('recipe', 'hidden_guidance')
    assert recipe in ('hidden_guidance', 'original')
    assert bool(suffix) == (recipe == 'hidden_guidance'), 'Teacher suffix does not match the selected recipe'
    maximum = (0, None)
    for row in rows:
        assert len(row['prompt']) == len(row['teacher_prompt']) == 1
        student, teacher = row['prompt'][0], row['teacher_prompt'][0]
        assert student['role'] == teacher['role'] == 'user'
        assert student['content'].startswith('<image>') and student['content'].count('<image>') == 1
        assert teacher['content'].startswith(student['content'] + '\n\n')
        if suffix:
            assert teacher['content'].endswith('\n\n' + suffix)
        assert '\nStructural relations:\n' in teacher['content'] and '\nSemantic relations:\n' in teacher['content']
        assert 'The following annotations describe' not in student['content']
        assert row['reward_model']['ground_truth'] == ''
        assert row['images'] == row['teacher_images'] and len(row['images']) == 1
        assert Path(row['images'][0]['path']).is_file(), 'Missing relocated training image'
        count = len(processor.tokenizer.encode(teacher['content'], add_special_tokens=False))
        if count > maximum[0]:
            maximum = (count, row)
    assert maximum[0] + 1024 < 4096, 'Conservative teacher input bound exceeds reprompt capacity'
    row = maximum[1]
    with Image.open(row['teacher_images'][0]['path']) as image:
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image.convert('RGB')},
            {'type': 'text', 'text': row['teacher_prompt'][0]['content'][len('<image>'):]}]}]
        encoded = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt')
    count = int(encoded['input_ids'].shape[-1])
    assert count < 4096
    assert suffix in processor.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=False)
    result = {'passed': True, 'rows': 8022, 'unique_questions': 8021, 'smoke_rows': 18,
              'max_teacher_text_tokens': maximum[0], 'teacher_input_upper_bound': maximum[0] + 1024,
              'longest_teacher_actual_tokens': count, 'suffix_sha256': sha(e / 'prompt_suffix.txt')}
    put(e / 'PROMPT_PREFLIGHT_PASSED.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    value = prepared(args.config)
    e = Path(value['experiment_dir'])
    framework = Path(value['framework'])
    sys.path.insert(0, str(framework))
    from omegaconf import OmegaConf
    from .train import config_for
    from verl.utils.config import validate_config
    import verl
    assert Path(verl.__file__).resolve().is_relative_to(framework.resolve())
    cfg = config_for(value, 'train')
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    old = OmegaConf.to_container(OmegaConf.load(e / 'audit/original_resolved_config.yaml'), resolve=True)
    current = OmegaConf.to_container(cfg, resolve=True)
    changes = compare_recipe(old, current)
    put(e / 'resolved_config_comparison.json', changes)
    alignment = read(e / 'audit/framework_alignment.json')
    assert alignment['passed'] and alignment['files_sha256']
    for name, expected in alignment['files_sha256'].items():
        assert sha(framework / name) == expected, 'Framework changed: ' + name
    assert not (framework / 'verl/utils/legacy6_adapter.py').exists()
    for path in framework.rglob('*.py'):
        assert 'PGPS_LOGPROB_RANK_EAGER' not in path.read_text(errors='replace'), 'New-stack patch is not part of the old recipe'
    assert 'override_model_config.get("attn_implementation", "flash_attention_2")' in (framework / 'verl/workers/fsdp_workers.py').read_text()
    versions = {}
    for package, expected in [('torch', '2.8.0'), ('transformers', '4.57.3'), ('vllm', '0.11.0'), ('flash-attn', '2.8.3')]:
        actual = importlib.metadata.version(package)
        assert actual.split('+', 1)[0] == expected, (package, expected, actual)
        versions[package] = actual
    extension = importlib.import_module('flash_attn_2_cuda')
    assert hasattr(extension, 'fwd') and hasattr(extension, 'bwd')
    prompt = prompt_audit(value)
    OmegaConf.save(cfg, e / 'planned_resolved_config.yaml', resolve=True)
    result = {'passed': True, 'time': time.time(), 'fingerprint': fingerprint(value),
              'gpu_count': 6, 'original_six_rank_loss': True, 'attention': 'flash_attention_2',
              'actor_remove_padding': True, 'versions': versions, 'config_differences': changes,
              'framework_files_checked': len(alignment['files_sha256']), 'prompt_audit': prompt,
              'python': sys.executable, 'gpu_smoke_still_required': True}
    put(e / 'CPU_PREFLIGHT_PASSED.json', result)
    put(e / 'CODE_FROZEN.json', {'fingerprint': result['fingerprint'], 'time': time.time()})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
