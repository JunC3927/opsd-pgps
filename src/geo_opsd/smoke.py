"""Verify two FA2/unpadded updates, six-rank export, and one real multimodal generation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .common import repo_root
from .runtime import prepared, read, put, sha, fingerprint, free_gpus, finished, checkpoint
from .export import export_checkpoint


def training_audit(value):
    from omegaconf import OmegaConf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from .train import assert_recipe
    run = Path(value['smoke_dir'])
    finished(run, True)
    cp, shards = checkpoint(run, 2, True)
    cfg = OmegaConf.load(run / 'resolved_config.yaml')
    assert_recipe(cfg)
    assert cfg.trainer.total_training_steps == cfg.trainer.save_freq == 2
    assert cfg.actor_rollout_ref.model.path == value['model']
    text = (run / 'train.log').read_text(errors='replace')
    for path in (run / 'runtime/ray').rglob('worker-*.out'):
        text += '\n' + path.read_text(errors='replace')
    assert 'Actor use_remove_padding=True' in text, 'Actual actor did not report unpadded execution'
    assert 'Actor use_remove_padding=False' not in text
    assert 'PGPS_LEGACY6_BATCH' not in text and 'PGPS_LOGPROB_RANK_EAGER' not in text
    accumulator = EventAccumulator(str(run / 'tensorboard'), size_guidance={'scalars': 0})
    accumulator.Reload()
    metrics = {}
    for tag in ['actor/vopd_loss', 'actor/grad_norm', 'response_length/mean',
                'response_length/clip_ratio', 'timing_s/step']:
        values = {entry.step: entry.value for entry in accumulator.Scalars(tag)}
        assert set(values) == {1, 2} and all(math.isfinite(x) for x in values.values()), tag
        if tag == 'actor/grad_norm':
            assert all(x > 0 for x in values.values())
        metrics[tag] = values
    for step, count in [(1, 12), (2, 6)]:
        rows = [json.loads(line) for line in (run / f'rollouts/{step}.jsonl').read_text().splitlines()]
        assert len(rows) == count and all(row['output'].strip() for row in rows)
        assert all('The following annotations describe' not in row['input'] for row in rows)
    return {'checkpoint': str(cp), 'source_shards': shards, 'metrics': metrics,
            'actual_actor_unpadded_log_verified': True, 'rollouts_per_step': [12, 6]}


def probe(value, gpu):
    """Run inside a short-lived child; its parent verifies complete GPU release."""
    assert gpu >= 0
    free_gpus([gpu])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_USE_V1='1')
    os.environ.pop('VLLM_ATTENTION_BACKEND', None)
    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    run = Path(value['smoke_dir'])
    stage = run / 'exports/step_0002'
    model = stage / 'model'
    assert not (stage / 'REAL_INFERENCE_PASSED.json').exists(), 'Successful smoke inference is already preserved'
    row = pq.read_table(value['smoke_parquet']).to_pylist()[0]
    student = row['prompt'][0]['content']
    assert student.startswith('<image>') and student.count('<image>') == 1
    question = student[len('<image>'):]
    image_path = Path(row['images'][0]['path'])
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    mm = {'min_pixels': 1024, 'max_pixels': 262144}
    processor = AutoProcessor.from_pretrained(str(model), trust_remote_code=True, **mm)
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors='pt')
    expected_tokens = int(encoded['input_ids'].shape[-1])
    assert expected_tokens + 128 <= 6144
    params = {'temperature': 0.0, 'max_tokens': 128, 'seed': 42, 'n': 1,
              'top_p': 1.0, 'top_k': -1, 'repetition_penalty': 1.0,
              'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'ignore_eos': False}
    attempt = stage / 'probe_attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    request = {'model': str(model), 'question_id': row['extra_info']['index'], 'question': question,
               'prompt': prompt, 'image_path': str(image_path), 'image_sha256': sha(image_path),
               'image_processor': mm, 'sampling_params': params, 'expected_prompt_tokens': expected_tokens,
               'purpose': 'engineering smoke only; not a benchmark evaluation'}
    put(attempt / 'request.json', request, True)
    engine = LLM(model=str(model), dtype='bfloat16', tensor_parallel_size=1,
                 gpu_memory_utilization=0.60, max_model_len=6144, max_num_seqs=1,
                 max_num_batched_tokens=4096, enforce_eager=True, seed=42,
                 mm_processor_kwargs=mm, limit_mm_per_prompt={'image': 1, 'video': 0},
                 generation_config='vllm')
    output = engine.generate([{'prompt': prompt, 'multi_modal_data': {'image': image}}],
                             SamplingParams(**params), use_tqdm=False)[0]
    answer = output.outputs[0]
    assert answer.text.strip() and answer.finish_reason in ('stop', 'length')
    assert len(output.prompt_token_ids) == expected_tokens, 'Processor/vLLM image-template input mismatch'
    result = {'passed': True, 'question_id': row['extra_info']['index'], 'model': str(model),
              'gpu': gpu, 'request': request, 'text': answer.text,
              'prompt_tokens': len(output.prompt_token_ids), 'completion_tokens': len(answer.token_ids),
              'finish_reason': answer.finish_reason, 'token_ids': list(answer.token_ids),
              'fingerprint': fingerprint(value), 'time': time.time(), 'judge_called': False}
    put(attempt / 'response.json', result, True)
    put(stage / 'REAL_INFERENCE_PASSED.json', result, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, required=True,
                        help='One explicitly selected free physical GPU; may be one of the two spare H100s')
    parser.add_argument('--probe-child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    value = prepared(args.config)
    if args.probe_child:
        probe(value, args.gpu)
        return
    e = Path(value['experiment_dir'])
    with (e / 'smoke_validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sig = fingerprint(value)
        assert read(e / 'CODE_FROZEN.json')['fingerprint'] == sig
        assert read(e / 'CPU_PREFLIGHT_PASSED.json')['fingerprint'] == sig
        audit = training_audit(value)
        model = export_checkpoint(value, 'smoke', 2)
        marker = model.parent / 'REAL_INFERENCE_PASSED.json'
        if not marker.exists():
            free_gpus([args.gpu])
            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join([str(Path(repo_root()) / 'src'), value['framework']])
            env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            for key in list(env):
                if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy') or key in ('DASHSCOPE_API_KEY', 'WANDB_API_KEY'):
                    env.pop(key)
            with (model.parent / 'probe.log').open('ab') as output:
                subprocess.run([value['python'], '-u', '-m', 'geo_opsd.smoke', '--config', value['config_path'],
                    '--gpu', str(args.gpu), '--probe-child'], cwd=repo_root(), env=env,
                    stdout=output, stderr=subprocess.STDOUT, check=True, start_new_session=True)
        result = read(marker)
        assert result['passed'] and result['fingerprint'] == sig and result['model'] == str(model)
        assert result['prompt_tokens'] == result['request']['expected_prompt_tokens']
        # A successful generation alone is insufficient: its process must have released the card.
        for retry in range(30):
            try:
                free_gpus([result['gpu']])
                break
            except AssertionError:
                if retry == 29:
                    raise
                time.sleep(1)
        assert fingerprint(value) == sig
        report = {'passed': True, 'fingerprint': sig, 'gpu_count': 6, 'time': time.time(),
                  'attention': 'flash_attention_2', 'original_six_rank_loss': True,
                  'real_exported_model_inference': True, 'real_probe_gpu_released': True,
                  'formal_initialization': 'original base model, never smoke weights', **audit}
        put(e / 'SMOKE_PASSED.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
