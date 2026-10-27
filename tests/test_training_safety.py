"""CPU/stdlib checks. No CUDA, model download, remote host, or paid API is used."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geo_opsd import runtime
from geo_opsd.preflight import compare_recipe
from geo_opsd.train import assert_recipe
from geo_opsd.export import restore_metadata


class Config(dict):
    def __getattr__(self, key):
        return self[key]


def cfg(value):
    return Config({key: cfg(item) for key, item in value.items()}) if isinstance(value, dict) else value


def recipe():
    return cfg({'actor_rollout_ref': {
        'model': {'use_remove_padding': True, 'override_config': {}},
        'actor': {'use_remove_padding': True, 'ppo_mini_batch_size': 12, 'ppo_micro_batch_size_per_gpu': 1,
                  'rollout_n': 1, 'self_distillation': {'alpha': 0.0, 'teacher_update_rate': 0.0},
                  'optim': {'lr': 1e-6, 'lr_scheduler_type': 'constant', 'lr_warmup_steps': 0}},
        'rollout': {'n': 1, 'response_length': 4096, 'max_model_len': 6144, 'temperature': 1.0,
                    'top_p': 1.0, 'top_k': -1, 'ignore_eos': False,
                    'engine_kwargs': {'vllm': {'mm_processor_kwargs': {'min_pixels': 1024, 'max_pixels': 262144}}}}},
        'trainer': {'n_gpus_per_node': 6, 'nnodes': 1, 'total_epochs': 1, 'resume_mode': 'disable',
                    'test_freq': -1, 'val_before_train': False, 'max_actor_ckpt_to_keep': 8},
        'data': {'train_batch_size': 12, 'train_drop_last': False, 'max_prompt_length': 2048, 'max_response_length': 4096},
        'algorithm': {'rollout_correction': {'rollout_is': None}}})


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TrainingSafetyTests(unittest.TestCase):
    def test_scientific_guard_uses_original_function(self):
        source = (ROOT / 'src/geo_opsd/train.py').read_text()
        tree = ast.parse(source)
        guard = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'assert_recipe')
        # Exact audited function source: stable across Python AST schema changes
        # (Python 3.12 adds FunctionDef.type_params to ast.dump).
        original_function = ast.get_source_segment(source, guard)
        self.assertIsNotNone(original_function)
        self.assertEqual(hashlib.sha256(original_function.encode()).hexdigest(),
                         'a199d578b14f85bf57f5c07eee8536819b110d1a5df18b93f3f385d957cc0e80')

    def test_guard_rejects_silent_eight_gpu_or_science_changes(self):
        assert_recipe(recipe())
        for path, new in [('trainer.n_gpus_per_node', 8), ('actor_rollout_ref.rollout.n', 4),
                          ('actor_rollout_ref.model.use_remove_padding', False),
                          ('actor_rollout_ref.model.override_config.attn_implementation', 'sdpa'),
                          ('actor_rollout_ref.actor.optim.lr', 2e-6), ('data.max_response_length', 8192),
                          ('actor_rollout_ref.rollout.presence_penalty', 1.5)]:
            value = recipe()
            target = value
            for key in path.split('.')[:-1]:
                target = target[key]
            target[path.split('.')[-1]] = new
            with self.subTest(path=path), self.assertRaises(AssertionError):
                assert_recipe(value)

    def test_full_recipe_comparison_allows_paths_not_math(self):
        old = {'data': {'train_files': ['/old'], 'train_batch_size': 12},
               'trainer': {'save_freq': 669}, 'actor': {'loss_coef': 1.0}}
        new = copy.deepcopy(old)
        new['data']['train_files'] = ['/new']
        new['trainer']['save_freq'] = 100
        self.assertEqual(len(compare_recipe(old, new)), 2)
        new['actor']['loss_coef'] = 2.0
        with self.assertRaisesRegex(AssertionError, 'loss_coef'):
            compare_recipe(old, new)

    def test_runtime_rejects_changed_allocation_or_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            settings = {'config_path': str(base / 'local.yaml'), 'paths': {'output': str(base), 'model': str(base / 'base-model')},
                        'experiment': {'name': 'experiment', 'gpus': list(range(6))}}
            e = base / 'experiment'
            value = {'schema_version': 1, 'experiment_dir': str(e), 'run_dir': str(e / 'run'), 'smoke_dir': str(e / 'smoke'),
                     'framework': str(ROOT / 'framework'), 'config_yaml': str(e / 'config.yaml'), 'model': str(e / 'model'),
                     'source_model': settings['paths']['model'], 'train_parquet': str(e / 'data/train.parquet'),
                     'smoke_parquet': str(e / 'data/smoke.parquet'), 'config_path': settings['config_path'],
                     'gpus': list(range(6)), 'python': sys.executable}
            save(e / 'runtime.json', value)
            with patch.object(runtime, 'load_settings', return_value=settings):
                self.assertEqual(runtime.prepared('unused')['gpus'], list(range(6)))
                value['gpus'] = list(range(8));save(e / 'runtime.json', value)
                with self.assertRaisesRegex(AssertionError, 'allocation changed'):
                    runtime.prepared('unused')
                value['gpus'] = list(range(6));value['run_dir'] = str(base / 'old-run');save(e / 'runtime.json', value)
                with self.assertRaisesRegex(AssertionError, 'run_dir'):
                    runtime.prepared('unused')

    def test_six_rank_shards_required(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp);cp = run / 'checkpoints/global_step_2';actor = cp / 'actor'
            save(actor / 'fsdp_config.json', {'world_size': 6})
            save(actor / 'huggingface/config.json', {'model_type': 'qwen3_vl'})
            (cp / 'data.pt').write_bytes(b'x')
            for kind in ('model', 'extra_state'):
                for rank in range(6):
                    (actor / f'{kind}_world_size_6_rank_{rank}.pt').write_bytes(b'x')
            self.assertEqual(len(runtime.checkpoint(run, 2, True)[1]), 12)
            (actor / 'model_world_size_6_rank_5.pt').unlink()
            with self.assertRaisesRegex(AssertionError, 'rank_5'):
                runtime.checkpoint(run, 2, True)

    def test_gpu_guard_does_not_take_busy_or_active_zero_memory_card(self):
        with patch.object(runtime.subprocess, 'check_output', side_effect=['0, gpu0, 0\n1, gpu1, 0\n', '']):
            self.assertEqual(set(runtime.free_gpus([0, 1])), {0, 1})
        with patch.object(runtime.subprocess, 'check_output', return_value='0, gpu0, 4000\n'):
            with self.assertRaisesRegex(AssertionError, 'not free'):
                runtime.free_gpus([0])
        with patch.object(runtime.subprocess, 'check_output', side_effect=['0, gpu0, 0\n', 'gpu0, 123\n']):
            with self.assertRaisesRegex(AssertionError, 'active process'):
                runtime.free_gpus([0])

    def test_export_restores_base_eos_and_processor(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp);original = base / 'original';target = base / 'model';stage = base / 'stage'
            stage.mkdir();target.mkdir()
            save(original / 'config.json', {'model_type': 'qwen3_vl', 'eos_token_id': 151645})
            save(original / 'preprocessor_config.json', {'max_pixels': 1048576})
            save(target / 'config.json', {'model_type': 'qwen3_vl', 'eos_token_id': 151643})
            audit = restore_metadata(original, target, stage)
            self.assertTrue(audit['config.json']['present'])
            self.assertEqual(runtime.read(target / 'config.json')['eos_token_id'], 151645)
            self.assertEqual(runtime.read(stage / 'exporter_metadata/config.json')['eos_token_id'], 151643)


if __name__ == '__main__':
    unittest.main(verbosity=2)
