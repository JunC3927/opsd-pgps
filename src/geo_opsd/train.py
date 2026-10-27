"""Original six-rank training entry; launch through scripts/launch.py."""
import argparse
import os
from pathlib import Path
import sys
from .runtime import prepared, read, fingerprint


def assert_recipe(cfg):
    model = cfg.actor_rollout_ref.model
    actor = cfg.actor_rollout_ref.actor
    rollout = cfg.actor_rollout_ref.rollout
    assert model.use_remove_padding is True and actor.use_remove_padding is True
    # The restored fsdp_workers factory defaults to FA2, exactly as in the old run.
    assert model.override_config.get('attn_implementation', 'flash_attention_2') == 'flash_attention_2'
    assert cfg.trainer.n_gpus_per_node == 6 and cfg.trainer.nnodes == 1
    assert cfg.data.train_batch_size == actor.ppo_mini_batch_size == 12
    assert actor.ppo_micro_batch_size_per_gpu == 1
    assert actor.rollout_n == rollout.n == 1
    assert cfg.data.train_drop_last is False
    assert cfg.data.max_prompt_length == 2048 and cfg.data.max_response_length == rollout.response_length == 4096
    assert rollout.max_model_len == 6144
    assert rollout.temperature == rollout.top_p == 1.0 and rollout.top_k == -1
    assert rollout.ignore_eos is False
    assert rollout.get('presence_penalty', 0.0) == rollout.get('frequency_penalty', 0.0) == 0.0
    assert rollout.get('repetition_penalty', 1.0) == 1.0
    assert rollout.engine_kwargs.vllm.mm_processor_kwargs == {'min_pixels': 1024, 'max_pixels': 262144}
    assert actor.self_distillation.alpha == 0.0 and actor.self_distillation.teacher_update_rate == 0.0
    assert actor.optim.lr == 1e-6 and actor.optim.lr_scheduler_type == 'constant'
    assert actor.optim.lr_warmup_steps == 0
    assert cfg.algorithm.rollout_correction.rollout_is is None
    assert cfg.trainer.total_epochs == 1 and cfg.trainer.resume_mode == 'disable'
    assert cfg.trainer.test_freq == -1 and cfg.trainer.val_before_train is False
    assert cfg.trainer.max_actor_ckpt_to_keep >= 7
    assert not cfg.data.get('apply_chat_template_kwargs', {}), 'No Qwen3.5 thinking override'


def config_for(value, mode):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    e = Path(value['experiment_dir'])
    framework = Path(value['framework'])
    run = Path(value['run_dir' if mode == 'train' else 'smoke_dir'])
    with initialize_config_dir(config_dir=str(framework / 'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='vopd')
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(value['config_yaml']))
    cfg.data.train_files = [value['train_parquet' if mode == 'train' else 'smoke_parquet']]
    cfg.actor_rollout_ref.model.path = value['model']
    cfg.trainer.experiment_name = e.name + ('-smoke' if mode == 'smoke' else '')
    cfg.trainer.default_local_dir = str(run / 'checkpoints')
    cfg.trainer.rollout_data_dir = str(run / 'rollouts')
    model = read(Path(value['model']) / 'config.json')
    assert model['model_type'] == 'qwen3_vl' and model['architectures'] == ['Qwen3VLForConditionalGeneration']
    if mode == 'smoke':
        cfg.trainer.total_training_steps = 2
        cfg.trainer.save_freq = 2
        cfg.actor_rollout_ref.actor.checkpoint.save_contents = ['model', 'extra']
        cfg.trainer.logger = ['console', 'tensorboard']
    else:
        assert cfg.trainer.total_training_steps is None and cfg.trainer.save_freq == 100
        assert set(cfg.actor_rollout_ref.actor.checkpoint.save_contents) >= {'model', 'optimizer', 'extra'}
        cfg.trainer.logger = ['console', 'tensorboard']
        if (value.get('wandb', {}).get('mode') or 'offline') != 'disabled':
            cfg.trainer.logger.append('wandb')
        cfg.trainer.project_name = value.get('wandb', {}).get('project') or 'geometry-opsd'
    assert_recipe(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args()
    value = prepared(args.config)
    framework = Path(value['framework'])
    sys.path.insert(0, str(framework))
    from omegaconf import OmegaConf
    from verl.utils.config import validate_config
    import verl
    assert Path(verl.__file__).resolve().is_relative_to(framework.resolve()), 'Wrong framework import'
    cfg = config_for(value, args.mode)
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    run = Path(value['run_dir' if args.mode == 'train' else 'smoke_dir'])
    run.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / 'resolved_config.yaml', resolve=True)
    if args.config_only:
        return
    e = Path(value['experiment_dir'])
    sig = fingerprint(value)
    assert read(e / 'CPU_PREFLIGHT_PASSED.json')['fingerprint'] == sig
    assert read(e / 'CODE_FROZEN.json')['fingerprint'] == sig
    if args.mode == 'train':
        assert read(e / 'SMOKE_PASSED.json')['fingerprint'] == sig
    assert os.environ.get('GEO_OPSD_LAUNCHED') == str(e), 'Start through scripts/launch.py'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(map(str, value['gpus']))
    os.environ['TENSORBOARD_DIR'] = str(run / 'tensorboard')
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device
    auto_set_device(cfg)
    run_ppo(cfg)
    (run / 'SUCCESS').write_text('Training completed successfully.\n')
    import ray
    ray.shutdown()


if __name__ == '__main__':
    main()
