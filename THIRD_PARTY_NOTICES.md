# Sources and modifications

`framework/` preserves the reviewed Vision-OPD/verl source at commit
`06860e69b5ed9dc24e96ca5c855f3a4ef25976aa` with the historical PGPS experiment changes.
It includes the original Apache-2.0 `LICENSE`, source copyright headers and existing
attribution. All 482 tracked files are recorded in `provenance/framework_alignment.json`.

Historical experiment modifications, compared with that upstream commit, are in:

- `verl/workers/actor/dp_actor.py`
- `verl/workers/config/actor.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`

The first five files have exact archived hashes from the earlier successful run.
The sixth matches the historical context-length compatibility patch. The remaining
files are reconstructed from the pinned Git commit. The portable entry points under
`src/geo_opsd/` and `scripts/` are new; they do not change the frozen training framework.

The configuration archive has historical filesystem locations replaced with
`/historical/...` placeholders. The actual old training settings are otherwise
preserved and compared with the destination's fully resolved configuration before
training. Paths, logging and requested checkpoint retention/frequency are the allowed
differences. The optional teacher suffix is recorded separately from source changes.

`configs/model_overlay/` contains small Qwen3-VL training metadata and chat-template
files preserved from the prior experiment. It contains no model weights or full
tokenizer vocabulary. Original model files must be downloaded separately and are
checked against `provenance/model_files.json`.

PGPS9K data/images, model weights and historical inference/judge outputs are not
redistributed through Git. Their original terms remain applicable.
