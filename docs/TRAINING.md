# Training on a new server

These commands are manual. Preparing the repository or finding free GPUs never starts training. No scheduler, background monitor, remote SSH login, or paid judge is installed by this workflow.

The destination may have eight H100s, but this recipe deliberately uses **six GPUs**. The remaining two cards may be used for independent evaluation. Changing to eight training ranks is a different experiment and is rejected by this implementation.

## Default one-command entry

Set `paths.data` to the official download root containing `Diagram_Visual` and `PGPS9K/train.json` (or flat `train.json`), then run:

```bash
python scripts/run.py --config configs/local.yaml
```

This explicitly starts CPU preparation, two smoke updates, export/inference checks, and then formal training. Use `--prepare-only` for CPU checks without GPU work. No manual Parquet export or image conversion is needed. Images remain in the download directory; a small automatic trainer cache is written under the experiment. The historical model processor settings remain; using official raw images instead of historical clight_images is recorded as a data-input change. Optional historical bundles remain supported.

## Preserved recipe

The complete resolved configuration is compared against the archived original six-rank experiment, with only relocated paths, logging, checkpoint frequency and retention allowed to differ. The model is the original Qwen3-VL-8B-Instruct. Both student and frozen teacher start from that base; the teacher additionally sees the original geometry relations and, for the `hidden_guidance` recipe, the selected teacher constraint.

The fixed scientific settings include:

| Setting | Value |
|---|---|
| Questions / physical parquet rows | 8,021 / 8,022 |
| Global training batch / final batch | 12 / 6 |
| Rollouts per question | 1 |
| Response cap | 4,096 |
| Temperature / top p / top k | 1 / 1 / −1 |
| Presence / frequency / repetition penalties | 0 / 0 / 1 |
| Learning rate / schedule | 1e−6 / constant |
| Epochs / updates | 1 / 669 |
| Training ranks | 6 |
| Attention / padding | FlashAttention2 / remove padding enabled |
| Saved checkpoints | 100, 200, 300, 400, 500, 600, 669 |

The original framework and loss implementation are used directly. There is no four-rank compatibility adapter and no new-vLLM eager-rank patch. H100 numerical behavior is not promised to be bitwise identical to A800; the two-step smoke is required on the destination hardware.

## Local configuration and prepared layout

Follow the main README to install the pinned environment, download the base model on the destination, extract the official PGPS9K download, and prepare a uniquely named experiment using `configs/local.yaml`.

Relative `paths.model`, `paths.data`, and `paths.output` are resolved relative to that YAML file. Do not put credentials in YAML or in Git. W&B defaults to offline logging; to use online logging, log in on the destination and set `wandb.mode: online` with your own project/entity.

Preparation creates:

```text
<paths.output>/<experiment.name>/          E
  runtime.json                            Paths, six GPU IDs, Python and W&B identity
  config.yaml                             Training override
  audit/original_resolved_config.yaml      Full original configuration
  audit/framework_alignment.json          Frozen framework file hashes
  data/train.parquet                      8022 auto-generated rows with official image paths
  data/smoke.parquet                      First 18 rows, in the same order
  model/                                 Original weights plus original training metadata overlay
  prompt_suffix.txt                       Constraint, or empty for the original control
  run/                                   Formal training output
  smoke/                                 Two-step smoke output
```

Preparation does not download models or copy a Python environment into the repository. An existing prepared experiment is not silently repurposed for a different allocation, model, recipe or output directory.

## Explicit execution

Run all commands from the repository root in the destination's pinned environment.

1. **CPU preflight.** This hides CUDA devices, resolves and compares the complete configuration, verifies the restored framework and dependencies, checks all data rows and image paths, bounds teacher input lengths, and processes the longest teacher prompt with its actual image.

   ```bash
   python scripts/preflight.py --config configs/local.yaml
   ```

   Passing writes `CPU_PREFLIGHT_PASSED.json` and `CODE_FROZEN.json`. The native FA2 extension is imported here, but GPU execution is not claimed. Configuration, source, data and model metadata changes invalidate the fingerprint.

2. **Two real six-GPU updates.** First check that the six configured physical GPUs are free. The launcher independently verifies memory use and the active-process list, and refuses to stop or replace another job.

   ```bash
   python scripts/launch.py --config configs/local.yaml --mode smoke
   ```

   The first update uses 12 examples and the second uses six. Logs and checkpoints are stored under `E/smoke`. This command stays attached until training exits; use your normal terminal multiplexer if needed.

3. **Export and test the smoke model.** Select one free physical GPU explicitly. On an eight-H100 host, GPU 6 or 7 may be convenient; this choice is independent of the six training ranks.

   ```bash
   python scripts/smoke_check.py --config configs/local.yaml --gpu 6
   ```

   The command requires finite loss/gradient metrics at both updates, actual unpadded actor logs, all six rank shards and completed training. It merges the model on CPU using the frozen framework, restores the original base inference metadata, and performs one real multimodal generation from the first smoke training question. The processor's expected input token count must match vLLM's actual count. Only after the probe process releases its GPU is `SMOKE_PASSED.json` written.

   The one-question probe uses temperature 0 and a 128-token output cap solely to check the exported model and image/template handling. It is **not an accuracy evaluation**, does not call a judge, and may legitimately end at the short cap. Its input, response, token IDs and finish reason are retained under `E/smoke/exports/step_0002/`. The formal training rollout parameters remain the table above.

4. **Formal training from the original base.** Smoke weights are never used as initialization.

   ```bash
   python scripts/launch.py --config configs/local.yaml --mode train
   ```

   This requires the matching CPU and smoke fingerprints and six free GPUs. It writes `E/run/train.log`, TensorBoard data under `E/run/tensorboard`, rollout JSONL and all seven checkpoints. W&B logs use the configured account and mode; smoke always disables W&B. There is no in-training API evaluation.

5. **Export a completed formal checkpoint.** Export is allowed after the complete 669-step run has successfully exited, preserving its `data.pt`, model, optimizer and extra-state shards.

   ```bash
   python scripts/export.py --config configs/local.yaml --mode train --step 669
   ```

   The model appears at `E/run/exports/step_0669/model`; `E/run/final_model` points to it. Use another saved step to export that checkpoint. Export runs on CPU and does not start vLLM or a judge. Pass the resulting model directory to the separate Unified Eval workflow described in the main README.

## Recovery and evidence

The launcher refuses to overwrite an existing process record, training log or checkpoint. Do not blindly repeat a failed training launch: inspect process identity, logs and checkpoints first. No automatic training resume is claimed by this recipe (`resume_mode=disable`).

Successful exports and smoke generations are retained. Re-running the smoke checker reuses its existing successful generation, verifies its fingerprint and checks GPU release; it does not resample it. Partial exports and failed attempt directories remain available for diagnosis. Existing base metadata is restored during export because the training overlay uses different EOS/padding metadata; inference should not accidentally inherit those training overrides.

No command kills an unrelated GPU process. If an inference child fails to release its GPU, the smoke gate fails and formal training remains blocked.

## Local tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_training_safety.py' -v
```

These standard-library tests require no model, GPU or API. They check the preserved recipe guard, refusal of scientific parameter changes and eight-rank substitution, runtime identity, complete six-rank checkpoints, GPU occupancy checks and restoration of original inference metadata. A passing local suite is not a substitute for destination CPU preflight and the real six-GPU smoke.
