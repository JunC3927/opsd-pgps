"""Prepare portable paths/data/metadata on CPU without changing the training recipe."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .common import load_settings, put, read, repo_root, sha


def verify_framework(root):
    manifest = read(root / "provenance/framework_alignment.json")
    assert manifest["passed"] and manifest["tracked_files_checked"] == 482
    for name, expected in manifest["files_sha256"].items():
        assert sha(root / "framework" / name) == expected, "Changed historical framework: " + name
    return manifest


def prepare_model(source, target, root):
    import tokenizers
    expected = read(root / "provenance/model_files.json")
    cfg = read(source / "config.json")
    assert cfg["model_type"] == "qwen3_vl"
    assert cfg["architectures"] == ["Qwen3VLForConditionalGeneration"]
    index = read(source / "model.safetensors.index.json")
    assert set(index["weight_map"].values()) == set(expected["weights"])
    for name, info in expected["weights"].items():
        assert (source / name).is_file() and (source / name).stat().st_size == info["size"], name
    # save_pretrained serialized the old tokenizer differently from the official
    # download. Accept only these two audited inputs, then recreate the exact old
    # bytes with the pinned serializer. The original model remains read-only.
    immutable = ["vocab.json", "merges.txt", "tokenizer.json", "model.safetensors.index.json"]
    for name in immutable:
        allowed = {expected["files_sha256"][name], expected["official_source_sha256"][name]}
        assert sha(source / name) in allowed, "Original model file mismatch: " + name
    assert tokenizers.__version__ == "0.22.2", "Use the locked tokenizers serializer"
    target.mkdir()
    tokenizer = tokenizers.Tokenizer.from_file(str(source / "tokenizer.json"))
    tokenizer.save(str(target / "tokenizer.json"), pretty=True)
    tokenizer.model.save(str(target))
    for name in ("tokenizer.json", "vocab.json", "merges.txt"):
        assert sha(target / name) == expected["files_sha256"][name], "Historical tokenizer reconstruction mismatch: " + name
    for name in ["model.safetensors.index.json"] + list(expected["weights"]):
        (target / name).symlink_to(os.path.relpath(source / name, target))
    for name in expected["overlay_files"]:
        p = root / "configs/model_overlay" / name
        assert sha(p) == expected["files_sha256"][name], "Training model overlay changed: " + name
        shutil.copy2(p, target / name)
    return {"source_model": str(source), "model_type": cfg["model_type"],
            "immutable_metadata_sha256": {n: sha(source / n) for n in immutable},
            "restored_tokenizer_sha256": {n: sha(target / n) for n in ("tokenizer.json", "vocab.json", "merges.txt")},
            "tokenizers_version": tokenizers.__version__,
            "training_overlay_sha256": {n: sha(target / n) for n in expected["overlay_files"]},
            "weight_sizes": expected["weights"], "weight_bytes_hashed": False}


def prepare(settings):
    import yaml
    from .data import prepare_data
    root = repo_root()
    verify_framework(root)
    source = Path(settings["paths"]["model"])
    assert source.is_dir(), "Download the original Qwen3-VL-8B-Instruct model first"
    exp = Path(settings["paths"]["output"]) / settings["experiment"]["name"]
    assert not exp.exists(), "Experiment already exists; use a new name, never overwrite"
    assert not exp.is_relative_to(root / "framework")
    exp.mkdir(parents=True)
    put(exp / "PREPARATION_STARTED.json", {"time": time.time(), "settings": settings})
    try:
        suffix = (root / "configs/prompts/teacher_hidden_guidance.txt").read_text().rstrip("\n")
        if settings["experiment"]["recipe"] == "original":
            suffix = ""
        (exp / "prompt_suffix.txt").write_text(suffix)
        data = prepare_data(Path(settings["paths"]["data"]), exp / "data", suffix)
        model = prepare_model(source, exp / "model", root)
        audit = exp / "audit"
        audit.mkdir(exist_ok=True)
        for name in ("original_resolved_config.yaml", "framework_alignment.json"):
            shutil.copy2(root / "provenance" / name, audit / name)
        put(audit / "model_preparation.json", model)
        put(exp / "DATA_PREPARED.json", data)
        cfg = yaml.safe_load((root / "configs/train_base.yaml").read_text())
        cfg["data"]["train_files"] = [str(exp / "data/train.parquet")]
        cfg["data"]["custom_cls"]["path"] = str(root / "src/geo_opsd/dataset.py")
        cfg["actor_rollout_ref"]["model"]["path"] = str(exp / "model")
        wb = copy.deepcopy(settings["wandb"])
        if not wb.get("run_id"):
            wb["run_id"] = "pgps-" + __import__("hashlib").sha256((str(exp) + str(time.time_ns())).encode()).hexdigest()[:16]
        cfg["trainer"].update(experiment_name=settings["experiment"]["name"],
            default_local_dir=str(exp / "run/checkpoints"), rollout_data_dir=str(exp / "run/rollouts"),
            project_name=wb["project"], group_name="Qwen3-VL-8B-portable-six-gpu",
            logger=["console", "tensorboard"] + (["wandb"] if wb["mode"] != "disabled" else []),
            save_freq=100, max_actor_ckpt_to_keep=8)
        (exp / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        runtime = dict(schema_version=1, config_path=settings["config_path"],
            experiment_dir=str(exp), run_dir=str(exp / "run"), smoke_dir=str(exp / "smoke"),
            framework=str(root / "framework"), config_yaml=str(exp / "config.yaml"),
            model=str(exp / "model"), source_model=str(source),
            train_parquet=str(exp / "data/train.parquet"), smoke_parquet=str(exp / "data/smoke.parquet"),
            gpus=settings["experiment"]["gpus"], python=sys.executable, wandb=wb,
            recipe=settings["experiment"]["recipe"], repo_root=str(root))
        put(exp / "runtime.json", runtime)
        put(exp / "PREPARED.json", {"passed": True, "time": time.time(), "training_started": False,
            "gpu_work_started": False, "framework_files_checked": 482,
            "data": data, "model": model, "runtime": runtime})
        print(json.dumps({"prepared": str(exp), "training_started": False}))
        return exp
    except Exception as exc:
        put(exp / "PREPARATION_FAILED.json", {"error": repr(exc), "time": time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    prepare(load_settings(args.config))


if __name__ == "__main__":
    main()
