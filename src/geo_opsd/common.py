"""Configuration and filesystem helpers; importing this module starts no jobs."""
import hashlib
import json
import os
from pathlib import Path
import re


def repo_root():
    return Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for name in ("DASHSCOPE_API_KEY", "WANDB_API_KEY", "HF_TOKEN"):
        secret = os.environ.get(name, "")
        if secret and secret in text:
            raise ValueError("Refusing to write a credential")
    tmp = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_settings(path):
    import yaml
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 configuration")
    allowed = {"schema_version", "paths", "experiment", "wandb"}
    if set(value) - allowed:
        raise ValueError("Unknown configuration fields: " + repr(set(value) - allowed))
    if set(value.get("paths", {})) != {"model", "data", "output"}:
        raise ValueError("paths must contain model, data and output")
    for key, raw in value["paths"].items():
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Invalid path: " + key)
        expanded = Path(os.path.expandvars(raw)).expanduser()
        if "$" in str(expanded):
            raise ValueError("Unexpanded path variable: " + key)
        value["paths"][key] = str((path.parent / expanded).resolve())
    exp = value.get("experiment", {})
    if set(exp) - {"name", "gpus", "recipe"}:
        raise ValueError("Unknown experiment field")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", exp.get("name", "")):
        raise ValueError("Use a unique, short experiment.name")
    gpus = exp.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != 6 or len(set(gpus)) != 6:
        raise ValueError("This aligned recipe requires exactly six GPUs, including on an eight-GPU host")
    if any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("GPU identifiers must be six distinct nonnegative integers")
    exp.setdefault("recipe", "hidden_guidance")
    if exp["recipe"] not in ("original", "hidden_guidance"):
        raise ValueError("Supported recipes: original, hidden_guidance")
    wb = value.setdefault("wandb", {})
    if set(wb) - {"mode", "project", "entity", "run_id"}:
        raise ValueError("Only non-secret W&B configuration is accepted; use wandb login separately")
    wb.setdefault("mode", "offline")
    wb.setdefault("project", "PGPS9K-Parsing-OPSD")
    wb.setdefault("entity", None)
    wb.setdefault("run_id", None)
    if wb["mode"] not in ("offline", "online", "disabled"):
        raise ValueError("wandb.mode must be offline, online or disabled")
    value["config_path"] = str(path)
    return value
