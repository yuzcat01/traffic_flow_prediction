from pathlib import Path
from typing import Union

import yaml


PathLike = Union[str, Path]


def load_yaml(path: PathLike):
    with open(Path(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"YAML config is empty or invalid: {path}")
    return cfg


def dump_yaml(path: PathLike, payload: dict):
    with open(Path(path), "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def deep_merge_dict(a: dict, b: dict):
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def merge_configs(*configs):
    merged = {}
    for cfg in configs:
        merged = deep_merge_dict(merged, cfg)
    return merged


def deep_update_dict(base: dict, updates: dict):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update_dict(base[key], value)
        else:
            base[key] = value
    return base
