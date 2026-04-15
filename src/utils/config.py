import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"YAML config is empty or invalid: {path}")
    return cfg


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