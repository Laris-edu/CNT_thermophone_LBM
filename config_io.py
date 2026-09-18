"""YAML include 读取；继承原项目递归加载和顶层浅合并语义。"""
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    include = config.get("include")
    if include:
        parent = load_config(path.parent / include)
        parent.update({k: v for k, v in config.items() if k != "include"})
        return parent
    return config
