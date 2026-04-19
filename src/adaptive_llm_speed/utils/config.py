"""YAML config loader with shallow include support.

A config may reference another YAML via the special keys `ffn_config` or
`attention_config`; those get resolved to nested dicts at load time. Anything
else is returned as-is so the rest of the codebase can treat configs as plain
dicts — no surprises.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a top-level mapping, got {type(data).__name__}")
    return _resolve_includes(data, base=p.parent)


def _resolve_includes(cfg: dict[str, Any], base: Path) -> dict[str, Any]:
    for key in ("ffn_config", "attention_config"):
        if key in cfg and isinstance(cfg[key], str):
            ref_path = (base / cfg[key]).resolve() if not Path(cfg[key]).is_absolute() else Path(cfg[key])
            cfg[key] = load_yaml(ref_path)
    return cfg


def merge(*cfgs: dict[str, Any]) -> dict[str, Any]:
    """Deep merge dicts, later overrides earlier. Lists are replaced, not concatenated."""
    out: dict[str, Any] = {}
    for c in cfgs:
        _deep_merge(out, c)
    return out


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
