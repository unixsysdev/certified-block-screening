"""Method dispatch: map `method.name` in a config to the right patcher.

A patcher takes (model, cfg) and mutates the model in-place, returning any
method-level metadata useful for reporting (e.g. which layers were touched).
"""
from __future__ import annotations

from typing import Any, Callable

from ..methods.adaptive_attention.patch import patch_adaptive_attention
from ..methods.baseline import passthrough_patch
from ..methods.ffn_lowrank.patch import patch_ffn_lowrank


_PATCHERS: dict[str, Callable[..., dict[str, Any]]] = {
    "baseline": passthrough_patch,
    "ffn_lowrank": patch_ffn_lowrank,
    "adaptive_attention": patch_adaptive_attention,
}


def apply_method(model, cfg: dict) -> dict[str, Any]:
    """Dispatch to the correct patcher based on cfg['method']['name']."""
    if "method" not in cfg:
        return _PATCHERS["baseline"](model, cfg)
    name = cfg["method"]["name"]
    if name not in _PATCHERS:
        raise KeyError(f"unknown method: {name!r}. known: {list(_PATCHERS)}")
    return _PATCHERS[name](model, cfg)
