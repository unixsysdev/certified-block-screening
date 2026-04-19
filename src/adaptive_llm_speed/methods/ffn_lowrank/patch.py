"""Walk a causal LM and replace per-layer FFN projections with LowRankLinear.

Qwen/LLaMA-family FFN (SwiGLU) has three linears: `gate_proj`, `up_proj`, `down_proj`.
Each is individually toggleable via the config so we can study:
- compress all three (maximal savings, ~50% params at rank 384 for Qwen3-0.6B)
- skip `gate_proj` (cheaper integration, but gate stays full-rank and may dominate compute)
- skip `down_proj` / `up_proj` individually (ablation)

Layer targeting:
- "all"         → every transformer block
- "upper_half"  → top half (deeper layers compress better empirically)
- "lower_half"  → bottom half
- "every_other" → odd-indexed blocks
- list[int]     → explicit indices
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn

from .factorize import reconstruction_error
from .layers import LowRankLinear


FFN_PROJECTION_NAMES = ("gate_proj", "up_proj", "down_proj")


def _iter_decoder_layers(model) -> list[nn.Module]:
    """Best-effort locator for the list of transformer blocks.

    Works for HF Qwen3 (model.model.layers) and the common decoder layout generally.
    """
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for c in candidates:
        if c is not None:
            return list(c)
    raise RuntimeError("could not locate decoder layers on this model; add a new entry to _iter_decoder_layers")


def _resolve_layer_indices(n_layers: int, spec: Any) -> list[int]:
    if spec is None or spec == "all":
        return list(range(n_layers))
    if spec == "upper_half":
        return list(range(n_layers // 2, n_layers))
    if spec == "lower_half":
        return list(range(0, n_layers // 2))
    if spec == "every_other":
        return list(range(1, n_layers, 2))
    if isinstance(spec, Iterable):
        idxs = sorted({int(i) for i in spec})
        for i in idxs:
            if i < 0 or i >= n_layers:
                raise ValueError(f"layer index {i} out of range [0, {n_layers})")
        return idxs
    raise ValueError(f"unknown target_layers spec: {spec!r}")


def _get_ffn_module(block: nn.Module) -> nn.Module | None:
    """Return the sub-module that holds gate/up/down projections, if any.

    Qwen3 puts them at `block.mlp`; some models call it `block.ffn` or `feed_forward`.
    """
    for attr in ("mlp", "feed_forward", "ffn"):
        m = getattr(block, attr, None)
        if m is not None and any(hasattr(m, n) for n in FFN_PROJECTION_NAMES):
            return m
    return None


def patch_ffn_lowrank(model, cfg: dict) -> dict[str, Any]:
    """In-place replacement of FFN linears with LowRankLinear factored via SVD.

    Config shape:
        method:
          name: ffn_lowrank
        ffn_lowrank:
          rank: 384            # int, or dict with 'gate', 'up', 'down' overrides
          init: svd            # only option for now
          patch_gate_proj: true
          patch_up_proj: true
          patch_down_proj: true
          target_layers: all   # see _resolve_layer_indices
          keep_dense_for_debug: false
    """
    params = cfg.get("ffn_lowrank", {})
    if params.get("init", "svd") != "svd":
        raise NotImplementedError(f"init={params.get('init')!r} not implemented yet")
    rank_spec = params["rank"]
    rank_per: dict[str, int] = {}
    if isinstance(rank_spec, int):
        rank_per = {n: rank_spec for n in FFN_PROJECTION_NAMES}
    elif isinstance(rank_spec, dict):
        rank_per = {n: int(rank_spec.get(n.split("_")[0], rank_spec.get(n, 0))) for n in FFN_PROJECTION_NAMES}
    else:
        raise ValueError(f"ffn_lowrank.rank must be int or dict, got {type(rank_spec).__name__}")

    enabled = {
        "gate_proj": bool(params.get("patch_gate_proj", True)),
        "up_proj": bool(params.get("patch_up_proj", True)),
        "down_proj": bool(params.get("patch_down_proj", True)),
    }
    keep_dense = bool(params.get("keep_dense_for_debug", False))

    layers = _iter_decoder_layers(model)
    n_layers = len(layers)
    target_idxs = _resolve_layer_indices(n_layers, params.get("target_layers", "all"))

    patched: list[dict[str, Any]] = []
    n_params_before = sum(p.numel() for p in model.parameters())
    max_rel_err = 0.0

    for li in target_idxs:
        block = layers[li]
        ffn = _get_ffn_module(block)
        if ffn is None:
            continue
        for pname in FFN_PROJECTION_NAMES:
            if not enabled[pname]:
                continue
            linear = getattr(ffn, pname, None)
            if not isinstance(linear, nn.Linear):
                continue
            r = min(rank_per[pname], min(linear.in_features, linear.out_features))
            if r <= 0:
                continue
            new = LowRankLinear.from_linear(linear, r, keep_dense_for_debug=keep_dense)
            setattr(ffn, pname, new)
            err = reconstruction_error(linear.weight.data, new.up.weight.data, new.down.weight.data)
            max_rel_err = max(max_rel_err, err["relative_frobenius"])
            patched.append({
                "layer_idx": li,
                "proj": pname,
                "rank": r,
                "in_features": linear.in_features,
                "out_features": linear.out_features,
                "relative_frobenius_error": err["relative_frobenius"],
            })

    n_params_after = sum(p.numel() for p in model.parameters())

    return {
        "method": "ffn_lowrank",
        "n_layers": n_layers,
        "targeted_layers": target_idxs,
        "projections_enabled": enabled,
        "rank_per_projection": rank_per,
        "patched_count": len(patched),
        "max_relative_frobenius_error": max_rel_err,
        "param_count_before": n_params_before,
        "param_count_after": n_params_after,
        "param_ratio": (n_params_after / n_params_before) if n_params_before else math.nan,
        "details": patched,
    }
