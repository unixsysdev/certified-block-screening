"""Install the adaptive attention forward on targeted Qwen3Attention instances.

Approach: keep the Qwen3Attention object (and all its learned weights) intact;
replace its bound `forward` method with a function that does the same prefix
(projections, RMSNorm on head dim, RoPE, KV-cache update) but swaps the final
attention_interface call for our `adaptive_attention` core.

This avoids subclassing or monkey-patching the class itself — the change is
scoped to specific instances, so non-targeted layers keep using SDPA untouched.
"""
from __future__ import annotations

import types
from typing import Any, Iterable

import torch

from .bounds import BoundCache
from .gather import adaptive_attention_gather
from .gather_shared import adaptive_attention_gather_shared
from .gather_windowed import adaptive_attention_gather_windowed
from .layer import AdaptiveAttentionCfg, AdaptiveAttentionStats, adaptive_attention
from ..ffn_lowrank.patch import _iter_decoder_layers, _resolve_layer_indices


_IMPL_DISPATCH = {
    "mask": adaptive_attention,
    "gather": adaptive_attention_gather,
    "gather_shared": adaptive_attention_gather_shared,
    "gather_windowed": adaptive_attention_gather_windowed,
}

_IMPLS_USING_BOUND_CACHE = {"gather_shared", "gather_windowed"}


def _adaptive_forward(self, hidden_states, position_embeddings, attention_mask,
                      past_key_values=None, **kwargs):
    """Drop-in replacement for Qwen3Attention.forward. See original for reference."""
    # Imported lazily to avoid loading transformers at module import time.
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    q_start = 0
    if past_key_values is not None:
        # past_key_values.update appends the new K/V and returns full K, V.
        # Before the update, the layer's cache length is the q_start offset for this forward's queries.
        try:
            q_start = int(past_key_values.get_seq_length(self.layer_idx))
        except Exception:
            q_start = 0
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    core = _IMPL_DISPATCH.get(self._adaptive_cfg.impl, adaptive_attention)
    core_kwargs = dict(
        cfg=self._adaptive_cfg,
        scaling=self.scaling,
        q_start=q_start,
        stats=self._adaptive_stats,
    )
    # gather_shared and gather_windowed share the bound_cache contract. Threading it in as a kwarg
    # keeps the other cores' signatures clean.
    if (self._adaptive_cfg.impl in _IMPLS_USING_BOUND_CACHE
            and self._adaptive_cfg.selector == "bound_screen"):
        core_kwargs["bound_cache"] = self._bound_cache
    attn_output, attn_weights = core(
        query_states, key_states, value_states, attention_mask, **core_kwargs,
    )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_adaptive_attention(model, cfg: dict) -> dict[str, Any]:
    """Install adaptive attention on layers specified by cfg.

    Config shape:
        method:
          name: adaptive_attention
        adaptive_attention:
          block_size: 64
          top_k: 4
          mode: coarse_replace
          selector: fixed_topk
          summary: mean
          target_layers: all
          debug_exact: false
    """
    params = cfg.get("adaptive_attention", {})
    ac = AdaptiveAttentionCfg.from_dict(params)

    layers = _iter_decoder_layers(model)
    n_layers = len(layers)
    target_idxs = _resolve_layer_indices(n_layers, params.get("target_layers", "all"))

    touched = []
    for li in target_idxs:
        block = layers[li]
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        # Stash cfg + stats on the instance.
        attn._adaptive_cfg = ac
        attn._adaptive_stats = AdaptiveAttentionStats()
        attn._bound_cache = BoundCache()
        # Swap forward. Keep a reference to original for rollback/debug.
        attn._original_forward = attn.forward
        attn.forward = types.MethodType(_adaptive_forward, attn)
        touched.append(li)

    # Transformers 5.x requires the overall model to commit to one _attn_implementation;
    # we do not change it, so non-targeted layers still use SDPA. The targeted layers
    # now ignore self.config._attn_implementation entirely via the bound method swap.
    return {
        "method": "adaptive_attention",
        "n_layers": n_layers,
        "targeted_layers": touched,
        "block_size": ac.block_size,
        "top_k": ac.top_k,
        "mode": ac.mode,
        "selector": ac.selector,
        "summary": ac.summary,
    }


def collect_adaptive_stats(model) -> list[dict[str, Any]]:
    """Aggregate per-layer adaptive stats for reporting."""
    out = []
    for li, block in enumerate(_iter_decoder_layers(model)):
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        stats: AdaptiveAttentionStats | None = getattr(attn, "_adaptive_stats", None)
        if stats is None:
            continue
        row = stats.to_dict()
        row["layer_idx"] = li
        out.append(row)
    return out


def unpatch_adaptive_attention(model) -> int:
    """Undo a prior patch; useful for tests and for running multiple configs in-process."""
    n = 0
    for block in _iter_decoder_layers(model):
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        orig = getattr(attn, "_original_forward", None)
        if orig is None:
            continue
        attn.forward = orig
        del attn._original_forward
        if hasattr(attn, "_adaptive_cfg"):
            del attn._adaptive_cfg
        if hasattr(attn, "_adaptive_stats"):
            del attn._adaptive_stats
        if hasattr(attn, "_bound_cache"):
            del attn._bound_cache
        n += 1
    return n
