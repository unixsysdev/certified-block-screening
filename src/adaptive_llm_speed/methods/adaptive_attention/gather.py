"""Gather-based fast path for adaptive coarse-to-exact attention.

Whereas `layer.adaptive_attention` expresses block gating as a mask over the
full (Lq, Lk) attention matrix (correct but offers no compute reduction), this
module materialises only the K/V slices that correspond to selected blocks
and runs the softmax over the reduced tensor.

Key shapes (per-query selection, batched over heads and queries):

    q:        (B, H_q, Lq, D)
    k, v:     (B, H_kv, Lk, D)
    K_bar:    (B, H_kv, Nb, D)     — block_size=b, Nb = ceil(Lk / b)
    scores:   (B, H_q, Lq, Nb)
    sel_idx:  (B, H_q, Lq, k_eff)  — block indices per query, int64
    tok_idx:  (B, H_q, Lq, k_eff*b)
    K_sel:    (B, H_q, Lq, k_eff*b, D)
    V_sel:    (B, H_q, Lq, k_eff*b, D)
    logits:   (B, H_q, Lq, k_eff*b [+ Nb for residual])

The `.expand()` trick is load-bearing: we expand the KV along the Lq axis
with stride 0, then gather — the memory for K_sel is
`B * H_q * Lq * k_eff*b * D`, not the full `B * H_q * Lq * Lk_pad * D` that
a naive unsqueeze-and-materialise would produce.

Modes:
- coarse_replace:   SDPA on gathered tokens only; unselected blocks don't
                    contribute at all.
- residual_refine:  SDPA on gathered tokens concatenated with (k_bar, v_bar)
                    of every *unselected valid* block. This is the version
                    that held quality in the mask-based experiments.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .layer import (
    AdaptiveAttentionCfg,
    AdaptiveAttentionStats,
    _build_causal_block_mask,
    _repeat_kv,
)
from .selectors import FixedTopKSelector
from .summaries import block_mean_summary


def _score_and_select(q, K_bar_rep, block_valid, scaling, top_k):
    """Return (selection indices (B,H,Lq,k_eff), raw scores (B,H,Lq,Nb))."""
    scores = torch.einsum("bhqd,bhrd->bhqr", q, K_bar_rep) * scaling
    scores_masked = scores.masked_fill(~block_valid, float("-inf"))
    selector = FixedTopKSelector(top_k)
    sel_idx = selector(scores_masked, block_valid_mask=block_valid)
    return sel_idx, scores_masked


def _gather_selected(K_rep, V_rep, sel_idx, block_size, Lk_pad):
    """Materialise (B,H,Lq,k_eff*b,D) gathers of K and V using the expand-gather trick."""
    B, H, Lq, k_eff = sel_idx.shape
    D = K_rep.shape[-1]
    device = sel_idx.device
    offsets = torch.arange(block_size, device=device)                              # (b,)
    tok_idx = (sel_idx.unsqueeze(-1) * block_size + offsets).reshape(B, H, Lq, k_eff * block_size)
    # Clamp any index that might overshoot Lk_pad — shouldn't happen since sel_idx < Nb,
    # but padded slots can sit past Lk_real; gathered_causal will zero them.
    tok_idx = tok_idx.clamp(max=Lk_pad - 1)
    tok_idx_for_kv = tok_idx.unsqueeze(-1).expand(B, H, Lq, k_eff * block_size, D)
    K_sel = K_rep.unsqueeze(2).expand(B, H, Lq, Lk_pad, D).gather(3, tok_idx_for_kv)
    V_sel = V_rep.unsqueeze(2).expand(B, H, Lq, Lk_pad, D).gather(3, tok_idx_for_kv)
    return K_sel, V_sel, tok_idx


def adaptive_attention_gather(
    query: torch.Tensor,            # (B, H_q, Lq, D)
    key: torch.Tensor,              # (B, H_kv, Lk, D)
    value: torch.Tensor,            # (B, H_kv, Lk, D)
    attention_mask: torch.Tensor | None,
    *,
    cfg: AdaptiveAttentionCfg,
    scaling: float,
    q_start: int = 0,
    stats: AdaptiveAttentionStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cfg.debug_exact:
        # Debug path — full causal attention, bypasses block logic entirely.
        from .layer import _full_attention
        return _full_attention(query, key, value, scaling, q_start=q_start)

    B, H_q, Lq, D = query.shape
    _, H_kv, Lk, _ = key.shape
    n_rep = H_q // H_kv
    device = query.device

    K_bar, V_bar, pad_len = block_mean_summary(key, value, cfg.block_size)
    Nb = K_bar.shape[-2]
    Lk_pad = Nb * cfg.block_size

    # Pad K,V to Lk_pad so gather indices are always in-range.
    if pad_len > 0:
        k_full = torch.cat([key, key.new_zeros(B, H_kv, pad_len, D)], dim=-2)
        v_full = torch.cat([value, value.new_zeros(B, H_kv, pad_len, D)], dim=-2)
    else:
        k_full, v_full = key, value

    K_rep = _repeat_kv(k_full, n_rep)           # (B, H_q, Lk_pad, D)
    V_rep = _repeat_kv(v_full, n_rep)
    K_bar_rep = _repeat_kv(K_bar, n_rep)        # (B, H_q, Nb, D)
    V_bar_rep = _repeat_kv(V_bar, n_rep)

    block_causal, key_causal = _build_causal_block_mask(Lq, Lk, cfg.block_size, q_start, device)
    block_valid = block_causal.unsqueeze(0).unsqueeze(0).expand(B, H_q, Lq, Nb)
    key_causal_exp = key_causal.unsqueeze(0).unsqueeze(0).expand(B, H_q, Lq, Lk_pad)

    sel_idx, scores_masked = _score_and_select(query, K_bar_rep, block_valid, scaling, cfg.top_k)
    k_eff = sel_idx.shape[-1]

    K_sel, V_sel, tok_idx = _gather_selected(K_rep, V_rep, sel_idx, cfg.block_size, Lk_pad)

    # Per-gathered-token causal validity.
    gathered_causal = key_causal_exp.gather(3, tok_idx)                   # (B, H_q, Lq, k_eff*b)

    # Gathered-token logits.
    logits = torch.einsum("bhqd,bhqjd->bhqj", query, K_sel) * scaling     # (B, H_q, Lq, k_eff*b)
    logits = logits.masked_fill(~gathered_causal, float("-inf"))

    if cfg.mode == "residual_refine":
        # Summaries for valid UNSELECTED blocks get stitched onto the softmax.
        nb_range = torch.arange(Nb, device=device).view(1, 1, 1, 1, Nb)   # (1,1,1,1,Nb)
        selected = (sel_idx.unsqueeze(-1) == nb_range).any(dim=-2)        # (B, H_q, Lq, Nb)
        summary_logits = scores_masked.masked_fill(selected, float("-inf"))
        # V_bar lifted to (B, H_q, Lq, Nb, D). Uses expand (no materialization until softmax collapse below).
        V_bar_lifted = V_bar_rep.unsqueeze(2).expand(B, H_q, Lq, Nb, D)
        combined_logits = torch.cat([logits, summary_logits], dim=-1)          # (..., k_eff*b + Nb)
        combined_values = torch.cat([V_sel, V_bar_lifted], dim=-2)             # (..., k_eff*b + Nb, D)
        weights = F.softmax(combined_logits, dim=-1)
        out = torch.einsum("bhqj,bhqjd->bhqd", weights, combined_values)
    elif cfg.mode == "coarse_replace":
        all_masked = (~gathered_causal).all(dim=-1, keepdim=True)
        weights = F.softmax(logits, dim=-1)
        weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        out = torch.einsum("bhqj,bhqjd->bhqd", weights, V_sel)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    if stats is not None:
        stats.observe(lq=Lq, lk=Lk, num_blocks=Nb, used_blocks=k_eff)

    return out, None
