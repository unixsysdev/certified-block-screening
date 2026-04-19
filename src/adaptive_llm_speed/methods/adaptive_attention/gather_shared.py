"""Per-head shared-across-queries block gather + fused SDPA.

Why this exists: the per-query gather impl in `gather.py` is bit-equivalent to
the mask impl but wall-clock-disastrous on ROCm (~12× slower than mask at 2k
prefill). The culprit is `torch.gather` with scattered reads along a big Lq
axis — the kernel doesn't coalesce, and there's no way to fold the result into
a fused SDPA call because each query has a different key set.

This module takes a different trade-off: within a single layer forward, *all
queries* share one block selection per head (computed from the mean query).
That loses per-query adaptivity but shrinks the gather to `(B, H_q, k·b, D)` —
no Lq dimension — and reduces the attention to a plain SDPA call on a smaller
K/V, which the fused kernel handles efficiently.

Trade-off explicitly:
- PRO: real wall-clock reduction because the post-gather softmax is a fused
  SDPA on (Lq, k·b) instead of (Lq, Lk).
- CON: less adaptive. A head picks the same blocks for every query in the
  forward pass. For autoregressive decode (Lq=1) this is identical to
  per-query selection; for prefill it's an approximation.

Causal correctness is preserved: gathered tokens that sit past a given query
are masked out via an attn_mask passed to SDPA. So the selection is shared
per head, but the attention each query actually applies is still causal.

Modes:
- coarse_replace:  gather selected blocks' tokens, SDPA on those alone.
- residual_refine: also append (k_bar, v_bar) of every unselected valid block
                   as extra virtual tokens. Uses a single fused SDPA on the
                   concatenated tensor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .bounds import (
    BoundCache,
    bound_tightness_stats,
    compute_block_centers_and_radii,
    upper_bound_logits,
)
from .layer import (
    AdaptiveAttentionCfg,
    AdaptiveAttentionStats,
    _repeat_kv,
)
from .selectors import BoundScreenSelector, FixedTopKSelector
from .summaries import block_mean_summary


def adaptive_attention_gather_shared(
    query: torch.Tensor,           # (B, H_q, Lq, D)
    key: torch.Tensor,             # (B, H_kv, Lk, D)
    value: torch.Tensor,           # (B, H_kv, Lk, D)
    attention_mask: torch.Tensor | None,
    *,
    cfg: AdaptiveAttentionCfg,
    scaling: float,
    q_start: int = 0,
    stats: AdaptiveAttentionStats | None = None,
    bound_cache: "BoundCache | None" = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if cfg.debug_exact:
        from .layer import _full_attention
        return _full_attention(query, key, value, scaling, q_start=q_start)

    B, H_q, Lq, D = query.shape
    _, H_kv, Lk, _ = key.shape
    n_rep = H_q // H_kv
    b = cfg.block_size
    device = query.device
    dtype = query.dtype

    if cfg.selector == "bound_screen":
        if bound_cache is not None:
            K_bar, V_bar, radii, pad_len = bound_cache.get_or_compute(key, value, b)
        else:
            K_bar, V_bar, radii, _K_padded, pad_len = compute_block_centers_and_radii(key, value, b)
    else:
        K_bar, V_bar, pad_len = block_mean_summary(key, value, b)
        radii = None
    Nb = K_bar.shape[-2]
    Lk_pad = Nb * b

    if pad_len > 0:
        k_full = torch.cat([key, key.new_zeros(B, H_kv, pad_len, D)], dim=-2)
        v_full = torch.cat([value, value.new_zeros(B, H_kv, pad_len, D)], dim=-2)
    else:
        k_full, v_full = key, value

    K_rep = _repeat_kv(k_full, n_rep)         # (B, H_q, Lk_pad, D)
    V_rep = _repeat_kv(v_full, n_rep)
    K_bar_rep = _repeat_kv(K_bar, n_rep)      # (B, H_q, Nb, D)
    V_bar_rep = _repeat_kv(V_bar, n_rep)
    radii_rep = _repeat_kv(radii.unsqueeze(-1), n_rep).squeeze(-1) if radii is not None else None  # (B, H_q, Nb)

    # ── Block scoring: one score per (batch, head, block) using the mean query over Lq.
    q_mean = query.mean(dim=-2)                                    # (B, H_q, D)
    scores = torch.einsum("bhd,bhrd->bhr", q_mean, K_bar_rep) * scaling  # (B, H_q, Nb)

    # ── Causal block validity: a block can ever contribute to *this forward* iff at
    #    least one query in [q_start, q_start + Lq) is allowed to see some token in it.
    #    Block r covers positions [r*b, (r+1)*b); query i (absolute q_start+i) sees
    #    position j iff j ≤ q_start+i. For the SHARED selection we require the block
    #    to be seen by at least the LAST query (q_start + Lq - 1). In practice this
    #    means the block starts at ≤ q_start + Lq - 1 and its real (non-padded)
    #    portion is non-empty.
    block_first = torch.arange(Nb, device=device) * b              # (Nb,)
    last_q_abs = q_start + max(Lq - 1, 0)
    block_valid = (block_first <= last_q_abs) & (block_first < Lk)  # (Nb,) bool
    block_valid_bh = block_valid.unsqueeze(0).unsqueeze(0).expand(B, H_q, Nb)
    scores_masked = scores.masked_fill(~block_valid_bh, float("-inf"))

    if cfg.selector == "bound_screen":
        # Score by U_r(q_mean) = q_mean·c_r + δ·||q_mean||·ρ_r. Same shape as scores.
        q_mean_norm = q_mean.to(torch.float32).norm(dim=-1)                                # (B, H_q)
        radius_term = (q_mean_norm.unsqueeze(-1) * radii_rep.to(torch.float32))             # (B, H_q, Nb)
        radius_term = (radius_term * scaling).to(query.dtype)
        bounds = scores + cfg.delta * radius_term                                           # (B, H_q, Nb)
        bounds_masked = bounds.masked_fill(~block_valid_bh, float("-inf"))
        k_eff = min(cfg.top_k, Nb)
        _, sel_idx = torch.topk(bounds_masked, k=k_eff, dim=-1)
    else:  # fixed_topk
        k_eff = min(cfg.top_k, Nb)
        _, sel_idx = torch.topk(scores_masked, k=k_eff, dim=-1)        # (B, H_q, k)

    # ── Gather K, V for selected blocks. Result shape (B, H_q, k·b, D).
    offsets = torch.arange(b, device=device)
    tok_idx = (sel_idx.unsqueeze(-1) * b + offsets).reshape(B, H_q, k_eff * b)  # (B, H_q, k·b)
    tok_idx = tok_idx.clamp(max=Lk_pad - 1)
    tok_idx_kv = tok_idx.unsqueeze(-1).expand(B, H_q, k_eff * b, D)
    K_sel = K_rep.gather(2, tok_idx_kv)                             # (B, H_q, k·b, D)
    V_sel = V_rep.gather(2, tok_idx_kv)

    # ── Per-query causal mask over gathered tokens. mask[b,h,q,j] = 0 iff gathered
    #    token j is at an absolute position ≤ query q's absolute position AND within
    #    the real (non-padded) key range [0, Lk).
    q_abs = torch.arange(Lq, device=device) + q_start              # (Lq,)
    # tok_abs_pos: same as tok_idx (absolute positions of gathered tokens). Pad-valid mask below.
    tok_abs = tok_idx                                              # (B, H_q, k·b), already absolute
    # Broadcast: (B, H_q, Lq, k·b)
    causal_ok = tok_abs.unsqueeze(-2) <= q_abs.view(1, 1, Lq, 1)
    in_range = tok_abs.unsqueeze(-2) < Lk
    token_valid = causal_ok & in_range                              # (B, H_q, Lq, k·b)

    # Build SDPA-style additive mask (0 where allowed, -inf where not).
    # Keep it in the model dtype to avoid a dtype-mismatch path in SDPA.
    NEG_INF = torch.finfo(dtype).min
    attn_bias = torch.zeros(B, H_q, Lq, k_eff * b, device=device, dtype=dtype)
    attn_bias = attn_bias.masked_fill(~token_valid, NEG_INF)

    if cfg.mode == "residual_refine":
        # Append (k_bar, v_bar) of valid *unselected* blocks as virtual keys.
        nb_range = torch.arange(Nb, device=device).view(1, 1, 1, Nb)
        selected = (sel_idx.unsqueeze(-1) == nb_range).any(dim=-2)  # (B, H_q, Nb)
        # Which summaries are real: valid AND unselected.
        summary_real = block_valid_bh & ~selected                    # (B, H_q, Nb)
        # Summary-level per-query causality: a summary is usable by query q iff the
        # block's *first* real position ≤ q's absolute position. For block r that's
        # block_first[r] ≤ q_abs.
        summary_causal = block_first.view(1, 1, 1, Nb) <= q_abs.view(1, 1, Lq, 1)
        summary_allowed = summary_causal & summary_real.unsqueeze(-2)  # (B, H_q, Lq, Nb)

        # Pack: K,V then summaries on the key axis.
        K_summary = K_bar_rep                                         # (B, H_q, Nb, D)
        V_summary = V_bar_rep
        K_packed = torch.cat([K_sel, K_summary], dim=-2)              # (B, H_q, k·b + Nb, D)
        V_packed = torch.cat([V_sel, V_summary], dim=-2)

        summary_bias = torch.zeros(B, H_q, Lq, Nb, device=device, dtype=dtype)
        summary_bias = summary_bias.masked_fill(~summary_allowed, NEG_INF)
        attn_bias_packed = torch.cat([attn_bias, summary_bias], dim=-1)  # (B, H_q, Lq, k·b + Nb)

        out = F.scaled_dot_product_attention(
            query, K_packed, V_packed, attn_mask=attn_bias_packed, scale=scaling
        )
    elif cfg.mode == "coarse_replace":
        out = F.scaled_dot_product_attention(
            query, K_sel, V_sel, attn_mask=attn_bias, scale=scaling
        )
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    if stats is not None:
        stats.observe(lq=Lq, lk=Lk, num_blocks=Nb, used_blocks=k_eff)

    return out, None
