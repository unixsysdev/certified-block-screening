"""Window-shared gather with per-query aggregation.

Diagnosis that motivated this file: `gather_shared.py` shares ONE block
selection across every query in a forward pass, scored by the mean query.
For sparse retrieval (passkey/needle), the one late query that wants the
needle block is drowned by the filler queries; the needle block almost
never makes it into the shared top-k. Every adaptive variant scored 0–10 %
on passkey while the dense baseline scored 100 %.

This file fixes the two design choices responsible for that:

1. **Windowed selection.** Queries are split into windows of size W; each
   window gets its own top-k selection. W=1 ≡ per-query (exact but slow);
   W=Lq ≡ the old per-forward shared (fast but bad for retrieval). Default
   W = block_size keeps the gather dense enough that SDPA fuses, while
   still localizing selection to a small query neighborhood.

2. **Max-over-queries aggregation inside the window.** A rare retrieval
   query only needs to win once; max preserves its vote, mean washes it
   out. We also support "mean" for backward comparison and "mean_plus_max"
   which splits the budget between broadly-useful blocks (mean) and
   rescue blocks (max).

Causal semantics: block r is valid for any query in window w iff block_first[r]
≤ last_real_q_pos_in_window(w). Per-gathered-token causal masking applies at
the SDPA call so individual queries in the window never attend to keys in
their own future.

Shape plan:
    query      (B, H_q, Lq, D)
    key, value (B, H_kv, Lk, D)
    K_bar, V_bar (B, H_kv, Nb, D)    — Nb = ceil(Lk / block_size)
    per-query logits  (B, H_q, Lq_pad, Nb)
    reshaped          (B, H_q, Nq, W, Nb)     — Nq = Lq_pad / W
    window scores     (B, H_q, Nq, Nb)
    sel_idx           (B, H_q, Nq, k)
    K_sel, V_sel      (B, H_q, Nq, k·b, D)    — one set of exact tokens per window
    SDPA is called once per window on the shape (B·Nq, H_q, W, k·b [+ Nb]).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .bounds import BoundCache, compute_block_centers_and_radii, compute_block_multiproto
from .layer import (
    AdaptiveAttentionCfg,
    AdaptiveAttentionStats,
    _repeat_kv,
)
from .summaries import block_mean_summary


_NEG_INF_CONST = float("-inf")


def _pad_queries_to_window(query: torch.Tensor, W: int) -> tuple[torch.Tensor, int]:
    B, H, Lq, D = query.shape
    rem = Lq % W
    if rem == 0:
        return query, 0
    pad_len = W - rem
    q_pad = query.new_zeros(B, H, pad_len, D)
    return torch.cat([query, q_pad], dim=-2), pad_len


def _aggregate_window(per_query: torch.Tensor,
                      real_q_mask: torch.Tensor,
                      statistic: str) -> torch.Tensor:
    """Collapse the W-axis of (..., Nq, W, Nb) into (..., Nq, Nb).

    Padded (non-real) queries are ignored: for max, their value becomes -inf; for
    mean, we divide by the count of real queries in the window (clamped to 1 to
    avoid 0/0 on a window where Lq doesn't cover any real queries, which
    shouldn't occur since we Lq-pad from the right).
    """
    real = real_q_mask.view(1, 1, *real_q_mask.shape, 1)  # broadcastable to (..., Nq, W, 1)
    if statistic == "max":
        masked = per_query.masked_fill(~real, _NEG_INF_CONST)
        return masked.max(dim=-2).values
    if statistic == "mean":
        masked = per_query.masked_fill(~real, 0.0)
        counts = real.sum(dim=-2).clamp(min=1)                                   # (..., Nq, 1)
        return masked.sum(dim=-2) / counts
    raise ValueError(f"unknown query_score statistic: {statistic!r}")


def adaptive_attention_gather_windowed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
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
    W = cfg.query_window_size or b   # 0 ⇒ auto (= block_size)
    device = query.device
    dtype = query.dtype

    # --- Block centers (+ per-prototype centers/radii for bound_screen with M > 1).
    M = max(1, cfg.num_prototypes)
    K_multi = V_multi = radii_multi = None   # only populated when M > 1
    if cfg.selector == "bound_screen":
        if M > 1:
            # Multi-prototype path. K_bar / V_bar become the overall block means for residual use.
            if bound_cache is not None:
                K_multi, V_multi, radii_multi, K_bar, V_bar, pad_len = bound_cache.get_or_compute_multi(
                    key, value, b, M
                )
            else:
                K_multi, V_multi, radii_multi, K_bar, V_bar, pad_len = compute_block_multiproto(
                    key, value, b, M
                )
            radii = None    # the block-overall radius is irrelevant; we use per-prototype radii
        else:
            if bound_cache is not None:
                K_bar, V_bar, radii, pad_len = bound_cache.get_or_compute(key, value, b)
            else:
                K_bar, V_bar, radii, _K_padded, pad_len = compute_block_centers_and_radii(key, value, b)
    else:
        # fixed_topk. No radii needed even with multi-prototype; just per-prototype centers.
        if M > 1:
            K_multi, V_multi, _radii_multi, K_bar, V_bar, pad_len = compute_block_multiproto(
                key, value, b, M
            )
            radii = None
        else:
            K_bar, V_bar, pad_len = block_mean_summary(key, value, b)
            radii = None
    Nb = K_bar.shape[-2]
    Lk_pad = Nb * b

    # Pad K, V right so gather indices are always in-range.
    if pad_len > 0:
        k_full = torch.cat([key, key.new_zeros(B, H_kv, pad_len, D)], dim=-2)
        v_full = torch.cat([value, value.new_zeros(B, H_kv, pad_len, D)], dim=-2)
    else:
        k_full, v_full = key, value

    K_rep = _repeat_kv(k_full, n_rep)
    V_rep = _repeat_kv(v_full, n_rep)
    K_bar_rep = _repeat_kv(K_bar, n_rep)                    # (B, H_q, Nb, D)
    V_bar_rep = _repeat_kv(V_bar, n_rep)
    radii_rep = _repeat_kv(radii.unsqueeze(-1), n_rep).squeeze(-1) if radii is not None else None

    if M > 1:
        # Lift per-prototype tensors from H_kv into H_q. (B, H_kv, Nb, M, D) -> (B, H_q, Nb, M, D).
        B0, H0, Nb0, M0, D0 = K_multi.shape
        K_multi_rep = K_multi.unsqueeze(2).expand(B0, H0, n_rep, Nb0, M0, D0).reshape(B0, H0 * n_rep, Nb0, M0, D0)
        if radii_multi is not None:
            B0r, H0r, Nb0r, M0r = radii_multi.shape
            radii_multi_rep = (
                radii_multi.unsqueeze(2).expand(B0r, H0r, n_rep, Nb0r, M0r)
                .reshape(B0r, H0r * n_rep, Nb0r, M0r)
            )
        else:
            radii_multi_rep = None
    else:
        K_multi_rep = None
        radii_multi_rep = None

    # --- Pad queries to a multiple of W.
    q_padded, q_pad_len = _pad_queries_to_window(query, W)
    Lq_pad = Lq + q_pad_len
    Nq = Lq_pad // W

    # --- Per-query block logits (coarse) and optional per-query bound term.
    if M > 1:
        # Per-prototype coarse logits: (B, H_q, Lq_pad, Nb, M).
        coarse_per_qm = torch.einsum("bhqd,bhrmd->bhqrm", q_padded, K_multi_rep) * scaling
        if cfg.selector == "bound_screen":
            q_norm = q_padded.to(torch.float32).norm(dim=-1)                           # (B, H_q, Lq_pad)
            # radius_term shape (B, H_q, Lq_pad, Nb, M) = q_norm[...,None,None] * radii[...,None,:,:] * scaling
            radius_term = (
                q_norm.unsqueeze(-1).unsqueeze(-1)
                * radii_multi_rep.to(torch.float32).unsqueeze(-3)
                * scaling
            ).to(dtype)
            per_qm_bound = coarse_per_qm + cfg.delta * radius_term                     # (B, H_q, Lq_pad, Nb, M)
        else:
            per_qm_bound = coarse_per_qm
        # Block-level certified bound = max over prototypes.
        per_q_bound = per_qm_bound.max(dim=-1).values                                  # (B, H_q, Lq_pad, Nb)
        coarse_per_q = coarse_per_qm.max(dim=-1).values                                # for residual summary stats
    else:
        coarse_per_q = torch.einsum("bhqd,bhrd->bhqr", q_padded, K_bar_rep) * scaling  # (B, H_q, Lq_pad, Nb)
        if cfg.selector == "bound_screen":
            q_norm = q_padded.to(torch.float32).norm(dim=-1)
            radius_term = (q_norm.unsqueeze(-1) * radii_rep.to(torch.float32).unsqueeze(-2) * scaling).to(dtype)
            per_q_bound = coarse_per_q + cfg.delta * radius_term                        # (B, H_q, Lq_pad, Nb)
        else:
            per_q_bound = coarse_per_q

    # --- Reshape to (B, H_q, Nq, W, Nb) and aggregate across the W axis.
    per_q_bound_w = per_q_bound.view(B, H_q, Nq, W, Nb)
    per_q_coarse_w = coarse_per_q.view(B, H_q, Nq, W, Nb)

    # Real-vs-padded query mask (Nq, W) — True for real queries.
    real_mask_flat = torch.arange(Lq_pad, device=device) < Lq                      # (Lq_pad,)
    real_mask = real_mask_flat.view(Nq, W)                                         # (Nq, W)

    # --- Causal block validity at the window level (block can contribute to any query in the window).
    block_first = torch.arange(Nb, device=device) * b                              # (Nb,)
    # Last absolute real-query position in each window.
    last_q_in_window = (
        torch.arange(Nq, device=device).view(Nq, 1) * W
        + torch.arange(W, device=device).view(1, W)
    ) + q_start                                                                     # (Nq, W)
    last_q_in_window = last_q_in_window.masked_fill(~real_mask, -1)                # (Nq, W)
    last_q_in_window = last_q_in_window.max(dim=-1).values                         # (Nq,)
    block_valid = (block_first.view(1, Nb) <= last_q_in_window.view(Nq, 1)) & (block_first < Lk)  # (Nq, Nb)
    block_valid_bh = block_valid.view(1, 1, Nq, Nb).expand(B, H_q, Nq, Nb)

    # --- Build the window-level score used by the selector.
    if cfg.query_score == "mean_plus_max":
        # Two aggregations: mean fills k_mean budget, max fills rescue_k budget.
        k_rescue = cfg.rescue_k
        k_mean = max(0, min(cfg.top_k - k_rescue, Nb))
        if k_rescue <= 0 or k_mean <= 0:
            raise ValueError(f"mean_plus_max needs rescue_k>0 and top_k>rescue_k; got {cfg.top_k=}, {cfg.rescue_k=}")
        mean_scores = _aggregate_window(per_q_bound_w, real_mask, "mean")
        max_scores = _aggregate_window(per_q_bound_w, real_mask, "max")
        mean_scores = mean_scores.masked_fill(~block_valid_bh, _NEG_INF_CONST)
        max_scores_for_rescue = max_scores.masked_fill(~block_valid_bh, _NEG_INF_CONST)

        _, mean_idx = torch.topk(mean_scores, k=k_mean, dim=-1)                    # (B, H_q, Nq, k_mean)

        # Rescue lane: pick top-k_rescue blocks by max that are NOT already in the mean selection.
        # Mask mean-picked blocks as -inf before rescue top-k.
        nb_range = torch.arange(Nb, device=device).view(1, 1, 1, Nb)
        in_mean = (mean_idx.unsqueeze(-1) == nb_range).any(dim=-2)                 # (B, H_q, Nq, Nb)
        max_for_rescue = max_scores_for_rescue.masked_fill(in_mean, _NEG_INF_CONST)
        _, rescue_idx = torch.topk(max_for_rescue, k=min(k_rescue, Nb), dim=-1)

        sel_idx = torch.cat([mean_idx, rescue_idx], dim=-1)                        # (B, H_q, Nq, k_mean + k_rescue)
    else:
        window_scores = _aggregate_window(per_q_bound_w, real_mask, cfg.query_score)
        window_scores = window_scores.masked_fill(~block_valid_bh, _NEG_INF_CONST)
        k_eff = min(cfg.top_k, Nb)
        _, sel_idx = torch.topk(window_scores, k=k_eff, dim=-1)                    # (B, H_q, Nq, k_eff)

    k_eff = sel_idx.shape[-1]

    # --- Gather K, V per window via expand+gather (no Lk-size materialisation).
    offsets = torch.arange(b, device=device)
    tok_idx = (sel_idx.unsqueeze(-1) * b + offsets).reshape(B, H_q, Nq, k_eff * b)  # (B, H_q, Nq, k·b)
    tok_idx = tok_idx.clamp(max=Lk_pad - 1)
    tok_idx_for_kv = tok_idx.unsqueeze(-1).expand(B, H_q, Nq, k_eff * b, D)
    K_sel = K_rep.unsqueeze(2).expand(B, H_q, Nq, Lk_pad, D).gather(3, tok_idx_for_kv)
    V_sel = V_rep.unsqueeze(2).expand(B, H_q, Nq, Lk_pad, D).gather(3, tok_idx_for_kv)

    # --- Per-query causal mask over gathered tokens (B, H_q, Nq, W, k·b).
    q_abs_w = (
        torch.arange(Nq, device=device).view(Nq, 1) * W
        + torch.arange(W, device=device).view(1, W)
    ) + q_start                                                                     # (Nq, W)
    # mark padded queries as -∞ — they won't contribute to SDPA output anyway since we slice Lq out.
    tok_idx_expanded = tok_idx.unsqueeze(-2)                                       # (B, H_q, Nq, 1, k·b)
    q_abs_expanded = q_abs_w.view(1, 1, Nq, W, 1)
    causal_ok = tok_idx_expanded <= q_abs_expanded
    in_range = tok_idx_expanded < Lk
    token_valid = causal_ok & in_range
    # Real-query gate on the W axis — SDPA will produce zero rows for padded queries anyway,
    # but this avoids any NaN if SDPA softmaxes on all-neg-inf rows.
    real_expanded = real_mask.view(1, 1, Nq, W, 1)
    token_valid = token_valid & real_expanded

    NEG_INF = torch.finfo(dtype).min
    attn_bias = torch.zeros(B, H_q, Nq, W, k_eff * b, device=device, dtype=dtype)
    attn_bias = attn_bias.masked_fill(~token_valid, NEG_INF)

    # --- Residual-refine: append valid-unselected block summaries.
    if cfg.mode == "residual_refine":
        nb_range = torch.arange(Nb, device=device).view(1, 1, 1, Nb)
        selected = (sel_idx.unsqueeze(-1) == nb_range).any(dim=-2)                  # (B, H_q, Nq, Nb)
        summary_real = block_valid_bh & ~selected                                    # (B, H_q, Nq, Nb)
        summary_causal = block_first.view(1, 1, 1, 1, Nb) <= q_abs_w.view(1, 1, Nq, W, 1)
        summary_allowed = summary_causal & summary_real.unsqueeze(-2) & real_expanded

        K_summary = K_bar_rep.unsqueeze(2).expand(B, H_q, Nq, Nb, D)
        V_summary = V_bar_rep.unsqueeze(2).expand(B, H_q, Nq, Nb, D)
        K_packed = torch.cat([K_sel, K_summary], dim=-2)                            # (B, H_q, Nq, k·b + Nb, D)
        V_packed = torch.cat([V_sel, V_summary], dim=-2)

        summary_bias = torch.zeros(B, H_q, Nq, W, Nb, device=device, dtype=dtype)
        summary_bias = summary_bias.masked_fill(~summary_allowed, NEG_INF)
        attn_bias_full = torch.cat([attn_bias, summary_bias], dim=-1)               # (..., k·b + Nb)
        K_final, V_final, bias_final = K_packed, V_packed, attn_bias_full
    elif cfg.mode == "coarse_replace":
        K_final, V_final, bias_final = K_sel, V_sel, attn_bias
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    # --- Flatten the Nq dim into batch for a single fused SDPA call.
    #     shapes: q (B*Nq, H_q, W, D), K/V (B*Nq, H_q, KV_len, D), bias (B*Nq, H_q, W, KV_len).
    def _nq_to_batch(x, *extra_dims):
        # x shape: (B, H_q, Nq, *extra_dims). Move Nq to just after batch.
        perm = (0, 2, 1) + tuple(range(3, 3 + len(extra_dims)))
        return x.permute(*perm).reshape(B * Nq, H_q, *extra_dims)

    q_bn = _nq_to_batch(q_padded.view(B, H_q, Nq, W, D), W, D)
    k_bn = _nq_to_batch(K_final, K_final.shape[-2], D)
    v_bn = _nq_to_batch(V_final, V_final.shape[-2], D)
    bias_bn = _nq_to_batch(bias_final, W, bias_final.shape[-1])

    out_bn = F.scaled_dot_product_attention(q_bn, k_bn, v_bn, attn_mask=bias_bn, scale=scaling)
    # (B*Nq, H_q, W, D) -> (B, Nq, H_q, W, D) -> (B, H_q, Nq*W, D).
    out_padded = out_bn.reshape(B, Nq, H_q, W, D).permute(0, 2, 1, 3, 4).reshape(B, H_q, Nq * W, D)
    out = out_padded[:, :, :Lq, :]

    if stats is not None:
        stats.observe(lq=Lq, lk=Lk, num_blocks=Nb, used_blocks=k_eff)

    return out, None
