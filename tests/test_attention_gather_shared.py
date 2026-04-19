"""Sanity tests for gather_shared.

Unlike `gather`, this impl is NOT bit-equivalent to `mask` — it shares block
selection across queries per head, so the softmax supports differ. These tests
verify that the sharing reduces to the correct limiting cases.
"""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.gather_shared import (
    adaptive_attention_gather_shared,
)
from adaptive_llm_speed.methods.adaptive_attention.layer import (
    AdaptiveAttentionCfg,
    _full_attention,
)


def _toy(B=1, H_q=2, H_kv=1, L=32, D=4, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H_q, L, D)
    k = torch.randn(B, H_kv, L, D)
    v = torch.randn(B, H_kv, L, D)
    return q, k, v


def test_topk_equals_all_blocks_matches_full():
    q, k, v = _toy(L=32)
    num_blocks = 32 // 4
    cfg = AdaptiveAttentionCfg(
        block_size=4, top_k=num_blocks, mode="coarse_replace", impl="gather_shared"
    )
    o_gs, _ = adaptive_attention_gather_shared(q, k, v, None, cfg=cfg, scaling=0.5)
    o_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(o_gs, o_full, atol=1e-4), (o_gs - o_full).abs().max()


def test_debug_exact_matches_full():
    q, k, v = _toy()
    cfg = AdaptiveAttentionCfg(
        block_size=4, top_k=2, mode="coarse_replace", impl="gather_shared", debug_exact=True
    )
    o_gs, _ = adaptive_attention_gather_shared(q, k, v, None, cfg=cfg, scaling=0.5)
    o_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(o_gs, o_full, atol=1e-5)


def test_single_query_is_per_query_case():
    """With Lq=1, shared selection ≡ per-query selection (only one query). Residual must
    hold quality in this limit."""
    q, k, v = _toy(L=32)
    # Take just one query.
    q1 = q[:, :, -1:, :]
    cfg_shared = AdaptiveAttentionCfg(
        block_size=4, top_k=3, mode="residual_refine", impl="gather_shared"
    )
    out_shared, _ = adaptive_attention_gather_shared(q1, k, v, None, cfg=cfg_shared, scaling=0.5)
    # For top_k=num_blocks=8 with L=32 and b=4, it must match full attention.
    cfg_full = AdaptiveAttentionCfg(
        block_size=4, top_k=8, mode="residual_refine", impl="gather_shared"
    )
    out_full_topk, _ = adaptive_attention_gather_shared(q1, k, v, None, cfg=cfg_full, scaling=0.5)
    # Now compare against analytic full attention for the single query.
    # (With topk=all, summaries get masked as -inf since no unselected blocks.)
    o_ref, _ = _full_attention(q1, k, v, scaling=0.5)
    assert torch.allclose(out_full_topk, o_ref, atol=1e-4), (out_full_topk - o_ref).abs().max()


def test_residual_shape_prefill():
    B, H_q, H_kv, L, D = 1, 4, 2, 48, 8
    q, k, v = _toy(B, H_q, H_kv, L, D)
    cfg = AdaptiveAttentionCfg(
        block_size=8, top_k=2, mode="residual_refine", impl="gather_shared"
    )
    out, _ = adaptive_attention_gather_shared(q, k, v, None, cfg=cfg, scaling=0.5)
    assert out.shape == (B, H_q, L, D)
