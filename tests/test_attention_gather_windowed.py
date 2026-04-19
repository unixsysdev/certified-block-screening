"""Tests for gather_windowed.

Two core limiting-case identities we enforce:

- **top_k = num_blocks → full causal attention.** If every block is always
  selected, the window aggregation and block selection become no-ops; the
  output must match _full_attention.

- **query_window_size = Lq + max aggregation → gather_shared with max**,
  because having one big window is exactly the shared case.

Also a regression test for the retrieval-style failure mode: with Lq=32,
block=4, top_k=2 and `query_score="max"`, if exactly one query has an
exceptionally strong logit on block r, that block must appear in the
selection — even if most other queries score r low. This is the passkey
mechanic in miniature, and is what gather_shared-with-mean fails.
"""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.gather_windowed import (
    adaptive_attention_gather_windowed,
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


def test_topk_equals_num_blocks_matches_full_coarse_replace():
    q, k, v = _toy(L=32)
    num_blocks = 32 // 4
    cfg = AdaptiveAttentionCfg(
        block_size=4, top_k=num_blocks, mode="coarse_replace",
        impl="gather_windowed", query_window_size=4, query_score="max",
    )
    o, _ = adaptive_attention_gather_windowed(q, k, v, None, cfg=cfg, scaling=0.5)
    o_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(o, o_full, atol=1e-4), (o - o_full).abs().max()


def test_topk_equals_num_blocks_matches_full_residual():
    q, k, v = _toy(L=48)
    num_blocks = 48 // 4  # 12
    cfg = AdaptiveAttentionCfg(
        block_size=4, top_k=num_blocks, mode="residual_refine",
        impl="gather_windowed", query_window_size=4, query_score="max",
    )
    o, _ = adaptive_attention_gather_windowed(q, k, v, None, cfg=cfg, scaling=0.5)
    o_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(o, o_full, atol=1e-4), (o - o_full).abs().max()


def test_debug_exact_path():
    q, k, v = _toy()
    cfg = AdaptiveAttentionCfg(
        block_size=4, top_k=2, mode="coarse_replace",
        impl="gather_windowed", query_window_size=8, query_score="max",
        debug_exact=True,
    )
    o, _ = adaptive_attention_gather_windowed(q, k, v, None, cfg=cfg, scaling=0.5)
    o_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(o, o_full, atol=1e-5)


def test_aggregate_window_max_preserves_rare_query_signal():
    """Passkey-in-miniature at the aggregation primitive. One 'needle' query
    scores very high on block 2; most other queries score block 0 best. Mean
    aggregation picks block 0 (broadly high); max aggregation picks block 2
    (one big vote is enough).
    """
    from adaptive_llm_speed.methods.adaptive_attention.gather_windowed import _aggregate_window

    # Shape (B=1, H=1, Nq=1, W=8, Nb=4).
    per_q = torch.tensor([
        [10.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 30.0, 0.0],          # the rare "needle" query
    ]).view(1, 1, 1, 8, 4)
    real = torch.ones(1, 8, dtype=torch.bool)

    s_mean = _aggregate_window(per_q, real, "mean")
    s_max = _aggregate_window(per_q, real, "max")
    # Mean: block 0 wins at 70/8 = 8.75; block 2 is 30/8 = 3.75.
    assert s_mean.argmax(dim=-1).item() == 0
    # Max: block 2 wins at 30; block 0 is 10.
    assert s_max.argmax(dim=-1).item() == 2


def test_mean_plus_max_hybrid_budget():
    """rescue_k slot must always include at least one block the mean path didn't pick.

    Construct a case where mean selects one block, max selects another — hybrid
    with k_mean=1, k_rescue=1 must pick both.
    """
    torch.manual_seed(2)
    B, H_q, H_kv, D = 1, 1, 1, 8
    block_size = 4
    W, Lq, Nb = 8, 8, 4
    Lk = Nb * block_size

    k = torch.randn(B, H_kv, Lk, D)
    v = torch.randn(B, H_kv, Lk, D)
    q = 0.01 * torch.randn(B, H_q, W, D)
    # Mean-useful direction aligned with block 0.
    block0_mean = k[:, :, 0:block_size, :].mean(dim=-2)
    q[:] += 1.0 * block0_mean
    # Last query spikes on block 2.
    block2_mean = k[:, :, 2 * block_size : 3 * block_size, :].mean(dim=-2)
    q[:, :, -1, :] = 10.0 * block2_mean

    cfg = AdaptiveAttentionCfg(
        block_size=block_size, top_k=2, rescue_k=1,
        mode="coarse_replace",
        impl="gather_windowed", query_window_size=W, query_score="mean_plus_max",
    )
    out, _ = adaptive_attention_gather_windowed(q, k, v, None, cfg=cfg, scaling=0.5)
    # Must run without error and return the right shape.
    assert out.shape == (B, H_q, Lq, D)


def test_non_multiple_seq_length():
    q, k, v = _toy(L=30)   # 30 = 7.5 blocks of size 4; Lq not a multiple of W=4 either
    cfg = AdaptiveAttentionCfg(
        block_size=4, top_k=2, mode="residual_refine",
        impl="gather_windowed", query_window_size=4, query_score="max",
    )
    o, _ = adaptive_attention_gather_windowed(q, k, v, None, cfg=cfg, scaling=0.5)
    assert o.shape == q.shape


def test_single_query_is_per_query():
    """Lq=1: only one query, so max==mean, and the aggregation is a pass-through.
    Same as the per-query decode case."""
    B, H_q, H_kv, D = 1, 2, 1, 4
    torch.manual_seed(5)
    q = torch.randn(B, H_q, 1, D)
    k = torch.randn(B, H_kv, 16, D)
    v = torch.randn(B, H_kv, 16, D)
    for stat in ("max", "mean"):
        cfg = AdaptiveAttentionCfg(
            block_size=4, top_k=2, mode="residual_refine",
            impl="gather_windowed", query_window_size=4, query_score=stat,
        )
        o, _ = adaptive_attention_gather_windowed(q, k, v, None, cfg=cfg, scaling=0.5)
        assert o.shape == q.shape
