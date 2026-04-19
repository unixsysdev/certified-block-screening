"""Tests for the certified upper-bound selector (A2) and the math underneath.

The cornerstone test: the bound `U_r(q) = q·c_r + ||q||·ρ_r` must be ≥
`max_j q·k_j` for every block r, every query q, every random seed. If this
ever fails, the "certified" adjective is a lie.
"""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.bounds import (
    bound_tightness_stats,
    compute_block_centers_and_radii,
    upper_bound_logits,
)
from adaptive_llm_speed.methods.adaptive_attention.gather_shared import (
    adaptive_attention_gather_shared,
)
from adaptive_llm_speed.methods.adaptive_attention.layer import (
    AdaptiveAttentionCfg,
    _full_attention,
    _repeat_kv,
)
from adaptive_llm_speed.methods.adaptive_attention.selectors import (
    BoundScreenSelector,
    FixedTopKSelector,
)


def test_bound_never_underestimates_actual_max():
    """For random (q, K, V), U_r(q) ≥ max_j q·k_j for every (B, H, query, block)."""
    torch.manual_seed(0)
    B, H_kv, L, D = 2, 1, 64, 8
    H_q = H_kv
    block_size = 8
    q = torch.randn(B, H_q, 16, D)
    k = torch.randn(B, H_kv, L, D)
    v = torch.randn(B, H_kv, L, D)

    K_bar, V_bar, radii, K_padded, pad_len = compute_block_centers_and_radii(k, v, block_size)
    assert pad_len == 0
    K_bar_rep = _repeat_kv(K_bar, H_q // H_kv)
    radii_rep = _repeat_kv(radii.unsqueeze(-1), H_q // H_kv).squeeze(-1)

    upper, coarse, q_norm = upper_bound_logits(q, K_bar_rep, radii_rep, scaling=1.0)

    # Compute actual max per block manually.
    K_blocks = K_padded.reshape(B, H_kv, -1, block_size, D)  # (B, H_kv, Nb, b, D)
    K_blocks_rep = _repeat_kv(K_blocks.reshape(B, H_kv, -1, D), H_q // H_kv).reshape(
        B, H_q, K_blocks.shape[2], block_size, D
    )
    # logits[b,h,q,r,j] = q[b,h,q,:] · K_blocks_rep[b,h,r,j,:]
    logits = torch.einsum("bhqd,bhrjd->bhqrj", q, K_blocks_rep)
    actual_max = logits.max(dim=-1).values  # (B, H_q, Lq, Nb)

    slack = upper - actual_max
    assert (slack > -1e-3).all(), slack.min().item()


def test_bound_tight_for_rank_one_block():
    """When all keys in a block are identical, ρ_r = 0 and U_r = q·c_r exactly
    matches max_j q·k_j.
    """
    B, H, D = 1, 1, 4
    block_size = 8
    k_vec = torch.randn(D)
    # Fill a block with identical keys.
    k = k_vec.view(1, 1, 1, D).expand(B, H, block_size, D).contiguous()
    v = torch.randn(B, H, block_size, D)
    q = torch.randn(B, H, 3, D)
    K_bar, _, radii, _, _ = compute_block_centers_and_radii(k, v, block_size)
    # With all keys identical, radius is 0 (up to fp noise).
    assert radii.item() < 1e-3
    upper, coarse, _ = upper_bound_logits(q, K_bar, radii, scaling=1.0)
    actual = (q @ k_vec.view(D, 1)).squeeze(-1).unsqueeze(-1)  # (1, 1, 3, 1)
    assert torch.allclose(upper, actual, atol=1e-3)
    assert torch.allclose(coarse, actual, atol=1e-3)


def test_bound_screen_selector_shape():
    B, H, Lq, Nb = 1, 2, 4, 8
    coarse = torch.randn(B, H, Lq, Nb)
    radius_term = torch.rand(B, H, Lq, Nb)  # ≥ 0
    sel = BoundScreenSelector(top_k=3, delta=0.5)
    idx = sel(coarse, radius_term)
    assert idx.shape == (B, H, Lq, 3)
    # All indices within range.
    assert ((idx >= 0) & (idx < Nb)).all()


def test_bound_screen_delta_zero_matches_fixed_topk():
    B, H, Lq, Nb = 1, 2, 4, 8
    coarse = torch.randn(B, H, Lq, Nb)
    radius_term = torch.rand(B, H, Lq, Nb)
    a = BoundScreenSelector(top_k=3, delta=0.0)(coarse, radius_term)
    b = FixedTopKSelector(top_k=3)(coarse)
    # Same set of selected indices per (B, H, Lq). Order may differ, so compare sorted.
    assert torch.equal(a.sort(dim=-1).values, b.sort(dim=-1).values)


def test_gather_shared_bound_screen_runs_and_preserves_shape():
    torch.manual_seed(7)
    B, H_q, H_kv, L, D = 1, 4, 2, 64, 8
    q = torch.randn(B, H_q, L, D)
    k = torch.randn(B, H_kv, L, D)
    v = torch.randn(B, H_kv, L, D)
    cfg = AdaptiveAttentionCfg(
        block_size=8, top_k=4, mode="residual_refine",
        selector="bound_screen", delta=1.0, impl="gather_shared",
    )
    out, _ = adaptive_attention_gather_shared(q, k, v, None, cfg=cfg, scaling=0.25)
    assert out.shape == (B, H_q, L, D)


def test_gather_shared_bound_screen_topk_all_matches_full():
    torch.manual_seed(9)
    B, H_q, H_kv, L, D = 1, 2, 1, 32, 4
    num_blocks = 32 // 4
    q = torch.randn(B, H_q, L, D)
    k = torch.randn(B, H_kv, L, D)
    v = torch.randn(B, H_kv, L, D)
    cfg = AdaptiveAttentionCfg(
        block_size=4, top_k=num_blocks, mode="coarse_replace",
        selector="bound_screen", delta=1.0, impl="gather_shared",
    )
    o_gs, _ = adaptive_attention_gather_shared(q, k, v, None, cfg=cfg, scaling=0.5)
    o_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(o_gs, o_full, atol=1e-4), (o_gs - o_full).abs().max()


def test_tightness_stats_runs():
    B, H, Lq, Nb = 1, 1, 3, 4
    upper = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]] * Lq]])
    actual = torch.tensor([[[[0.5, 1.8, 2.9, 3.7]] * Lq]])
    stats = bound_tightness_stats(upper, actual)
    assert stats["count"] > 0
    assert stats["violation_rate"] == 0.0
    assert stats["median"] >= 1.0
