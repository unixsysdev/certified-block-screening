"""Tests for BoundCache — must produce identical outputs to the full-recompute path.

If the cache ever diverges from the reference, the bound ceases to be "certified" —
a stale radius could be smaller than the true max distance and blocks could be
wrongly dropped. These tests are mechanical but load-bearing.
"""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.bounds import (
    BoundCache,
    compute_block_centers_and_radii,
)


def _random_kv(B=1, H_kv=2, L=48, D=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, H_kv, L, D), torch.randn(B, H_kv, L, D)


def test_cache_hit_returns_same_tensors():
    k, v = _random_kv(L=48)
    cache = BoundCache()
    K1, V1, R1, p1 = cache.get_or_compute(k, v, block_size=8)
    K2, V2, R2, p2 = cache.get_or_compute(k, v, block_size=8)
    # Same object identity — cache hit means we reuse.
    assert K1 is K2 and V1 is V2 and R1 is R2 and p1 == p2


def test_cache_invalidates_on_block_size_change():
    k, v = _random_kv(L=64)
    cache = BoundCache()
    _K1, _V1, R1, _ = cache.get_or_compute(k, v, block_size=8)
    _K2, _V2, R2, _ = cache.get_or_compute(k, v, block_size=16)
    # Different block_size -> different Nb -> different shape (and identity).
    assert R1.shape != R2.shape


def test_cache_full_recompute_matches_direct():
    k, v = _random_kv(L=56)  # not a multiple of 8 -> pad_len = 4
    cache = BoundCache()
    K_c, V_c, R_c, pad_c = cache.get_or_compute(k, v, block_size=8)
    K_d, V_d, R_d, _, pad_d = compute_block_centers_and_radii(k, v, 8)
    assert torch.allclose(K_c, K_d, atol=1e-5)
    assert torch.allclose(V_c, V_d, atol=1e-5)
    assert torch.allclose(R_c, R_d, atol=1e-5)
    assert pad_c == pad_d


def test_cache_incremental_matches_full():
    """Grow the KV one token at a time, compare cache's K_bar / radii to a fresh
    compute every step. If the incremental update is wrong this catches it."""
    torch.manual_seed(42)
    B, H, D = 1, 2, 8
    block_size = 4
    # Pre-generate a long sequence to slice.
    full_k = torch.randn(B, H, 32, D)
    full_v = torch.randn(B, H, 32, D)

    cache = BoundCache()
    # Seed the cache at length 10.
    cache.get_or_compute(full_k[..., :10, :], full_v[..., :10, :], block_size=block_size)

    for L in range(11, 33):  # 11..32
        k, v = full_k[..., :L, :], full_v[..., :L, :]
        K_c, V_c, R_c, pad_c = cache.get_or_compute(k, v, block_size=block_size)
        K_ref, V_ref, R_ref, _, pad_ref = compute_block_centers_and_radii(k, v, block_size)
        assert torch.allclose(K_c, K_ref, atol=1e-4), f"L={L} K differs"
        assert torch.allclose(V_c, V_ref, atol=1e-4), f"L={L} V differs"
        assert torch.allclose(R_c, R_ref, atol=1e-4), f"L={L} R differs"
        assert pad_c == pad_ref


def test_cache_handles_length_jump_as_full_recompute():
    """If KV length jumps by more than 1, we must recompute from scratch, not increment."""
    torch.manual_seed(1)
    B, H, D = 1, 1, 4
    full_k = torch.randn(B, H, 24, D)
    full_v = torch.randn(B, H, 24, D)
    cache = BoundCache()
    cache.get_or_compute(full_k[..., :8, :], full_v[..., :8, :], block_size=4)
    # Now jump to length 24 — can't increment; must recompute fully.
    K_c, V_c, R_c, _ = cache.get_or_compute(full_k, full_v, block_size=4)
    K_ref, V_ref, R_ref, _, _ = compute_block_centers_and_radii(full_k, full_v, 4)
    assert torch.allclose(K_c, K_ref, atol=1e-5)
    assert torch.allclose(V_c, V_ref, atol=1e-5)
    assert torch.allclose(R_c, R_ref, atol=1e-5)


def test_radius_upper_bounds_actual_max_distance():
    """Faster fp32-sum radius compute must still satisfy the invariant."""
    torch.manual_seed(3)
    k = torch.randn(1, 2, 32, 6)
    v = torch.zeros_like(k)
    K_bar, _, radii, _, _ = compute_block_centers_and_radii(k, v, 4)
    B, H, Nb, D = K_bar.shape
    K_blocks = k.reshape(B, H, Nb, 4, D)
    diffs = K_blocks - K_bar.unsqueeze(-2)
    actual_max = diffs.to(torch.float32).norm(dim=-1).max(dim=-1).values
    # radii >= actual_max, up to fp noise.
    assert (radii.float() + 1e-3 >= actual_max).all(), (radii - actual_max).min().item()
