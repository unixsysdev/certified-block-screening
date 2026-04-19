"""Equivalence tests for the adaptive attention core.

Rules we enforce (violations are scientific bugs, not style):

1. **Debug-exact ≡ full attention.** When debug_exact=True the wrapper must
   return the same output as a standard causal softmax attention.

2. **top_k = num_blocks ⇒ full attention.** Selecting every block means every
   token is "allowed" and the mask collapses to the causal-only case.

3. **top_k = num_blocks + residual ⇒ full attention.** Since no blocks are
   unselected, residual summaries vanish and we again reduce to full attention.

All tests run on CPU with small tensors — no GPU, no model loading.
"""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.layer import (
    AdaptiveAttentionCfg,
    AdaptiveAttentionStats,
    _full_attention,
    adaptive_attention,
)


def _toy_qkv(B=1, H_q=2, H_kv=1, L=16, D=4, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H_q, L, D)
    k = torch.randn(B, H_kv, L, D)
    v = torch.randn(B, H_kv, L, D)
    return q, k, v


def test_debug_exact_matches_full():
    q, k, v = _toy_qkv()
    cfg = AdaptiveAttentionCfg(block_size=4, top_k=2, mode="coarse_replace", debug_exact=True)
    out_adaptive, _ = adaptive_attention(q, k, v, None, cfg=cfg, scaling=0.5)
    out_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(out_adaptive, out_full, atol=1e-5), (out_adaptive - out_full).abs().max()


def test_topk_equals_num_blocks_matches_full_coarse_replace():
    B, H_q, H_kv, L, D = 1, 2, 1, 16, 4
    q, k, v = _toy_qkv(B, H_q, H_kv, L, D)
    block_size = 4
    num_blocks = L // block_size  # 4
    cfg = AdaptiveAttentionCfg(block_size=block_size, top_k=num_blocks, mode="coarse_replace")
    out_adaptive, _ = adaptive_attention(q, k, v, None, cfg=cfg, scaling=0.5)
    out_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(out_adaptive, out_full, atol=1e-5), (out_adaptive - out_full).abs().max()


def test_topk_equals_num_blocks_matches_full_residual():
    B, H_q, H_kv, L, D = 1, 2, 1, 16, 4
    q, k, v = _toy_qkv(B, H_q, H_kv, L, D)
    block_size = 4
    num_blocks = L // block_size
    cfg = AdaptiveAttentionCfg(block_size=block_size, top_k=num_blocks, mode="residual_refine")
    out_adaptive, _ = adaptive_attention(q, k, v, None, cfg=cfg, scaling=0.5)
    out_full, _ = _full_attention(q, k, v, scaling=0.5)
    # Residual-refine with all blocks selected: no summaries contribute, must match full.
    assert torch.allclose(out_adaptive, out_full, atol=1e-5), (out_adaptive - out_full).abs().max()


def test_stats_observed():
    q, k, v = _toy_qkv()
    cfg = AdaptiveAttentionCfg(block_size=4, top_k=2, mode="coarse_replace")
    stats = AdaptiveAttentionStats()
    adaptive_attention(q, k, v, None, cfg=cfg, scaling=0.5, stats=stats)
    assert stats.n_forwards == 1
    assert stats.avg_num_blocks == 4   # L=16, block=4
    assert stats.avg_exact_blocks == 2


def test_low_topk_differs_from_full():
    """Sanity: if top_k < num_blocks, output SHOULD differ from full attention.
    (Otherwise the gating logic isn't actually doing anything.)"""
    q, k, v = _toy_qkv(L=32)
    cfg = AdaptiveAttentionCfg(block_size=4, top_k=2, mode="coarse_replace")
    out_adaptive, _ = adaptive_attention(q, k, v, None, cfg=cfg, scaling=0.5)
    out_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert not torch.allclose(out_adaptive, out_full, atol=1e-3)
