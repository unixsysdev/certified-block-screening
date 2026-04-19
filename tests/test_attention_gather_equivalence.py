"""Gather impl must match mask impl bit-for-float-tolerance at identical configs.

If these two diverge, we're either masking wrong or gathering wrong — either
way it's a correctness bug, not a numerical one.
"""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.gather import adaptive_attention_gather
from adaptive_llm_speed.methods.adaptive_attention.layer import (
    AdaptiveAttentionCfg,
    _full_attention,
    adaptive_attention,
)


def _toy(B=1, H_q=2, H_kv=1, L=32, D=4, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H_q, L, D)
    k = torch.randn(B, H_kv, L, D)
    v = torch.randn(B, H_kv, L, D)
    return q, k, v


def test_gather_matches_mask_coarse_replace():
    q, k, v = _toy()
    cfg = AdaptiveAttentionCfg(block_size=4, top_k=3, mode="coarse_replace", impl="gather")
    o_gather, _ = adaptive_attention_gather(q, k, v, None, cfg=cfg, scaling=0.5)
    cfg_mask = AdaptiveAttentionCfg(block_size=4, top_k=3, mode="coarse_replace", impl="mask")
    o_mask, _ = adaptive_attention(q, k, v, None, cfg=cfg_mask, scaling=0.5)
    assert torch.allclose(o_gather, o_mask, atol=1e-5), (o_gather - o_mask).abs().max()


def test_gather_matches_mask_residual_refine():
    q, k, v = _toy()
    cfg = AdaptiveAttentionCfg(block_size=4, top_k=3, mode="residual_refine", impl="gather")
    o_gather, _ = adaptive_attention_gather(q, k, v, None, cfg=cfg, scaling=0.5)
    cfg_mask = AdaptiveAttentionCfg(block_size=4, top_k=3, mode="residual_refine", impl="mask")
    o_mask, _ = adaptive_attention(q, k, v, None, cfg=cfg_mask, scaling=0.5)
    assert torch.allclose(o_gather, o_mask, atol=1e-5), (o_gather - o_mask).abs().max()


def test_gather_topk_all_matches_full():
    q, k, v = _toy(L=16)
    num_blocks = 16 // 4
    cfg = AdaptiveAttentionCfg(block_size=4, top_k=num_blocks, mode="coarse_replace", impl="gather")
    o_gather, _ = adaptive_attention_gather(q, k, v, None, cfg=cfg, scaling=0.5)
    o_full, _ = _full_attention(q, k, v, scaling=0.5)
    assert torch.allclose(o_gather, o_full, atol=1e-5), (o_gather - o_full).abs().max()


def test_gather_handles_nonmultiple_seq_length():
    # L = 30 with block_size 4 -> Nb = 8, pad_len = 2. Last block has 2 padded positions.
    q, k, v = _toy(L=30)
    cfg = AdaptiveAttentionCfg(block_size=4, top_k=2, mode="residual_refine", impl="gather")
    o_gather, _ = adaptive_attention_gather(q, k, v, None, cfg=cfg, scaling=0.5)
    cfg_mask = AdaptiveAttentionCfg(block_size=4, top_k=2, mode="residual_refine", impl="mask")
    o_mask, _ = adaptive_attention(q, k, v, None, cfg=cfg_mask, scaling=0.5)
    assert torch.allclose(o_gather, o_mask, atol=1e-5), (o_gather - o_mask).abs().max()


def test_gather_gqa_ratio():
    # GQA: more Q heads than KV heads. Ensures _repeat_kv path is consistent.
    q, k, v = _toy(H_q=4, H_kv=2, L=16)
    cfg = AdaptiveAttentionCfg(block_size=4, top_k=2, mode="coarse_replace", impl="gather")
    o_gather, _ = adaptive_attention_gather(q, k, v, None, cfg=cfg, scaling=0.5)
    cfg_mask = AdaptiveAttentionCfg(block_size=4, top_k=2, mode="coarse_replace", impl="mask")
    o_mask, _ = adaptive_attention(q, k, v, None, cfg=cfg_mask, scaling=0.5)
    assert torch.allclose(o_gather, o_mask, atol=1e-5), (o_gather - o_mask).abs().max()
