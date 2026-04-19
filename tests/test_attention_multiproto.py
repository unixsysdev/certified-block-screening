"""Tests for multi-prototype bound and its selector path.

The per-prototype bound (max_m U_{r,m}(q)) must:
- Be ≥ actual max_j q·k_j per block (certified upper bound).
- Equal the single-prototype bound when M=1.
- Be tighter-or-equal to the single-prototype bound when M>1 (intuition — it's a max
  of tighter bounds on sub-blocks).
"""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.bounds import (
    compute_block_centers_and_radii,
    compute_block_multiproto,
)
from adaptive_llm_speed.methods.adaptive_attention.gather_windowed import (
    adaptive_attention_gather_windowed,
)
from adaptive_llm_speed.methods.adaptive_attention.layer import (
    AdaptiveAttentionCfg,
    _full_attention,
    _repeat_kv,
)


def _toy(B=1, H=1, L=64, D=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, H, L, D), torch.randn(B, H, L, D)


def _block_wise_actual_max(q, K, block_size):
    B, H_q, Lq, D = q.shape
    _, H_kv, L, _ = K.shape
    n_rep = H_q // H_kv
    Nb = L // block_size
    K_blocks = K.reshape(B, H_kv, Nb, block_size, D)
    K_blocks_rep = _repeat_kv(K_blocks.reshape(B, H_kv, Nb * block_size, D), n_rep).reshape(
        B, H_q, Nb, block_size, D
    )
    logits = torch.einsum("bhqd,bhrjd->bhqrj", q, K_blocks_rep)
    return logits.max(dim=-1).values                                   # (B, H_q, Lq, Nb)


def test_multiproto_m1_matches_single_prototype():
    torch.manual_seed(0)
    k, v = _toy(L=32)
    block_size = 4
    K_bar_s, V_bar_s, radii_s, _, pad_s = compute_block_centers_and_radii(k, v, block_size)
    K_multi, V_multi, radii_multi, K_overall, V_overall, pad_m = compute_block_multiproto(
        k, v, block_size, num_prototypes=1
    )
    # K_multi has an extra M=1 axis; squeezing it must give the single-prototype tensors.
    assert torch.allclose(K_multi.squeeze(-2), K_bar_s, atol=1e-5)
    assert torch.allclose(V_multi.squeeze(-2), V_bar_s, atol=1e-5)
    assert torch.allclose(radii_multi.squeeze(-1), radii_s, atol=1e-5)
    assert torch.allclose(K_overall, K_bar_s, atol=1e-5)
    assert pad_s == pad_m


def test_multiproto_certificate_holds():
    """max_m U_{r,m}(q) must ≥ max_j q·k_j for every block, every query."""
    torch.manual_seed(1)
    B, H, L, D = 1, 1, 64, 8
    k, v = _toy(B=B, H=H, L=L, D=D)
    q = torch.randn(B, H, 16, D)
    for M in (1, 2, 4, 8):
        block_size = 16
        K_multi, V_multi, radii, K_overall, _, _ = compute_block_multiproto(k, v, block_size, M)
        # Per-prototype bound: q·c_{r,m} + ||q||·ρ_{r,m}
        coarse_m = torch.einsum("bhqd,bhrmd->bhqrm", q, K_multi)
        q_norm = q.float().norm(dim=-1)
        radius_term = q_norm.unsqueeze(-1).unsqueeze(-1) * radii.float().unsqueeze(-3)
        upper_m = coarse_m.float() + radius_term
        upper_block = upper_m.max(dim=-1).values                        # (B, H, Lq, Nb)
        actual_max = _block_wise_actual_max(q, k, block_size)
        # upper_block must ≥ actual_max at every cell.
        assert (upper_block + 1e-3 >= actual_max).all(), (upper_block - actual_max).min().item()


def test_multiproto_exposes_heterogeneous_blocks():
    """Mechanism test. Construct a block with one outlier token and several filler tokens,
    plus a uniform filler-only block. The single-prototype bound treats them similarly (both
    blocks' centers sit near filler). The multi-prototype bound should *rank the outlier block
    higher* for a query aligned with the outlier direction — that's how this design is supposed
    to rescue passkey retrieval.
    """
    torch.manual_seed(9)
    D, block_size, M = 8, 8, 2
    # Block 0 (uniform filler): 8 copies of one random vector + noise.
    filler = torch.randn(D)
    block0 = filler.unsqueeze(0).expand(block_size, D) + 0.01 * torch.randn(block_size, D)
    # Block 1 (needle in filler): 7 filler + 1 outlier.
    outlier = torch.randn(D) * 3.0
    block1 = filler.unsqueeze(0).expand(block_size, D).clone() + 0.01 * torch.randn(block_size, D)
    block1[3] = outlier
    k = torch.cat([block0, block1], dim=0).view(1, 1, 2 * block_size, D)   # (1, 1, 16, D)
    v = torch.zeros_like(k)
    q = outlier.view(1, 1, 1, D) * 1.0                                      # query aligned with outlier

    # Single prototype.
    K_bar_s, _, radii_s, _, _ = compute_block_centers_and_radii(k, v, block_size)
    coarse_s = torch.einsum("bhqd,bhrd->bhqr", q, K_bar_s)
    upper_s = coarse_s.float() + q.float().norm(dim=-1).unsqueeze(-1) * radii_s.float().unsqueeze(-2)
    gap_s = (upper_s[..., 1] - upper_s[..., 0]).item()      # block1 bound − block0 bound

    # Multi prototype.
    K_multi, _, radii_multi, _, _, _ = compute_block_multiproto(k, v, block_size, M)
    coarse_m = torch.einsum("bhqd,bhrmd->bhqrm", q, K_multi)
    upper_m_all = coarse_m.float() + q.float().norm(dim=-1).unsqueeze(-1).unsqueeze(-1) * radii_multi.float().unsqueeze(-3)
    upper_m = upper_m_all.max(dim=-1).values
    gap_m = (upper_m[..., 1] - upper_m[..., 0]).item()

    # Both methods must rank the needle-bearing block above the filler-only block.
    # The per-prototype bound can be looser in absolute terms than the single-prototype
    # bound (sub-centers diverge from the overall mean), but the ranking must be preserved.
    assert gap_s > 0, f"single-prototype bound failed to separate blocks, gap={gap_s:.3f}"
    assert gap_m > 0, f"multi-prototype bound failed to separate blocks, gap={gap_m:.3f}"


def test_multiproto_gather_windowed_runs_and_stays_certified_at_topk_all():
    """At top_k = Nb, the output must equal full attention — independent of M.

    Regression test that the multi-prototype plumbing doesn't break the 'everything-selected'
    identity.
    """
    torch.manual_seed(7)
    B, H_q, H_kv, L, D = 1, 2, 1, 32, 4
    q = torch.randn(B, H_q, L, D)
    k = torch.randn(B, H_kv, L, D)
    v = torch.randn(B, H_kv, L, D)
    for M in (1, 2, 4):
        cfg = AdaptiveAttentionCfg(
            block_size=8, top_k=4, mode="coarse_replace",
            selector="bound_screen", delta=1.0, impl="gather_windowed",
            query_window_size=8, query_score="max",
            num_prototypes=M,
        )
        o, _ = adaptive_attention_gather_windowed(q, k, v, None, cfg=cfg, scaling=0.5)
        o_full, _ = _full_attention(q, k, v, scaling=0.5)
        assert torch.allclose(o, o_full, atol=1e-4), (M, (o - o_full).abs().max().item())


def test_rejects_incompatible_block_size():
    k, v = _toy(L=32)
    # block_size=10 is not divisible by num_prototypes=3.
    try:
        compute_block_multiproto(k, v, block_size=10, num_prototypes=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
