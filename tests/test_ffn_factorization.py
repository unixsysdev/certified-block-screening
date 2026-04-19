"""Unit tests for the SVD factorization primitives.

These run on CPU with tiny tensors — fast, no GPU needed, safe for pre-commit.
"""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.ffn_lowrank.factorize import (
    energy_retained,
    reconstruction_error,
    truncated_svd_linear,
)


def test_full_rank_reconstructs_exactly():
    torch.manual_seed(0)
    W = torch.randn(32, 64)
    U, V = truncated_svd_linear(W, rank=32)  # rank = min(out, in)
    err = reconstruction_error(W, U, V)
    assert err["relative_frobenius"] < 1e-5, err


def test_rank_one_approximation_captures_dominant_direction():
    # If W = outer(u, v) * s, then rank-1 SVD should recover it perfectly up to sign.
    torch.manual_seed(1)
    u = torch.randn(8)
    v = torch.randn(16)
    W = torch.outer(u, v)
    U, V = truncated_svd_linear(W, rank=1)
    W_hat = U @ V
    assert torch.allclose(W, W_hat, atol=1e-4), (W - W_hat).abs().max()


def test_energy_retained_is_monotone_in_rank():
    torch.manual_seed(2)
    W = torch.randn(64, 128)
    e_prev = -1.0
    for r in (4, 8, 16, 32, 64):
        e = energy_retained(W, r)
        assert 0.0 <= e <= 1.0 + 1e-6
        assert e + 1e-9 >= e_prev, f"energy not monotone: {r=} {e_prev=} {e=}"
        e_prev = e


def test_rejects_invalid_rank():
    W = torch.randn(8, 16)
    try:
        truncated_svd_linear(W, rank=9)  # > min(out,in)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for rank > min(out,in)")

    try:
        truncated_svd_linear(W, rank=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for rank=0")
