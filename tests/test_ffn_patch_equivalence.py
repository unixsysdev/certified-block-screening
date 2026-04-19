"""Equivalence tests for LowRankLinear and the FFN patcher.

These are the rules we care about enforcing from M1 onward:

1. LowRankLinear.from_linear with rank = min(in, out) reproduces the dense linear
   numerically (up to fp tolerance).
2. The debug fallback path exactly reproduces the original linear's output.
3. The patcher is idempotent w.r.t. parameter counts for the same rank.
4. Lower ranks always reduce parameter count.

Runs on CPU with small random tensors — no GPU required.
"""
from __future__ import annotations

import torch
from torch import nn

from adaptive_llm_speed.methods.ffn_lowrank.layers import LowRankLinear


def _make_dense(in_f: int, out_f: int, bias: bool = False, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    L = nn.Linear(in_f, out_f, bias=bias)
    return L


def test_full_rank_equivalent_to_dense():
    L = _make_dense(32, 64)
    LR = LowRankLinear.from_linear(L, rank=32)  # min(32, 64)
    x = torch.randn(4, 32)
    y_dense = L(x)
    y_lr = LR(x)
    assert torch.allclose(y_dense, y_lr, atol=1e-4), (y_dense - y_lr).abs().max()


def test_debug_fallback_exact():
    L = _make_dense(16, 32)
    LR = LowRankLinear.from_linear(L, rank=4, keep_dense_for_debug=True)
    LR._debug_exact = True
    x = torch.randn(3, 16)
    y = LR(x)
    y_expected = L(x)
    assert torch.equal(y, y_expected)


def test_bias_preserved():
    L = _make_dense(8, 16, bias=True)
    LR = LowRankLinear.from_linear(L, rank=8)
    assert LR.up.bias is not None
    assert torch.equal(LR.up.bias.data, L.bias.data)


def test_param_count_formula():
    in_f, out_f, r = 64, 96, 8
    LR = LowRankLinear(in_f, out_f, r, bias=True)
    expected = in_f * r + r * out_f + out_f
    actual = sum(p.numel() for p in LR.parameters())
    assert actual == expected, (actual, expected)
    assert LR.params_kept == expected
    assert LR.params_original == in_f * out_f + out_f


def test_lower_rank_reduces_params():
    in_f, out_f = 64, 96
    p_prev = None
    for r in (4, 8, 16, 32):
        LR = LowRankLinear(in_f, out_f, r)
        p = sum(p.numel() for p in LR.parameters())
        if p_prev is not None:
            assert p >= p_prev
        p_prev = p


def test_from_linear_preserves_dtype_bf16():
    """If we hand in a bf16 linear, the replacement must also be bf16 everywhere.

    This caught a regression where the factorized submodules stayed at the
    nn.Linear default (fp32) and blew up with dtype-mismatch errors during the
    first forward pass on a bf16 model.
    """
    L = _make_dense(16, 32, bias=True).to(torch.bfloat16)
    LR = LowRankLinear.from_linear(L, rank=8)
    assert LR.down.weight.dtype == torch.bfloat16
    assert LR.up.weight.dtype == torch.bfloat16
    assert LR.up.bias.dtype == torch.bfloat16
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    y = LR(x)
    assert y.dtype == torch.bfloat16


def test_from_linear_preserves_dtype_fp16():
    L = _make_dense(16, 32, bias=False).to(torch.float16)
    LR = LowRankLinear.from_linear(L, rank=4)
    assert LR.down.weight.dtype == torch.float16
    assert LR.up.weight.dtype == torch.float16
