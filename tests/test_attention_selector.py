"""Unit tests for FixedTopKSelector."""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.selectors import FixedTopKSelector


def test_picks_highest():
    # (B=1, H=1, Lq=1, Nb=4). Scores chosen so order is 2, 0, 3, 1.
    scores = torch.tensor([[[[0.1, 0.0, 0.5, 0.2]]]])
    sel = FixedTopKSelector(top_k=2)(scores)
    # top-2 by value are indices {2, 3}.
    s = set(sel[0, 0, 0].tolist())
    assert s == {2, 3}, s


def test_respects_validity_mask():
    scores = torch.tensor([[[[0.9, 0.8, 0.7, 0.6]]]])
    valid = torch.tensor([[[[False, True, True, False]]]])
    sel = FixedTopKSelector(top_k=2)(scores, block_valid_mask=valid)
    s = set(sel[0, 0, 0].tolist())
    assert s == {1, 2}


def test_top_k_clamped_to_nb():
    scores = torch.randn(1, 1, 1, 3)
    sel = FixedTopKSelector(top_k=10)(scores)
    # With Nb = 3 and top_k = 10, selector returns 3 indices (no crash).
    assert sel.shape == (1, 1, 1, 3)


def test_rejects_zero_top_k():
    try:
        FixedTopKSelector(top_k=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
