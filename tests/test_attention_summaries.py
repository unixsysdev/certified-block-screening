"""Unit tests for block summary builders."""
from __future__ import annotations

import torch

from adaptive_llm_speed.methods.adaptive_attention.summaries import block_mean_summary


def test_shapes_and_padding():
    B, H, L, D = 1, 2, 100, 8  # L = 100, block_size = 16 -> Nb = 7, pad_len = 12
    K = torch.randn(B, H, L, D)
    V = torch.randn(B, H, L, D)
    K_bar, V_bar, pad_len = block_mean_summary(K, V, block_size=16)
    assert pad_len == 12
    assert K_bar.shape == (B, H, 7, D)
    assert V_bar.shape == (B, H, 7, D)


def test_mean_values_exact_on_clean_division():
    B, H, L, D = 1, 1, 8, 2  # two blocks of size 4
    K = torch.tensor([[[[1., 0.], [2., 0.], [3., 0.], [4., 0.],   # block 0 mean = (2.5, 0)
                        [5., 1.], [6., 1.], [7., 1.], [8., 1.]]]])  # block 1 mean = (6.5, 1)
    V = K.clone()
    K_bar, V_bar, pad_len = block_mean_summary(K, V, block_size=4)
    assert pad_len == 0
    assert torch.allclose(K_bar[0, 0, 0], torch.tensor([2.5, 0.]))
    assert torch.allclose(K_bar[0, 0, 1], torch.tensor([6.5, 1.]))
    assert torch.equal(K_bar, V_bar)


def test_zero_padding_does_not_affect_real_blocks():
    # When L is not a multiple of block_size, padding should not change means of complete blocks.
    B, H, L, D = 1, 1, 6, 1
    K = torch.tensor([[[[1.], [2.], [3.], [4.], [5.], [6.]]]])
    V = K.clone()
    K_bar, _, pad_len = block_mean_summary(K, V, block_size=4)
    assert pad_len == 2
    # Block 0 spans indices 0..3 = mean 2.5; block 1 is padded with last value (6), so = (5+6+6+6)/4 = 5.75
    assert torch.allclose(K_bar[0, 0, 0], torch.tensor([2.5]))
    assert torch.allclose(K_bar[0, 0, 1], torch.tensor([5.75]))
