"""Block summaries for the adaptive attention coarse pass.

A summary compresses a block of `block_size` consecutive keys (or values) into
a single vector. The simplest choice — mean — is what we ship in v0. More
elaborate choices (mean+max concat, learned projection, PCA) go in other
functions here and are picked via config.

Expected tensor layout throughout this module:
    K, V: (B, H_kv, Lk, D)    — batch, kv-heads, key length, head dim
    K_bar, V_bar: (B, H_kv, Nb, D) with Nb = ceil(Lk / block_size)

We pad Lk up to a multiple of block_size by repeating the last key/value
(and marking the padded positions in a block-valid mask). The caller is
expected to also build a mask that says "this block has at least one real
token that the current query is allowed to see" — see causal masking in
`layer.py`.
"""
from __future__ import annotations

import torch


def _pad_to_block_multiple(x: torch.Tensor, block_size: int, dim: int) -> tuple[torch.Tensor, int]:
    # Normalize negative dim so the per-axis expand shape can index it directly.
    if dim < 0:
        dim += x.ndim
    L = x.shape[dim]
    rem = L % block_size
    if rem == 0:
        return x, 0
    pad_len = block_size - rem
    # Repeat the last slice along `dim` as padding. The block-validity mask in layer.py
    # will ignore padded positions; the actual values don't matter for correctness.
    expand_shape = [-1 if i != dim else pad_len for i in range(x.ndim)]
    pad_slice = x.narrow(dim, L - 1, 1).expand(*expand_shape)
    x_padded = torch.cat([x, pad_slice], dim=dim)
    return x_padded, pad_len


def block_mean_summary(K: torch.Tensor, V: torch.Tensor, block_size: int
                       ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (K_bar, V_bar, pad_len). Shapes: (B, H_kv, Nb, D)."""
    K_padded, pad_len = _pad_to_block_multiple(K, block_size, dim=-2)
    V_padded, _ = _pad_to_block_multiple(V, block_size, dim=-2)
    B, H, L_pad, D = K_padded.shape
    Nb = L_pad // block_size
    K_blocks = K_padded.reshape(B, H, Nb, block_size, D)
    V_blocks = V_padded.reshape(B, H, Nb, block_size, D)
    K_bar = K_blocks.mean(dim=-2)
    V_bar = V_blocks.mean(dim=-2)
    return K_bar, V_bar, pad_len
