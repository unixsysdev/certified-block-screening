"""Drop-in low-rank replacement for nn.Linear.

Semantics match nn.Linear(in, out, bias) but weight is factored:
    y = x @ V^T @ U^T + b
with U: (out, rank), V: (rank, in). We express that with two nn.Linear modules
so standard optimizers, quantizers, and fine-tuning workflows keep working.

Invariants (enforced by tests):
- from_linear(L).forward(x) ≈ L(x) up to SVD truncation error
- parameter count is in*rank + rank*out + (out if bias else 0)
- supports a debug flag to bypass the low-rank path and call the original linear
"""
from __future__ import annotations

import torch
from torch import nn

from .factorize import truncated_svd_linear


class LowRankLinear(nn.Module):
    """y = U (V x) + b, where W ≈ U V is a rank-r factorization of an nn.Linear."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False) -> None:
        super().__init__()
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError(f"rank {rank} out of range for linear ({in_features},{out_features})")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        # Two linears, no intermediate bias. Bias stays on the outer projection.
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
        # Strict-baseline switch: when True, forward falls through to the captured dense linear.
        self._debug_exact = False
        self._dense_fallback: nn.Linear | None = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, *, keep_dense_for_debug: bool = False) -> "LowRankLinear":
        has_bias = linear.bias is not None
        mod = cls(linear.in_features, linear.out_features, rank, bias=has_bias)
        # Match device AND dtype of the original linear so we don't leak fp32 into a bf16 model.
        mod.to(device=linear.weight.device, dtype=linear.weight.dtype)
        U, V = truncated_svd_linear(linear.weight.data, rank)
        dst_dtype = linear.weight.dtype
        dst_device = linear.weight.device
        # nn.Linear.weight is (out, in); down.weight is (rank, in) = V; up.weight is (out, rank) = U.
        mod.down.weight.data.copy_(V.to(device=dst_device, dtype=dst_dtype))
        mod.up.weight.data.copy_(U.to(device=dst_device, dtype=dst_dtype))
        if has_bias:
            mod.up.bias.data.copy_(linear.bias.data.to(device=dst_device, dtype=dst_dtype))
        if keep_dense_for_debug:
            # Hold a reference so we can flip back to the exact path. Not registered as a submodule,
            # so its params don't show up in model.parameters() and don't get optimized.
            object.__setattr__(mod, "_dense_fallback", linear)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._debug_exact and self._dense_fallback is not None:
            return self._dense_fallback(x)
        return self.up(self.down(x))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, bias={self.up.bias is not None}"

    @property
    def params_kept(self) -> int:
        p = self.in_features * self.rank + self.rank * self.out_features
        if self.up.bias is not None:
            p += self.out_features
        return p

    @property
    def params_original(self) -> int:
        p = self.in_features * self.out_features
        if self.up.bias is not None:
            p += self.out_features
        return p
