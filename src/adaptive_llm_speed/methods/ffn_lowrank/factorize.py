"""Truncated SVD factorization of a dense linear weight.

Given W: (out, in), we want W ≈ U V where U: (out, r), V: (r, in). The two
sequential matmuls cost (in*r + r*out) which beats (in*out) for r < in*out/(in+out).

We absorb the singular values symmetrically: U = Uh @ diag(sqrt(S)), V = diag(sqrt(S)) @ Vh^T.
That keeps both factors at similar numerical scale, which helps downstream fine-tuning.
"""
from __future__ import annotations

import torch


def truncated_svd_linear(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (U, V) with W ≈ U @ V, U: (out, rank), V: (rank, in).

    Runs in float32 on the weight's device. Safe for bf16 input weights.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    if rank <= 0 or rank > min(out_dim, in_dim):
        raise ValueError(f"rank {rank} must be in [1, min(out,in)={min(out_dim, in_dim)}]")

    orig_dtype = weight.dtype
    W = weight.detach().to(torch.float32)
    # full_matrices=False is essential — gives reduced SVD sized for min(out, in).
    Uh, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Uh = Uh[:, :rank].contiguous()
    S = S[:rank]
    Vh = Vh[:rank, :].contiguous()

    sqrt_s = torch.sqrt(S)
    U = (Uh * sqrt_s.unsqueeze(0)).to(orig_dtype)
    V = (sqrt_s.unsqueeze(1) * Vh).to(orig_dtype)
    return U, V


def energy_retained(weight: torch.Tensor, rank: int) -> float:
    """Fraction of squared-singular-value energy kept by a rank-r truncation."""
    W = weight.detach().to(torch.float32)
    S = torch.linalg.svdvals(W)
    total = (S**2).sum().item()
    if total == 0:
        return 1.0
    return float((S[:rank] ** 2).sum().item() / total)


def reconstruction_error(weight: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> dict:
    W = weight.detach().to(torch.float32)
    W_hat = (U.to(torch.float32) @ V.to(torch.float32))
    diff = W - W_hat
    return {
        "frobenius": float(torch.linalg.norm(diff).item()),
        "relative_frobenius": float(torch.linalg.norm(diff).item() / max(torch.linalg.norm(W).item(), 1e-12)),
        "max_abs": float(diff.abs().max().item()),
    }
