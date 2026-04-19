"""Block selectors for adaptive attention.

A selector takes per-block coarse scores (and, optionally, bounds) and returns
a set of block indices to run exact attention on. Variants:

- FixedTopK (A1)      — plain top-k of q · c_r. What the FixedTopK runs earlier used.
- BoundScreen (A2)    — certified upper-bound screening. Picks top-k by
                        U_r(q) = q·c_r + ||q||·ρ_r, optionally with a margin δ
                        that biases toward or away from the radius term.
- LearnedGate (A3)    — tiny MLP on block features (future).
"""
from __future__ import annotations

import torch


class FixedTopKSelector:
    """Per-query top-k selection by score. Stateless and batched."""

    def __init__(self, top_k: int) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k

    def __call__(self, scores: torch.Tensor, block_valid_mask: torch.Tensor | None = None,
                 ) -> torch.Tensor:
        """Return indices of selected blocks.

        Args:
            scores: (B, H_q, Lq, Nb) block-level attention logits per query.
            block_valid_mask: (B, H_q, Lq, Nb) bool or {0,1} — False/0 means the
                block is invalid for this query (e.g. entirely in the future under
                a causal mask). Invalid blocks get -inf score before top-k.
        Returns:
            (B, H_q, Lq, k) int64, where k = min(self.top_k, Nb).
        """
        if block_valid_mask is not None:
            # Cast to bool to avoid -inf * 0 NaNs, then replace invalid with -inf.
            mask = block_valid_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, float("-inf"))
        Nb = scores.shape[-1]
        k = min(self.top_k, Nb)
        # torch.topk along last dim. For rows where all scores are -inf (e.g. first-query-with-no-valid-blocks,
        # which can't happen under sensible causal masking but we guard anyway) topk still returns indices;
        # the caller's gate mask (in layer.py) zeroes their contribution.
        _, idx = torch.topk(scores, k=k, dim=-1)
        return idx


class BoundScreenSelector:
    """Top-k by certified upper bound U_r(q) = q·c_r + δ·||q||·ρ_r.

    The margin δ interpolates:
      δ = 0  → ranking by coarse score alone (equivalent to FixedTopK).
      δ = 1  → ranking by the full Cauchy–Schwarz bound (most-informative but
               most conservative; favours wider blocks even when their center
               is not as aligned).

    The guarantee (when δ = 1): any block we drop has `max_j q·k_j ≤ U_r(q)`,
    so if U_r(q) is below the kept threshold there is a *proof* that dropping
    is safe relative to the coarse-score ordering. Tests in
    `test_attention_bound_screen.py` enforce this property.
    """

    def __init__(self, top_k: int, delta: float = 1.0) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        self.delta = float(delta)

    def __call__(self, coarse_score: torch.Tensor, radius_term: torch.Tensor,
                 block_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return indices of selected blocks.

        Args:
            coarse_score: (B, H_q, Lq, Nb) — q·c_r (already scaled).
            radius_term: (B, H_q, Lq, Nb) — ||q||·ρ_r (already scaled).
                Caller can pre-multiply by `delta` if they want one fewer
                operand; we apply it here for clarity.
            block_valid_mask: (B, H_q, Lq, Nb) bool or {0,1}.
        Returns:
            (B, H_q, Lq, k) int64, where k = min(self.top_k, Nb).
        """
        bound = coarse_score + self.delta * radius_term
        if block_valid_mask is not None:
            mask = block_valid_mask.to(dtype=torch.bool)
            bound = bound.masked_fill(~mask, float("-inf"))
        Nb = bound.shape[-1]
        k = min(self.top_k, Nb)
        _, idx = torch.topk(bound, k=k, dim=-1)
        return idx
