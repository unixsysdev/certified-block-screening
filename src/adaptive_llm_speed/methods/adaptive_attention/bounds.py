"""Certified upper bounds on per-block attention logits.

For block r containing keys {k_j}_{j in B_r} of dimension D, define:

    c_r = mean of keys in B_r        # center
    ρ_r = max_j ||k_j - c_r||_2      # radius

Then for any query q of dimension D:

    max_j  q · k_j  ≤  q · c_r  +  ||q||_2 · ρ_r  =:  U_r(q)                (★)

Proof (one line): q·k_j = q·c_r + q·(k_j − c_r). Apply Cauchy–Schwarz to the
second term and bound by radius.

Inequality (★) is tight when k_j − c_r is parallel to q and has length exactly
ρ_r. For Gaussian-ish key clouds it's usually loose by a factor of √D or so,
but the *direction* of looseness (always an upper bound, never under) is what
makes it useful for screening: we can drop a block whose U_r(q) is below some
threshold without ever losing an actually-important key.

This module produces `(c_r, ρ_r)` per (batch, H_kv, block) and the per-query
bound `U_r(q)` per (batch, H_q, Lq, block).

Metric we track — **bound tightness**:

    tightness_r(q)  =  U_r(q) / max_j q · k_j

Always ≥ 1 by (★). Reporting it tells reviewers whether the bound is doing
useful screening or is vacuous. A median close to 1 means the bound rarely
over-estimates; a long tail of large ratios means the bound is loose.
"""
from __future__ import annotations

import torch

from .summaries import _pad_to_block_multiple


def compute_block_centers_and_radii(
    K: torch.Tensor, V: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return (K_bar, V_bar, radii, padded_K, pad_len).

    Shapes (B, H_kv, Nb, D) for K_bar / V_bar / padded-K-by-block,
    and (B, H_kv, Nb) for radii.

    Invariant: for every real (non-padded) token k_j in block r,
    ||k_j - K_bar[.., r, :]||_2 ≤ radii[.., r].

    Perf note: the earlier version did a full fp32 cast of the per-token
    differences before the norm. On Qwen3-0.6B prefill at 8k this added ~200 ms
    per layer × 28 layers. We now accumulate the squared distances in fp32 via
    `sum(dtype=torch.float32)` — no fp32 copy of the full diffs tensor, same
    numerical stability for the reduction.
    """
    K_padded, pad_len = _pad_to_block_multiple(K, block_size, dim=-2)
    V_padded, _ = _pad_to_block_multiple(V, block_size, dim=-2)
    B, H, L_pad, D = K_padded.shape
    Nb = L_pad // block_size
    K_blocks = K_padded.reshape(B, H, Nb, block_size, D)
    V_blocks = V_padded.reshape(B, H, Nb, block_size, D)
    K_bar = K_blocks.mean(dim=-2)                                          # (B, H, Nb, D)
    V_bar = V_blocks.mean(dim=-2)
    diffs = K_blocks - K_bar.unsqueeze(-2)                                 # (B, H, Nb, b, D) in K's dtype
    # Accumulate the sum of squares in fp32 without allocating a full fp32 copy of `diffs`.
    radii_sq = (diffs * diffs).sum(dim=-1, dtype=torch.float32)            # (B, H, Nb, b) fp32
    radii = radii_sq.max(dim=-1).values.sqrt().to(K.dtype)                 # (B, H, Nb)
    return K_bar, V_bar, radii, K_padded, pad_len


def upper_bound_logits(
    query: torch.Tensor,        # (B, H_q, Lq, D)
    K_bar: torch.Tensor,        # (B, H_q, Nb, D)   — already GQA-expanded
    radii: torch.Tensor,        # (B, H_q, Nb)      — already GQA-expanded
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (upper_bound, coarse_score, query_norm) all scaled by `scaling`.

    upper_bound[b,h,i,r] = (q_i·c_r + ||q_i||·ρ_r) · scaling
    coarse_score[b,h,i,r] = q_i · c_r · scaling

    scaling is applied to both so comparisons are in the same units the downstream
    softmax uses.
    """
    # q·c: einsum over D. Cast to fp32 to avoid bf16 underflow for small radii.
    qcomp = torch.einsum("bhqd,bhrd->bhqr", query.to(torch.float32), K_bar.to(torch.float32))
    q_norm = query.to(torch.float32).norm(dim=-1)                          # (B, H_q, Lq)
    # Broadcast: (B, H_q, Lq, 1) * (B, H_q, 1, Nb) -> (B, H_q, Lq, Nb).
    margin = q_norm.unsqueeze(-1) * radii.to(torch.float32).unsqueeze(-2)
    coarse = (qcomp * scaling).to(query.dtype)
    upper = ((qcomp + margin) * scaling).to(query.dtype)
    return upper, coarse, q_norm.to(query.dtype)


def compute_block_multiproto(
    K: torch.Tensor, V: torch.Tensor, block_size: int, num_prototypes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Per-block summaries with M prototypes, each a certified (center, radius) pair.

    A block of `block_size` tokens is split into `num_prototypes` equal sub-blocks
    of size `b / M`. Each sub-block produces one (center, radius) pair whose
    invariant is

        ∀ j in sub_m of block r:  ||k_j − c_{r,m}||_2  ≤  ρ_{r,m}      (★)

    So `max_j q·k_j` in sub_m is bounded by `q·c_{r,m} + ||q||·ρ_{r,m}` via
    Cauchy–Schwarz, and the *block-level* bound is

        U_r(q)  =  max_{m=1..M}  U_{r,m}(q)

    which is still a certified upper bound because the pointwise max of upper
    bounds is an upper bound on the pointwise max.

    Returns (K_multi, V_multi, radii, K_overall, V_overall, pad_len):
        K_multi:   (B, H_kv, Nb, M, D)  — per-prototype centers
        V_multi:   (B, H_kv, Nb, M, D)
        radii:     (B, H_kv, Nb, M)     — per-prototype radii
        K_overall: (B, H_kv, Nb, D)     — block mean (= average of prototype centers)
        V_overall: (B, H_kv, Nb, D)     — used for residual summaries so the key count
                                           in the softmax stays Nb, not Nb·M.
    """
    if block_size % num_prototypes != 0:
        raise ValueError(f"block_size ({block_size}) must be divisible by num_prototypes ({num_prototypes})")
    sub_b = block_size // num_prototypes
    K_padded, pad_len = _pad_to_block_multiple(K, block_size, dim=-2)
    V_padded, _ = _pad_to_block_multiple(V, block_size, dim=-2)
    B, H, L_pad, D = K_padded.shape
    Nb = L_pad // block_size

    # (B, H, Nb, M, sub_b, D)
    K_sub = K_padded.reshape(B, H, Nb, num_prototypes, sub_b, D)
    V_sub = V_padded.reshape(B, H, Nb, num_prototypes, sub_b, D)
    K_multi = K_sub.mean(dim=-2)                                           # (B, H, Nb, M, D)
    V_multi = V_sub.mean(dim=-2)
    diffs = K_sub - K_multi.unsqueeze(-2)                                   # (B, H, Nb, M, sub_b, D)
    radii_sq = (diffs * diffs).sum(dim=-1, dtype=torch.float32)             # (B, H, Nb, M, sub_b)
    radii = radii_sq.max(dim=-1).values.sqrt().to(K.dtype)                  # (B, H, Nb, M)

    K_overall = K_multi.mean(dim=-2)                                       # (B, H, Nb, D)
    V_overall = V_multi.mean(dim=-2)
    return K_multi, V_multi, radii, K_overall, V_overall, pad_len


class BoundCache:
    """Per-layer cache of block centers, V means, and radii keyed by KV length.

    Every layer's attention forward gets exactly one instance. During decode
    (Lq=1, kv_len grows by 1 each call) most blocks don't change; the cache
    lets the radius computation skip all but the last block. During a fresh
    prefill the cache invalidates automatically (kv_len jumps) and we do a
    full recompute once.

    We compare against (kv_len, block_size, dtype, device) because any of
    those changing breaks the invariant.

    Incremental update is done only when kv_len grows by exactly 1 (the
    autoregressive-decode case). Anything else recomputes from scratch; the
    logic is simpler than tracking which block each new token lands in.
    """

    __slots__ = (
        "kv_len", "block_size", "num_prototypes", "dtype", "device",
        "K_bar", "V_bar", "radii",                          # single-prototype / "overall"
        "K_multi", "V_multi", "radii_multi",                 # per-prototype (None when M=1)
        "pad_len",
    )

    def __init__(self) -> None:
        self.kv_len: int = -1
        self.block_size: int = -1
        self.num_prototypes: int = 1
        self.dtype = None
        self.device = None
        self.K_bar = None
        self.V_bar = None
        self.radii = None
        self.K_multi = None
        self.V_multi = None
        self.radii_multi = None
        self.pad_len: int = 0

    def invalidate(self) -> None:
        for s in self.__slots__:
            setattr(self, s, -1 if s in ("kv_len", "block_size") else (1 if s == "num_prototypes" else None))
        self.pad_len = 0

    def _matches(self, key: torch.Tensor, block_size: int, num_prototypes: int) -> bool:
        return (
            self.kv_len == key.shape[-2]
            and self.block_size == block_size
            and self.num_prototypes == num_prototypes
            and self.dtype == key.dtype
            and self.device == key.device
        )

    def _can_increment(self, key: torch.Tensor, block_size: int, num_prototypes: int) -> bool:
        # Incremental update only supports the single-prototype path. Re-enable later if we need
        # multi-prototype incremental cache; for now fall back to full recompute when M > 1.
        return (
            num_prototypes == 1
            and self.num_prototypes == 1
            and self.kv_len == key.shape[-2] - 1
            and self.block_size == block_size
            and self.dtype == key.dtype
            and self.device == key.device
        )

    def _store_single(self, key, K_bar, V_bar, radii, pad_len, block_size) -> None:
        self.kv_len = key.shape[-2]
        self.block_size = block_size
        self.num_prototypes = 1
        self.dtype = key.dtype
        self.device = key.device
        self.K_bar = K_bar
        self.V_bar = V_bar
        self.radii = radii
        self.K_multi = None
        self.V_multi = None
        self.radii_multi = None
        self.pad_len = pad_len

    def _store_multi(self, key, K_multi, V_multi, radii_multi, K_overall, V_overall, pad_len,
                     block_size, num_prototypes) -> None:
        self.kv_len = key.shape[-2]
        self.block_size = block_size
        self.num_prototypes = num_prototypes
        self.dtype = key.dtype
        self.device = key.device
        self.K_bar = K_overall
        self.V_bar = V_overall
        # The "aggregate" radius isn't stored — callers that want a single-prototype radius
        # pass num_prototypes=1 and get the classic path.
        self.radii = None
        self.K_multi = K_multi
        self.V_multi = V_multi
        self.radii_multi = radii_multi
        self.pad_len = pad_len

    def get_or_compute(self, key: torch.Tensor, value: torch.Tensor, block_size: int
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Single-prototype path (M=1). Returns (K_bar, V_bar, radii, pad_len)."""
        if self._matches(key, block_size, 1):
            return self.K_bar, self.V_bar, self.radii, self.pad_len
        if self._can_increment(key, block_size, 1):
            self._increment(key, value, block_size)
            return self.K_bar, self.V_bar, self.radii, self.pad_len
        # Full recompute path.
        K_bar, V_bar, radii, _, pad_len = compute_block_centers_and_radii(key, value, block_size)
        self._store_single(key, K_bar, V_bar, radii, pad_len, block_size)
        return K_bar, V_bar, radii, pad_len

    def get_or_compute_multi(self, key: torch.Tensor, value: torch.Tensor,
                             block_size: int, num_prototypes: int,
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                         torch.Tensor, torch.Tensor, int]:
        """Multi-prototype path. Returns (K_multi, V_multi, radii_multi, K_overall, V_overall, pad_len).

        Cache reuses M=1 entries if they're compatible via `K_overall = K_bar`. But the per-prototype
        tensors always require a fresh compute on cache miss (we don't incrementally update per
        sub-block; falls back to full recompute).
        """
        if self._matches(key, block_size, num_prototypes):
            return self.K_multi, self.V_multi, self.radii_multi, self.K_bar, self.V_bar, self.pad_len
        K_multi, V_multi, radii_multi, K_overall, V_overall, pad_len = compute_block_multiproto(
            key, value, block_size, num_prototypes
        )
        self._store_multi(key, K_multi, V_multi, radii_multi, K_overall, V_overall, pad_len,
                          block_size, num_prototypes)
        return K_multi, V_multi, radii_multi, K_overall, V_overall, pad_len

    def _increment(self, key: torch.Tensor, value: torch.Tensor, block_size: int) -> None:
        """One new token appended. Only the block that now contains (or will contain) it changes.

        We recompute center + radius for exactly that block — O(b·D) instead of
        O(Lk·D) for the full path.
        """
        new_kv_len = key.shape[-2]
        # Index of the block that now includes the newly-appended token (at position new_kv_len - 1).
        affected = (new_kv_len - 1) // block_size

        # If adding this token grows Nb (we crossed a block boundary), extend the cached tensors.
        cached_Nb = self.K_bar.shape[-2]
        new_Nb = (new_kv_len + block_size - 1) // block_size
        if new_Nb > cached_Nb:
            # Need one more row in K_bar, V_bar, radii. Zero-pad; will be overwritten below.
            extra = new_Nb - cached_Nb
            B, H, _, D = self.K_bar.shape
            self.K_bar = torch.cat(
                [self.K_bar, self.K_bar.new_zeros(B, H, extra, D)], dim=-2
            )
            self.V_bar = torch.cat(
                [self.V_bar, self.V_bar.new_zeros(B, H, extra, D)], dim=-2
            )
            self.radii = torch.cat(
                [self.radii, self.radii.new_zeros(B, H, extra)], dim=-1
            )

        # Slice out the affected block's real tokens from `key`, compute fresh center + radius.
        block_start = affected * block_size
        block_end = min(block_start + block_size, new_kv_len)
        k_block = key[..., block_start:block_end, :]
        v_block = value[..., block_start:block_end, :]
        # Center: mean over the *real* tokens in the block. Padded-side tokens were never part of
        # K; they don't exist in the tensor. When the block isn't yet full, the center represents
        # only the real tokens we've seen — consistent with what the full-path compute does when
        # the sequence length is a partial block (we pad by repeating the last token, which the
        # mean-of-real path implicitly matches the full-path mean up to pad-repeat artifacts).
        # For strict consistency with block_mean_summary's "repeat-last-token" padding, we pad
        # the block here the same way.
        real_len = block_end - block_start
        if real_len < block_size:
            pad_len = block_size - real_len
            pad_k = k_block[..., -1:, :].expand(*k_block.shape[:-2], pad_len, k_block.shape[-1])
            pad_v = v_block[..., -1:, :].expand(*v_block.shape[:-2], pad_len, v_block.shape[-1])
            k_full_block = torch.cat([k_block, pad_k], dim=-2)
            v_full_block = torch.cat([v_block, pad_v], dim=-2)
        else:
            k_full_block = k_block
            v_full_block = v_block
        new_center_k = k_full_block.mean(dim=-2)                # (B, H, D)
        new_center_v = v_full_block.mean(dim=-2)
        diffs = k_full_block - new_center_k.unsqueeze(-2)
        new_radius_sq = (diffs * diffs).sum(dim=-1, dtype=torch.float32)
        new_radius = new_radius_sq.max(dim=-1).values.sqrt().to(key.dtype)

        self.K_bar[..., affected, :] = new_center_k
        self.V_bar[..., affected, :] = new_center_v
        self.radii[..., affected] = new_radius

        self.kv_len = new_kv_len
        self.block_size = block_size
        self.dtype = key.dtype
        self.device = key.device
        # pad_len here is the "last-block padding" the full-path would report for the current
        # kv_len — kept consistent so downstream code that gates on pad_len still works.
        self.pad_len = (new_Nb * block_size) - new_kv_len


def bound_tightness_stats(
    upper: torch.Tensor,          # (B, H_q, Lq, Nb) — U_r scaled
    actual_max: torch.Tensor,     # (B, H_q, Lq, Nb) — max_j q·k_j scaled
    *,
    valid_mask: torch.Tensor | None = None,  # (B, H_q, Lq, Nb) bool
) -> dict[str, float]:
    """Summarise U_r / max_j q·k_j across a batch. Always ≥ 1 by construction."""
    eps = 1e-6
    # Only compare where the actual max is strictly positive; ratio is ill-defined otherwise.
    pos = actual_max > eps
    mask = pos if valid_mask is None else (pos & valid_mask.bool())
    if not mask.any():
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan"), "count": 0}
    ratio = upper[mask] / actual_max[mask]
    # On correctness: any ratio strictly < 1 means the bound failed. Count those.
    violations = (ratio < 1.0 - 1e-3).float().mean().item()
    return {
        "median": float(ratio.median().item()),
        "p95": float(torch.quantile(ratio.float(), 0.95).item()),
        "max": float(ratio.max().item()),
        "violation_rate": violations,
        "count": int(mask.sum().item()),
    }
