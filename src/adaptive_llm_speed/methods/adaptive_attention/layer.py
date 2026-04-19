"""Adaptive attention core.

Two modes, both preserving the Qwen3Attention forward signature:

- **coarse_replace**: for each query, pick the top-k blocks by coarse score
  q·k_bar; run exact softmax attention over tokens in the selected blocks only;
  unselected blocks contribute nothing.

- **residual_refine**: same as coarse_replace, plus each unselected block
  contributes its (k_bar, v_bar) summary as a single virtual key/value. The
  softmax is joint over selected real tokens + unselected summaries. This is
  the safer mode the research plan recommends starting from.

v0 implementation uses mask-based attention — we materialise a (B, H, Lq, Lk)
attention matrix and zero out positions in unselected blocks. This is **not
faster than full attention** — same matmul shape — but it is correct, it lets
us evaluate the quality loss of a given (block_size, top_k) setting, and it
gives us a clean target for a future gather-based fast path.

Strict baseline switch: if `cfg['debug_exact']` is true, we fall through to the
baseline (full) attention and skip the selector entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .selectors import FixedTopKSelector
from .summaries import block_mean_summary


@dataclass
class AdaptiveAttentionCfg:
    block_size: int = 64
    top_k: int = 4
    mode: str = "coarse_replace"  # or "residual_refine"
    selector: str = "fixed_topk"  # or "bound_screen"
    delta: float = 1.0            # margin on the radius term for bound_screen
    summary: str = "mean"
    impl: str = "mask"            # "mask", "gather", "gather_shared", "gather_windowed"
    query_window_size: int = 0    # only used by gather_windowed. 0 -> auto (= block_size).
    query_score: str = "max"      # aggregation within a window: "max" | "mean" | "mean_plus_max"
    rescue_k: int = 0             # for mean_plus_max: extra k picked by max on top of k_mean
    num_prototypes: int = 1       # 1 = classic single-center summary. >1 = M sub-block prototypes;
                                  # selection uses max_m U_{r,m}(q); gather + residual stay at block level.
    debug_exact: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveAttentionCfg":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AdaptiveAttentionStats:
    """Method-level telemetry accumulated across forward calls on a single layer."""
    n_forwards: int = 0
    avg_exact_blocks: float = 0.0        # running mean across queries of `min(top_k, num_blocks)`
    avg_num_blocks: float = 0.0          # running mean of total blocks per forward
    avg_prefill_len: float = 0.0         # running mean of Lk (observed per-layer)
    # Bound-tightness running stats — only populated when a bound_screen selector runs.
    bound_tightness_median_sum: float = 0.0
    bound_violation_rate_sum: float = 0.0
    bound_n: int = 0
    last: dict[str, float] = field(default_factory=dict)

    def observe(self, *, lq: int, lk: int, num_blocks: int, used_blocks: int) -> None:
        n = self.n_forwards + 1
        self.avg_exact_blocks += (used_blocks - self.avg_exact_blocks) / n
        self.avg_num_blocks += (num_blocks - self.avg_num_blocks) / n
        self.avg_prefill_len += (lk - self.avg_prefill_len) / n
        self.last = {"lq": lq, "lk": lk, "num_blocks": num_blocks, "used_blocks": used_blocks}
        self.n_forwards = n

    def observe_bound(self, *, tightness_median: float, violation_rate: float) -> None:
        self.bound_tightness_median_sum += tightness_median
        self.bound_violation_rate_sum += violation_rate
        self.bound_n += 1

    def to_dict(self) -> dict[str, Any]:
        out = {
            "n_forwards": self.n_forwards,
            "avg_exact_blocks": self.avg_exact_blocks,
            "avg_num_blocks": self.avg_num_blocks,
            "avg_prefill_len": self.avg_prefill_len,
            "last": self.last,
        }
        if self.bound_n > 0:
            out["avg_bound_tightness_median"] = self.bound_tightness_median_sum / self.bound_n
            out["avg_bound_violation_rate"] = self.bound_violation_rate_sum / self.bound_n
            out["bound_samples"] = self.bound_n
        return out


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for grouped-query attention.

    x: (B, H_kv, L, D) -> (B, H_kv * n_rep, L, D). Equivalent to
    repeat_interleave(n_rep, dim=1) but with broadcasting semantics that keeps
    the operation cheap until a materialize is forced downstream.
    """
    if n_rep == 1:
        return x
    B, H_kv, L, D = x.shape
    x = x.unsqueeze(2).expand(B, H_kv, n_rep, L, D).reshape(B, H_kv * n_rep, L, D)
    return x


def _build_causal_block_mask(Lq: int, Lk: int, block_size: int, q_start: int,
                             device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build helpers for causal masking under blocking.

    Returns:
        block_causal: (Lq, Nb) bool. True iff the block contains at least one
            key position that query i (global position q_start + i) is allowed
            to see, i.e. r * block_size <= q_pos.
        key_causal: (Lq, Lk_padded) bool. True iff key j is allowed for query i.
    """
    Nb = (Lk + block_size - 1) // block_size
    Lk_pad = Nb * block_size
    q_positions = torch.arange(Lq, device=device) + q_start        # (Lq,)
    k_positions = torch.arange(Lk_pad, device=device)              # (Lk_pad,)
    key_causal = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)  # (Lq, Lk_pad)
    key_causal = key_causal & (k_positions.unsqueeze(0) < Lk)          # drop padded positions
    # A block is valid if its first position is ≤ q_pos.
    block_first = torch.arange(Nb, device=device) * block_size         # (Nb,)
    block_causal = block_first.unsqueeze(0) <= q_positions.unsqueeze(1)  # (Lq, Nb)
    return block_causal, key_causal


def adaptive_attention(
    query: torch.Tensor,            # (B, H_q, Lq, D)
    key: torch.Tensor,              # (B, H_kv, Lk, D)
    value: torch.Tensor,            # (B, H_kv, Lk, D)
    attention_mask: torch.Tensor | None,  # (B, 1, Lq, Lk) or None; additive or bool; ignored here
    *,
    cfg: AdaptiveAttentionCfg,
    scaling: float,
    q_start: int = 0,
    stats: AdaptiveAttentionStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Coarse-to-exact attention. Returns (attn_output, None).

    Signature chosen so this can slot into Qwen3Attention.forward's attention_interface position.
    `q_start` is the position of query[0] in the full sequence (past_len for decoding).
    """
    if cfg.debug_exact:
        return _full_attention(query, key, value, scaling, q_start=q_start)

    B, H_q, Lq, D = query.shape
    _, H_kv, Lk, _ = key.shape
    n_rep = H_q // H_kv
    device = query.device

    # --- Build block summaries on K, V. Pads Lk up to a multiple of block_size internally.
    K_bar, V_bar, pad_len = block_mean_summary(key, value, cfg.block_size)
    Nb = K_bar.shape[-2]
    Lk_pad = Nb * cfg.block_size

    # --- Score blocks: per query (per head), s_{h,i,r} = q_{h,i} · K_bar_{h_kv, r}
    K_bar_rep = _repeat_kv(K_bar, n_rep)                          # (B, H_q, Nb, D)
    scores = torch.einsum("bhqd,bhrd->bhqr", query, K_bar_rep) * scaling  # (B, H_q, Lq, Nb)

    # --- Causal block validity: block r is valid for query at q_start+i iff r*b <= q_pos.
    block_causal, key_causal = _build_causal_block_mask(Lq, Lk, cfg.block_size, q_start, device)
    # Broadcast to (1, 1, Lq, Nb) / (1, 1, Lq, Lk_pad).
    block_valid = block_causal.unsqueeze(0).unsqueeze(0).expand(B, H_q, Lq, Nb)
    key_valid = key_causal.unsqueeze(0).unsqueeze(0).expand(B, H_q, Lq, Lk_pad)

    # --- Select top-k blocks per query. sel: (B, H_q, Lq, k).
    selector = FixedTopKSelector(cfg.top_k)
    sel = selector(scores, block_valid_mask=block_valid)
    k_eff = sel.shape[-1]

    # --- Expand selected block indices into per-key indices, yielding a boolean
    #     "allowed" mask over Lk_pad for every query.
    #     allowed[b, h, i, j] = True iff block_of(j) ∈ sel[b, h, i, :]
    Nb_range = torch.arange(Nb, device=device).view(1, 1, 1, Nb)
    selected_block_mask = (sel.unsqueeze(-1) == Nb_range).any(dim=-2)  # (B, H_q, Lq, Nb)
    # expand to per-key mask.
    per_key_mask = selected_block_mask.repeat_interleave(cfg.block_size, dim=-1)  # (B, H_q, Lq, Lk_pad)

    # --- Residual mode: unselected blocks contribute as summaries. We merge the
    #     summary logits into the softmax for the unselected blocks only.
    #     Implementation: we still run the main softmax over key positions, but
    #     we augment with an extra "virtual" summary column per unselected block.
    if cfg.mode == "residual_refine":
        # Summary logits for unselected valid blocks.
        summary_logits = scores.clone()
        summary_logits = summary_logits.masked_fill(~block_valid, float("-inf"))
        summary_logits = summary_logits.masked_fill(selected_block_mask, float("-inf"))
    elif cfg.mode == "coarse_replace":
        summary_logits = None
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    # --- Pad key/value to Lk_pad to align with per_key_mask.
    if pad_len > 0:
        pad_k = key.new_zeros(B, H_kv, pad_len, D)
        pad_v = value.new_zeros(B, H_kv, pad_len, D)
        key_full = torch.cat([key, pad_k], dim=-2)
        value_full = torch.cat([value, pad_v], dim=-2)
    else:
        key_full, value_full = key, value

    K_rep = _repeat_kv(key_full, n_rep)                             # (B, H_q, Lk_pad, D)
    V_rep = _repeat_kv(value_full, n_rep)
    # Token-level logits.
    logits = torch.einsum("bhqd,bhkd->bhqk", query, K_rep) * scaling  # (B, H_q, Lq, Lk_pad)
    allowed = per_key_mask & key_valid                                # token in selected block AND causal-valid AND not padding
    logits = logits.masked_fill(~allowed, float("-inf"))

    if summary_logits is not None:
        # Concatenate token logits with summary logits on the last dim: (B, H_q, Lq, Lk_pad + Nb).
        # Same for values: V is the real tokens, then V_bar stacked for unselected valid blocks.
        V_bar_rep = _repeat_kv(V_bar, n_rep)                        # (B, H_q, Nb, D)
        combined_logits = torch.cat([logits, summary_logits], dim=-1)      # (..., Lk_pad + Nb)
        combined_values = torch.cat([V_rep, V_bar_rep], dim=-2)            # (B, H_q, Lk_pad + Nb, D)
        # Expand combined_values across queries for the weighted sum below.
        weights = F.softmax(combined_logits, dim=-1)                        # (B, H_q, Lq, Lk_pad + Nb)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, combined_values)
    else:
        # Any row where everything is -inf (no selected block valid for query) collapses to NaN;
        # we guard by replacing those rows with the original row-sum-zero value vector (i.e. zeros).
        all_masked = (~allowed).all(dim=-1, keepdim=True)                   # (B, H_q, Lq, 1)
        weights = F.softmax(logits, dim=-1)
        weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, V_rep)

    if stats is not None:
        stats.observe(lq=Lq, lk=Lk, num_blocks=Nb, used_blocks=k_eff)

    return out, None


def _full_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    scaling: float, *, q_start: int = 0) -> tuple[torch.Tensor, None]:
    """Debug/baseline path: exact causal softmax attention, all tokens visible."""
    B, H_q, Lq, D = query.shape
    _, H_kv, Lk, _ = key.shape
    n_rep = H_q // H_kv
    K_rep = _repeat_kv(key, n_rep)
    V_rep = _repeat_kv(value, n_rep)
    logits = torch.einsum("bhqd,bhkd->bhqk", query, K_rep) * scaling
    # Causal mask (with q_start offset for decode).
    device = query.device
    q_positions = torch.arange(Lq, device=device) + q_start
    k_positions = torch.arange(Lk, device=device)
    causal = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)         # (Lq, Lk)
    logits = logits.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    weights = F.softmax(logits, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", weights, V_rep)
    return out, None
