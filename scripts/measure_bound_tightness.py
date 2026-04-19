"""Measure bound slack U_r(q) − max_j q·k_j on calibration queries.

Produces the "certified" paper artifact: a per-layer, per-block-size distribution
of how loose the certified upper bound is. Without this, the A2 contribution is
mechanism-only; with it, we can claim the bound is tight enough to matter.

Collects:
  - slack_abs:  U_r − max_j q·k_j                     (always ≥ 0)
  - slack_rel:  slack_abs / max(|max_j q·k_j|, eps)    (unitless)
  - separate distributions for (a) blocks the selector keeps in top-k and
    (b) blocks it drops, so we can say "the bound is tight on the blocks
    it decides to discard."
  - fixed-topk overlap: what fraction of bound_screen's selection equals
    fixed_topk's selection on the same data. If overlap is high, the bound
    is mostly agreeing with coarse ranking and only rescuing edge cases;
    if it's low, the bound is restructuring the selection.

Usage:
  ./tb python3 scripts/measure_bound_tightness.py \
      --model configs/base/model_qwen3_0p6b.yaml \
      --block-size 64 --top-k 6 --delta 1.0 --ctx-len 2048

Outputs: results/tightness/<model>_<ctx>_<block>_<topk>.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from adaptive_llm_speed.methods.adaptive_attention.bounds import (
    compute_block_centers_and_radii,
    upper_bound_logits,
)
from adaptive_llm_speed.methods.adaptive_attention.layer import _repeat_kv
from adaptive_llm_speed.methods.adaptive_attention.selectors import (
    BoundScreenSelector,
    FixedTopKSelector,
)
from adaptive_llm_speed.models.loaders import load_model
from adaptive_llm_speed.utils.config import load_yaml
from adaptive_llm_speed.utils.seed import set_seed


@torch.inference_mode()
def probe_layer(attn_module, hidden_states, position_embeddings, *,
                block_size: int, top_k: int, delta: float,
                ) -> dict[str, Any]:
    """Run the layer's q/k/v projections once, collect per-block slack stats."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    self = attn_module
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    B, H_q, Lq, D = q.shape
    _, H_kv, Lk, _ = k.shape
    n_rep = H_q // H_kv

    K_bar, V_bar, radii, K_padded, pad_len = compute_block_centers_and_radii(k, v, block_size)
    Nb = K_bar.shape[-2]
    K_bar_rep = _repeat_kv(K_bar, n_rep)
    radii_rep = _repeat_kv(radii.unsqueeze(-1), n_rep).squeeze(-1)

    scaling = self.scaling
    upper, coarse, _ = upper_bound_logits(q, K_bar_rep, radii_rep, scaling=scaling)

    # Compute actual per-block max logits: max_j q·k_j * scaling.
    K_blocks = K_padded.reshape(B, H_kv, Nb, block_size, D)
    K_blocks_rep = _repeat_kv(K_blocks.reshape(B, H_kv, Nb * block_size, D), n_rep).reshape(
        B, H_q, Nb, block_size, D
    )
    # logits[b, h, q, r, j] = q[b,h,q,:] · k[b,h,r,j,:] * scaling
    logits = torch.einsum("bhqd,bhrjd->bhqrj", q, K_blocks_rep) * scaling
    # Mask out padded positions in last block: only first (block_size - pad_len) are real in that block.
    if pad_len > 0:
        mask = torch.ones(Nb, block_size, dtype=torch.bool, device=q.device)
        mask[-1, block_size - pad_len:] = False
        logits = logits.masked_fill(~mask.view(1, 1, 1, Nb, block_size), float("-inf"))
    actual_max = logits.max(dim=-1).values   # (B, H_q, Lq, Nb)

    # Block-validity under causal attention: block r can be seen by q_i iff r*b <= i.
    block_first = torch.arange(Nb, device=q.device) * block_size
    q_pos = torch.arange(Lq, device=q.device)
    block_valid = (block_first.view(1, Nb) <= q_pos.view(Lq, 1)).view(1, 1, Lq, Nb).expand(B, H_q, Lq, Nb)

    # Slack only on valid (b, h, q, r) cells with finite actual_max.
    valid = block_valid & torch.isfinite(actual_max)
    if not valid.any():
        return {"ok": False, "reason": "no valid cells"}

    slack_abs = (upper - actual_max)[valid].float()
    slack_rel = slack_abs / actual_max[valid].abs().clamp(min=1e-6).float()

    # torch.quantile has a 2^24-element hard limit. Subsample if we exceeded that.
    # The sample size caps stay large enough for stable p95/p99 estimates.
    MAX_Q_SAMPLE = 1 << 23   # 8.4M, well under the limit and plenty for quantiles.
    if slack_abs.numel() > MAX_Q_SAMPLE:
        idx = torch.randint(0, slack_abs.numel(), (MAX_Q_SAMPLE,), device=slack_abs.device)
        slack_abs = slack_abs[idx]
        slack_rel = slack_rel[idx]

    # Selector outputs for the same data: where does bound_screen disagree with fixed_topk?
    fixed_sel = FixedTopKSelector(top_k=top_k)(coarse, block_valid_mask=block_valid)
    # BoundScreen takes (coarse, radius_term) — the radius term must be pre-scaled the same way.
    radius_term = (q.float().norm(dim=-1).unsqueeze(-1) * radii_rep.float().unsqueeze(-2) * scaling).to(q.dtype)
    bound_sel = BoundScreenSelector(top_k=top_k, delta=delta)(
        coarse, radius_term, block_valid_mask=block_valid
    )
    # Overlap fraction: per (b, h, q), how many of the top_k picks agree?
    # Each sel is (B, H_q, Lq, k). Compare as sorted sets.
    fixed_sorted = fixed_sel.sort(dim=-1).values
    bound_sorted = bound_sel.sort(dim=-1).values
    overlap_mask = (fixed_sorted == bound_sorted)
    overlap_frac = overlap_mask.float().mean().item()  # per-position element agreement, not set overlap
    # Stricter: fraction of (b,h,q) rows where the two selections are identical as a set.
    row_identical = overlap_mask.all(dim=-1).float().mean().item()
    # Also: average set-intersection size divided by top_k.
    # Compare each element of bound_sel against the fixed_sel set via broadcasting.
    # bound_sel: (..., k), fixed_sel: (..., k). Reshape for comparison:
    set_overlap_size = (bound_sel.unsqueeze(-1) == fixed_sel.unsqueeze(-2)).any(dim=-2).float().sum(dim=-1)
    avg_set_overlap_frac = (set_overlap_size / top_k).mean().item()

    return {
        "ok": True,
        "Lq": int(Lq), "Lk": int(Lk), "Nb": int(Nb),
        "n_cells": int(valid.sum().item()),
        "slack_abs": {
            "mean": float(slack_abs.mean().item()),
            "median": float(slack_abs.median().item()),
            "p95": float(torch.quantile(slack_abs, 0.95).item()),
            "p99": float(torch.quantile(slack_abs, 0.99).item()),
            "max": float(slack_abs.max().item()),
        },
        "slack_rel": {
            "mean": float(slack_rel.mean().item()),
            "median": float(slack_rel.median().item()),
            "p95": float(torch.quantile(slack_rel, 0.95).item()),
            "p99": float(torch.quantile(slack_rel, 0.99).item()),
        },
        "violations": int((slack_abs < -1e-3).sum().item()),  # must be 0 for certified bound
        "fixed_topk_vs_bound_screen": {
            "positionwise_equal_frac": overlap_frac,
            "row_identical_frac": row_identical,
            "avg_set_overlap_frac": avg_set_overlap_frac,
        },
    }


@torch.inference_mode()
def collect(model, tokenizer, device: str, *,
            ctx_len: int, block_size: int, top_k: int, delta: float,
            calibration_text: str,
            ) -> dict[str, Any]:
    # Tokenise + truncate/pad to ctx_len.
    ids = tokenizer(calibration_text, return_tensors="pt")["input_ids"][0].to(device)
    if ids.numel() < ctx_len:
        # Pad by repeating (ok for a smoke-test of the bound).
        reps = (ctx_len + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(reps)[:ctx_len]
    else:
        ids = ids[:ctx_len]
    input_ids = ids.unsqueeze(0)

    # Forward hook on every attention module: capture its (hidden_states, position_embeddings).
    captures: list[tuple[int, torch.Tensor, tuple]] = []

    def _hook_factory(layer_idx):
        def _pre(mod, args, kwargs):
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            pe = kwargs.get("position_embeddings")
            if pe is None and len(args) > 1:
                pe = args[1]
            captures.append((layer_idx, hs.detach(), (pe[0].detach(), pe[1].detach())))
            return args, kwargs
        return _pre

    handles = []
    for i, block in enumerate(model.model.layers):
        h = block.self_attn.register_forward_pre_hook(_hook_factory(i), with_kwargs=True)
        handles.append(h)

    try:
        model(input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # Now replay each capture through `probe_layer` to collect slack stats.
    per_layer = []
    for layer_idx, hs, pe in captures:
        attn = model.model.layers[layer_idx].self_attn
        stats = probe_layer(attn, hs, pe, block_size=block_size, top_k=top_k, delta=delta)
        if stats.get("ok"):
            stats["layer_idx"] = layer_idx
            per_layer.append(stats)

    # Aggregate across layers.
    def _pool(field_path, reducer=statistics.mean):
        vals = []
        for row in per_layer:
            cur = row
            for k in field_path:
                cur = cur[k]
            vals.append(cur)
        return reducer(vals) if vals else float("nan")

    total_violations = sum(row["violations"] for row in per_layer)

    return {
        "ctx_len": ctx_len,
        "block_size": block_size,
        "top_k": top_k,
        "delta": delta,
        "n_layers_probed": len(per_layer),
        "aggregate": {
            "slack_abs_mean": _pool(("slack_abs", "mean")),
            "slack_abs_median": _pool(("slack_abs", "median")),
            "slack_abs_p95": _pool(("slack_abs", "p95")),
            "slack_rel_mean": _pool(("slack_rel", "mean")),
            "slack_rel_median": _pool(("slack_rel", "median")),
            "slack_rel_p95": _pool(("slack_rel", "p95")),
            "bound_vs_topk_row_identical_mean": _pool(
                ("fixed_topk_vs_bound_screen", "row_identical_frac")),
            "bound_vs_topk_set_overlap_mean": _pool(
                ("fixed_topk_vs_bound_screen", "avg_set_overlap_frac")),
            "total_violations": total_violations,
        },
        "per_layer": per_layer,
    }


_DEFAULT_CAL = """
Information theory is the mathematical study of the quantification, storage,
and communication of information. The field was established by Claude Shannon
in the 1940s. A key measure in information theory is entropy, which quantifies
the amount of uncertainty in a random variable. Shannon's insight enabled
lossless data compression, channel coding, cryptography, and many other fields.

The theory has applications in statistical inference, linguistics, molecular
biology, thermal physics, quantum computing, and plagiarism detection. Its
breadth reflects the generality of the mathematical machinery Shannon
introduced to model communication over a noisy channel. Entropy, mutual
information, and channel capacity are the three canonical tools.
""" * 8  # plenty of tokens for 2k+ contexts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/tightness")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_yaml(args.model)
    loaded = load_model(cfg)
    out = collect(
        loaded.model, loaded.tokenizer, loaded.device,
        ctx_len=args.ctx_len, block_size=args.block_size, top_k=args.top_k,
        delta=args.delta, calibration_text=_DEFAULT_CAL,
    )
    out["model"] = cfg["model"]["name"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{Path(args.model).stem}_ctx{args.ctx_len}_b{args.block_size}_k{args.top_k}_d{args.delta}.json"
    path = out_dir / fname
    with path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[tightness] wrote {path}")
    agg = out["aggregate"]
    print(f"  slack_abs   mean={agg['slack_abs_mean']:.3f} median={agg['slack_abs_median']:.3f} "
          f"p95={agg['slack_abs_p95']:.3f}")
    print(f"  slack_rel   mean={agg['slack_rel_mean']:.3f} median={agg['slack_rel_median']:.3f} "
          f"p95={agg['slack_rel_p95']:.3f}")
    print(f"  row-identical vs fixed_topk: {agg['bound_vs_topk_row_identical_mean']:.3f}")
    print(f"  set-overlap  vs fixed_topk: {agg['bound_vs_topk_set_overlap_mean']:.3f}")
    print(f"  total bound violations: {agg['total_violations']} (must be 0)")


if __name__ == "__main__":
    main()
