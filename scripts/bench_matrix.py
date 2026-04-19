"""Latency benchmark matrix with variance reporting.

Runs (method config × context length) combinations under a single process
with controlled warmup/iter counts, and reports median / p95 / std per cell.
One results JSON per invocation — plots downstream pull from it directly.

Important: no quality measurement here. This script is intentionally scoped
to wall-clock only. Quality runs through `scripts/run_eval.py` with the
matching configs.

Usage:
  ./tb python3 scripts/bench_matrix.py \\
    --model configs/base/model_qwen3_0p6b.yaml \\
    --methods baseline configs/attention/attn_block64_topk8_residual_gather_shared.yaml \\
              configs/attention/attn_bound_screen_topk6_residual.yaml \\
    --ctx-lens 1024 2048 4096 8192 16384 \\
    --iters 10 --warmup 3 \\
    --out results/bench/bench_matrix.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from adaptive_llm_speed.models.loaders import load_model
from adaptive_llm_speed.models.patching import apply_method
from adaptive_llm_speed.utils.config import load_yaml, merge
from adaptive_llm_speed.utils.seed import set_seed
from adaptive_llm_speed.utils.timers import time_many


def _prefill_fn(model, input_ids):
    def f():
        model(input_ids, use_cache=False)
    return f


def _decode_fn(model, tokenizer, input_ids, new_tokens):
    def f():
        model.generate(
            input_ids, max_new_tokens=new_tokens, do_sample=False, use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or 0,
        )
    return f


@torch.inference_mode()
def bench_one(model, tokenizer, model_cfg, method_cfg_or_name: str | None,
              ctx_lens: list[int], decode_len: int,
              warmup_prefill: int, iters_prefill: int,
              warmup_decode: int, iters_decode: int,
              ) -> dict[str, Any]:
    name = "baseline" if method_cfg_or_name in (None, "baseline") else Path(method_cfg_or_name).stem
    if method_cfg_or_name not in (None, "baseline"):
        method_cfg = load_yaml(method_cfg_or_name)
        cfg = merge(model_cfg, method_cfg)
        patch_meta = apply_method(model, cfg)
    else:
        patch_meta = {"method": "baseline"}

    # Flush any CUDA residue from a prior config in the same process.
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)

    rows = []
    for L in ctx_lens:
        torch.manual_seed(0)
        input_ids = torch.randint(0, vocab_size, (1, L), device=device, dtype=torch.long)
        pre = time_many(_prefill_fn(model, input_ids), warmup=warmup_prefill, iters=iters_prefill)
        dec = time_many(_decode_fn(model, tokenizer, input_ids, decode_len),
                        warmup=warmup_decode, iters=iters_decode)
        rows.append({
            "ctx_len": L,
            "prefill_median_ms": pre.median_ms,
            "prefill_mean_ms": pre.mean_ms,
            "prefill_p95_ms": pre.p95_ms,
            "prefill_std_ms": pre.std_ms,
            "decode_total_median_ms": dec.median_ms,
            "decode_total_p95_ms": dec.p95_ms,
            "decode_tok_s_approx": (decode_len / (dec.median_ms / 1000.0)) if dec.median_ms > 0 else float("inf"),
        })

    # Unpatch: put the model back to clean state so subsequent methods reuse the same weights.
    from adaptive_llm_speed.methods.adaptive_attention.patch import unpatch_adaptive_attention
    if patch_meta.get("method") == "adaptive_attention":
        unpatch_adaptive_attention(model)

    return {"name": name, "patch_meta": patch_meta, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="method config paths, or the literal 'baseline'")
    ap.add_argument("--ctx-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--decode-len", type=int, default=16)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--decode-iters", type=int, default=3)
    ap.add_argument("--decode-warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bench/bench_matrix.json")
    args = ap.parse_args()

    set_seed(args.seed)
    model_cfg = load_yaml(args.model)
    loaded = load_model(model_cfg)
    model = loaded.model

    all_results = []
    for method in args.methods:
        print(f"[bench] {method}")
        res = bench_one(
            model, loaded.tokenizer, model_cfg, method,
            ctx_lens=args.ctx_lens, decode_len=args.decode_len,
            warmup_prefill=args.warmup, iters_prefill=args.iters,
            warmup_decode=args.decode_warmup, iters_decode=args.decode_iters,
        )
        for row in res["rows"]:
            print(f"  ctx={row['ctx_len']:>6}  prefill med={row['prefill_median_ms']:.1f}ms "
                  f"p95={row['prefill_p95_ms']:.1f}ms  std={row['prefill_std_ms']:.1f}ms  "
                  f"decode~{row['decode_tok_s_approx']:.1f} tok/s")
        all_results.append(res)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_cfg["model"]["name"],
        "ctx_lens": args.ctx_lens,
        "iters_prefill": args.iters,
        "warmup_prefill": args.warmup,
        "iters_decode": args.decode_iters,
        "warmup_decode": args.decode_warmup,
        "results": all_results,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
