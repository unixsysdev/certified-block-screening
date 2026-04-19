"""Patch + evaluate. Same eval suite as profile_baseline, but runs through a method patch first.

Usage:
    ./tb python3 scripts/run_eval.py \
        --model configs/base/model_qwen3_0p6b.yaml \
        --method configs/ffn/ffn_rank_384.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adaptive_llm_speed.eval.composite_score import composite_q
from adaptive_llm_speed.eval.latency import profile_latency_grid
from adaptive_llm_speed.eval.perplexity import perplexity_on_calibration
from adaptive_llm_speed.models.loaders import load_model, model_info
from adaptive_llm_speed.models.patching import apply_method
from adaptive_llm_speed.utils.config import config_hash, load_yaml, merge
from adaptive_llm_speed.utils.io import save_result
from adaptive_llm_speed.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model config YAML")
    ap.add_argument("--method", required=True, help="method config YAML")
    ap.add_argument("--baseline-result", default=None, help="path to baseline JSON for composite Q")
    ap.add_argument("--ctx-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--decode-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    model_cfg = load_yaml(args.model)
    method_cfg = load_yaml(args.method)
    cfg = merge(model_cfg, method_cfg)
    cfg_hash = config_hash(cfg)
    run_id = f"{cfg['method']['name']}/{Path(args.model).stem}/{Path(args.method).stem}/{cfg_hash}"

    print(f"[eval] loading {cfg['model']['name']} for method={cfg['method']['name']}")
    loaded = load_model(cfg)
    info = model_info(loaded)

    print(f"[eval] applying patch ...")
    patch_meta = apply_method(loaded.model, cfg)
    print(f"[eval] patched: {patch_meta}")

    # Put model back in eval mode after any in-place module swap.
    loaded.model.eval()

    print("[eval] perplexity ...")
    ppl = perplexity_on_calibration(loaded.model, loaded.tokenizer, loaded.device)
    print(f"[eval] ppl={ppl['perplexity']:.4f}")

    latency = {}
    if not args.skip_latency:
        print(f"[eval] latency grid ctx_lens={args.ctx_lens}")
        latency = profile_latency_grid(
            loaded.model, loaded.tokenizer, loaded.device,
            ctx_lens=args.ctx_lens, decode_len=args.decode_len,
        )
        for row in latency["prefill"]:
            print(f"  prefill ctx={row['ctx_len']:>6}: median={row['median_ms']:.2f}ms")
        for row in latency["decode"]:
            print(f"  decode  ctx={row['ctx_len']:>6}: ~{row['decode_tok_s_approx']:.1f} tok/s")

    metrics = {
        "perplexity": ppl["perplexity"],
        "nll_mean": ppl["nll_mean"],
        "peak_memory_mb": latency.get("peak_memory_mb", 0.0),
    }

    q_report = None
    if args.baseline_result:
        import json
        with open(args.baseline_result) as f:
            baseline = json.load(f)
        q_report = composite_q(metrics, baseline["metrics"])
        print(f"[eval] composite Q = {q_report['q']:.4f} vs baseline {args.baseline_result}")

    payload = {
        "config_hash": cfg_hash,
        "config": cfg,
        "model_config_path": args.model,
        "method_config_path": args.method,
        "model_info": info,
        "method": cfg["method"],
        "patch_meta": patch_meta,
        "metrics": metrics,
        "latency": latency,
        "composite_q": q_report,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
    }
    path = save_result(args.results_dir, run_id, payload)
    print(f"[eval] wrote {path}")


if __name__ == "__main__":
    main()
