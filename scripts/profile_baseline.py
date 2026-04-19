"""Baseline profiling: load a model, run perplexity + latency grid, dump JSON.

Usage:
    ./tb python3 scripts/profile_baseline.py --config configs/base/model_qwen3_0p6b.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adaptive_llm_speed.eval.latency import profile_latency_grid
from adaptive_llm_speed.eval.perplexity import perplexity_on_calibration
from adaptive_llm_speed.models.loaders import load_model, model_info
from adaptive_llm_speed.utils.config import config_hash, load_yaml
from adaptive_llm_speed.utils.io import save_result
from adaptive_llm_speed.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ctx-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--decode-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="results/raw")
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_yaml(args.config)
    cfg_hash = config_hash(cfg)
    run_id = f"baseline/{Path(args.config).stem}/{cfg_hash}"

    print(f"[profile] loading {cfg['model']['name']}")
    loaded = load_model(cfg)
    info = model_info(loaded)
    print(f"[profile] loaded: {info['num_hidden_layers']} layers, hidden={info['hidden_size']}, "
          f"ffn={info['intermediate_size']}, params={info['total_params']:,}")

    print("[profile] perplexity on calibration text ...")
    ppl = perplexity_on_calibration(loaded.model, loaded.tokenizer, loaded.device)
    print(f"[profile] ppl={ppl['perplexity']:.4f} over {ppl['tokens']} tokens")

    latency = {}
    if not args.skip_latency:
        print(f"[profile] latency grid: ctx_lens={args.ctx_lens} decode_len={args.decode_len}")
        latency = profile_latency_grid(
            loaded.model, loaded.tokenizer, loaded.device,
            ctx_lens=args.ctx_lens, decode_len=args.decode_len,
        )
        for row in latency["prefill"]:
            print(f"  prefill ctx={row['ctx_len']:>6}: median={row['median_ms']:.2f}ms  p95={row['p95_ms']:.2f}ms")
        for row in latency["decode"]:
            print(f"  decode  ctx={row['ctx_len']:>6}: ~{row['decode_tok_s_approx']:.1f} tok/s "
                  f"(per-step ~{row['per_step_ms_approx']:.2f}ms)")
        print(f"  peak mem: {latency['peak_memory_mb']:.1f} MB")

    payload = {
        "config_path": args.config,
        "config_hash": cfg_hash,
        "config": cfg,
        "method": {"name": "baseline"},
        "model_info": info,
        "metrics": {
            "perplexity": ppl["perplexity"],
            "nll_mean": ppl["nll_mean"],
            "peak_memory_mb": latency.get("peak_memory_mb", 0.0),
        },
        "latency": latency,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
    }
    path = save_result(args.results_dir, run_id, payload)
    print(f"[profile] wrote {path}")


if __name__ == "__main__":
    main()
