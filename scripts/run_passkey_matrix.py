"""Passkey retrieval across (method × ctx × depth), one process.

Loads the model once, then for each method applies the patch, runs the passkey
suite, unpatches, moves on. All results land in a single JSON so the downstream
plot (correct-fraction vs k; correct-fraction vs depth, broken out by method)
reads from one file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adaptive_llm_speed.eval.retrieval import run_passkey_suite
from adaptive_llm_speed.methods.adaptive_attention.patch import unpatch_adaptive_attention
from adaptive_llm_speed.models.loaders import load_model
from adaptive_llm_speed.models.patching import apply_method
from adaptive_llm_speed.utils.config import load_yaml, merge
from adaptive_llm_speed.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="method config paths, or the literal 'baseline'")
    ap.add_argument("--ctx-lens", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/retrieval/passkey_matrix.json")
    args = ap.parse_args()

    set_seed(args.seed)
    model_cfg = load_yaml(args.model)
    loaded = load_model(model_cfg)
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device

    all_results = []
    for method in args.methods:
        print(f"[passkey] {method}")
        if method not in (None, "baseline"):
            cfg = merge(model_cfg, load_yaml(method))
            patch_meta = apply_method(model, cfg)
        else:
            patch_meta = {"method": "baseline"}

        res = run_passkey_suite(
            model, tokenizer, device,
            ctx_lens=args.ctx_lens, depths=args.depths,
            n_trials_per_cell=args.trials, seed=args.seed,
        )
        for row in res["rows"]:
            print(f"  ctx={row['ctx_len']:>5}  depth={row['depth']:.2f}  "
                  f"correct={row['correct_frac']:.2f}  lpt={row['mean_logprob_per_token']:.3f}")

        all_results.append({
            "method": Path(method).stem if method != "baseline" else "baseline",
            "patch_meta": patch_meta,
            "rows": res["rows"],
        })
        if patch_meta.get("method") == "adaptive_attention":
            unpatch_adaptive_attention(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "model": model_cfg["model"]["name"],
            "ctx_lens": args.ctx_lens,
            "depths": args.depths,
            "trials": args.trials,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"[passkey] wrote {out_path}")


if __name__ == "__main__":
    main()
