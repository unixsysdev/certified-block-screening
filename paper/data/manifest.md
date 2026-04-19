# Data manifest

One-line source of truth for every number, figure, and table in the paper. Re-run
`./tb python3 scripts/make_paper_artifacts.py` to regenerate tables/figures from
these JSONs — never copy numbers by hand into the manuscript.

## Raw latency benchmarks (prefill, same-session, variance reported)

| file | methods | ctx lens | iters |
|------|---------|----------|-------|
| `results/bench/bench_matrix_v1.json` | baseline, fixed_topk k=8, bound_screen k=6 | 1 k / 2 k / 4 k / 8 k | 10 |
| `results/bench/bench_fixed_topk6_short.json` | fixed_topk k=6 | 1 k / 2 k / 4 k / 8 k | 10 |
| `results/bench/bench_matrix_long_ctx.json` | baseline, fixed_topk k=6, fixed_topk k=8, bound_screen k=6 | 16 k / 32 k | 5 |
| `results/bench/bench_matrix_64k.json` | baseline, fixed_topk k=6, bound_screen k=6 | 64 k | 3 |

## Single-config eval JSONs (PPL + latency grid)

| file pattern | selector × k | used in |
|--------------|---------------|---------|
| `results/raw/baseline_model_qwen3_0p6b_*.json` | dense SDPA | PPL reference |
| `results/raw/adaptive_attention_*_attn_block64_topk8_residual_gather_shared_*.json` | fixed_topk k=8 | Table 2 |
| `results/raw/adaptive_attention_*_attn_fixed_topk{5,6}_residual_gather_shared_*.json` | fixed_topk k∈{5,6} | Table 2 |
| `results/raw/adaptive_attention_*_attn_bound_screen_topk{4,5,6,8}_residual_*.json` | bound_screen k∈{4,5,6,8} | Table 2 |
| `results/raw/adaptive_attention_*_attn_bound_mp{2,4}_topk6_*.json` | multi-prototype M∈{2,4} k=6 | Appendix / §6.2 |

fixed_topk k=4 PPL is taken from `docs/failure_log.md` (legacy run, coarse_replace; value 13,865 documented with its config).

## Bound-tightness probes

| file | ctx | block | k | source |
|------|-----|-------|---|--------|
| `results/tightness/model_qwen3_0p6b_ctx2048_b64_k4_d1.0.json` | 2 k | 64 | 4 | Fig 4 |
| `results/tightness/model_qwen3_0p6b_ctx2048_b64_k6_d1.0.json` | 2 k | 64 | 6 | Fig 4 |
| `results/tightness/model_qwen3_0p6b_ctx2048_b64_k8_d1.0.json` | 2 k | 64 | 8 | Fig 4 |
| `results/tightness/model_qwen3_0p6b_ctx8192_b64_k6_d1.0.json` | 8 k | 64 | 6 | Fig 4 |
| `results/tightness/model_qwen3_0p6b_ctx16384_b64_k6_d1.0.json` | 16 k | 64 | 6 | Fig 4 |
| `results/tightness/model_qwen3_0p6b_ctx32768_b64_k6_d1.0.json` | 32 k | 64 | 6 | Fig 4 |

## Passkey retrieval probes

| file | focus |
|------|-------|
| `results/retrieval/passkey_small.json` | shared-mean b=64 baseline (first negative result) |
| `results/retrieval/passkey_windowed.json` | windowed + max-aggregation ablations at b=64 |
| `results/retrieval/passkey_block_size_sweep.json` | b=64 / 32 / 16 at windowed_max |
| `results/retrieval/passkey_fine_grained.json` | b=16 / 8 with five-depth sweep |
| `results/retrieval/passkey_delta_and_coarse.json` | high-δ at b=64 and b=16 coarse_replace |
| `results/retrieval/passkey_multiproto.json` | M ∈ {1,2,4,8} vs b=16 reference |

## Consolidated paper artifacts (regenerated from the above)

| file | section |
|------|---------|
| `paper/tables/table1_latency_vs_ctx.md` | §5.1 |
| `paper/tables/table2_ppl_vs_k.md` | §5.2 |
| `paper/tables/table3_passkey.md` | §6.1 |
| `paper/figures/fig2_latency_vs_ctx.{png,pdf}` | §5.1 |
| `paper/figures/fig3_ppl_vs_k.{png,pdf}` | §5.2 |
| `paper/figures/fig4_tightness_and_overlap.{png,pdf}` | §5.3 |
| `paper/figures/fig5_passkey.{png,pdf}` | §6.1 |
| `paper/data/latency_vs_ctx.csv` | raw for Fig 2 |
| `paper/data/ppl_vs_k.csv` | raw for Fig 3 |
| `paper/data/tightness.csv` | raw for Fig 4 |
| `paper/data/passkey.csv` | raw for Fig 5 / Table 3 |

Fig 1 (method diagram) is drawn separately; not sourced from JSON.

## Known gaps (closed as of 2026-04-19)

- ~~`fixed_topk k=6` at ctx 1k–8k~~ — closed via `bench_fixed_topk6_short.json` and `bench_matrix_short_v2.json`.
- ~~Tightness at 16k / 32k~~ — closed; set-overlap with `fixed_topk` drops monotonically with ctx (0.64 → 0.45 → 0.40 → 0.37 at k=6).
- ~~Bound variance at 32k~~ — confirmed: zero violations, slack_rel median 9.48 (actually *tighter* than shorter-context runs, likely because larger Nb gives more denominator cells).
- Open: fixed_topk k=8 at 64k — single missing cell in Table 1; not essential for the paper claim (the 64k data is bracketed as appendix due to exceeding Qwen3's training context).
- Open: bound_screen shared-mean passkey rows at depths 0.3 / 0.7 — low value; the paper's §6.1 narrative uses the windowed+max and multi-prototype configs, both of which have full depth coverage.
