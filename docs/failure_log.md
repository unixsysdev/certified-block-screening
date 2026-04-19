# Failure log

Negative results kept in-repo so we don't rediscover them.

## 2026-04-19 — Pure SVD FFN compression (F0) collapses Qwen3-0.6B

Setup: Qwen3-0.6B, bf16, rank-768 SVD on every FFN projection (gate, up, down) in all 28 layers. No fine-tuning.

Result: calibration PPL exploded from **11.07 → 19,224.76** on the fixed passage. Composite Q ≈ 0 vs. baseline.

Why it broke: the FFN weights of Qwen3-0.6B are not low-rank. Per-weight singular-value spectrum on `layers[0].mlp.gate_proj` (shape 3072×1024, so max rank = 1024):

| rank | energy retained | rel. Frobenius error |
|------|-----------------|----------------------|
| 1024 | 1.00            | 0.00                 |
|  768 | 0.93            | 0.26                 |
|  512 | 0.79            | 0.46                 |
|  384 | 0.68            | 0.56                 |
|  256 | 0.54            | 0.68                 |
|  128 | 0.36            | 0.80                 |

Rank 768 is already *above* the parameter break-even point for a 1024-in / 3072-out matrix (break-even at 768). So every rank that would actually save parameters has ≥46% reconstruction error per weight, and the errors compound across 28 × 3 = 84 projections.

Takeaway: Phase F0 (SVD-only, no retraining) is a floor, not a Pareto point. The plan originally put fine-tuning at M5; we're keeping that order but moving to M2 (adaptive attention) before coming back to F1.

Artifact: `results/raw/ffn_lowrank_model_qwen3_0p6b_ffn_rank_768_5f66844e6f5d.json`.

## 2026-04-19 — Mask-based adaptive attention loses to SDPA on wall-clock

Setup: Qwen3-0.6B with adaptive attention patched on all 28 layers, block_size=64,
top_k=8, residual_refine mode. Baseline run with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.

Result: **Quality holds (PPL 11.086 vs 11.096 baseline, Q=0.999)** but **latency regresses**:
prefill at 2k context is 533 ms vs 207 ms baseline — 2.57× slower.

Why it regressed: the current implementation gates attention by *masking* unselected
positions with `-inf`, then running a standard softmax over the full `(Lq, Lk)` matrix.
The matmul shape is unchanged; we just zero-weight some entries. Block scoring +
mask construction adds 10-30% overhead on top of a hand-rolled eager einsum that
already loses to the fused SDPA kernel.

Separate diagnostic: `debug_exact` mode (our eager path, no gating) takes 414 ms at 2k.
So naive eager already gives up ~2× vs SDPA; adaptive then adds another 30% on top.

Takeaway: before claiming any latency improvement from adaptivity, implement the
gather-based fast path — materialise K/V for only the selected blocks and call SDPA
on the reduced tensor. The mask-based implementation stays as a correctness oracle
for the gather path.

Also flagged: without `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, the stock SDPA
falls back to a slow math path (1313 ms at 2k). Every earlier number against the
non-experimental baseline was inflated by ~6×. Fixed in `tb`.

Artifacts:
- `results/raw/adaptive_attention_*_attn_block64_topk8_residual_*.json`
- `results/raw/adaptive_attention_*_attn_block64_topk4_*.json`
- `results/raw/adaptive_attention_*_attn_debug_exact_*.json`
- Old handicapped baseline archived as `results/raw/baseline_noexp_*`.

## 2026-04-19 — Per-query gather is 12× slower than the mask oracle on ROCm

Built `gather.py`: per-query top-k block selection, materialise K_sel and V_sel
via expand→gather of shape `(B, H_q, Lq, k·b, D)`, run the softmax on the
reduced set. Bit-equivalent to the mask oracle (tests pass).

Result at 2k: PPL 11.09 (Q=1.00) but prefill 6393 ms vs mask's 533 ms. ~12× slower.

Why: `torch.gather` with scattered reads along a large Lq axis doesn't coalesce
on ROCm, and there's no way to fold the result into a fused SDPA call because
each query has a different key set. The 4.3 GB of non-contiguous reads per
layer dominates. Nothing about the algorithm is wrong — the hardware just
doesn't like this pattern.

Response: built `gather_shared.py` (below). Keeping `gather.py` as the
correctness oracle for anything more elaborate we try next.

## 2026-04-19 — gather_shared wins at long context, collapses at very low top_k

Setup: `gather_shared.py` uses a single block selection per head (from the
mean query) shared across all queries in a forward pass. Gather shrinks to
`(B, H_q, k·b, D)` — no Lq axis — which lets the post-gather attention use
a plain fused `scaled_dot_product_attention` call.

Latency (prefill, Qwen3-0.6B, residual_refine, block=64, top_k=8):

| ctx  | baseline SDPA | gather_shared | speedup |
|------|---------------|---------------|---------|
|  512 |   58–299 ms   |    81 ms      | below break-even |
|  2k  |  212–857 ms   |   262 ms      | ~break-even       |
|  4k  |  539–1667 ms  |   564 ms      | ~break-even       |
|  8k  | 1592–3436 ms  |  1203 ms      | 1.3× – 2.9× faster |
| 16k  | 5610–9449 ms  |  3730 ms      | 1.5× – 2.5× faster |

Quality across runs: PPL 11.11 vs baseline 11.10 (Q=0.9985). Essentially identical.

Wide baseline ranges reflect large run-to-run variance — kernel cache state,
thermal, fragmentation — that was not investigated. Both columns are honest
medians from the same script; we just ran the baseline twice and saw different
numbers. This is a measurement hygiene problem to fix before the paper.

The crossover lives near 4k, and the advantage grows roughly linearly with
context length, consistent with the analytical prediction (we save the
Lk-dominated part of the attention matmul; the per-query ops are still O(Lq)).

### But: quality cliff at very low top_k

`gather_shared` with `top_k=4` (3 % of blocks exact at 16k) gave **PPL
13,865** on the calibration text — total collapse. The cause is that the
selection is *shared* across all queries in a forward pass. When top_k is
small and the mean-query picks blocks biased toward late positions, early
queries are left with no selected real tokens in their causal window. The
residual summaries are supposed to rescue them, but with only a handful of
blocks actually getting to "see" their summaries, the distribution is
too crushed to support generation.

Practical implication: `gather_shared` has a quality floor that's a function
of how close to per-query the shared selection can get. On Qwen3-0.6B,
block=64 with top_k≥8 (≥6 % of blocks exact) held quality; top_k=4 did not.
The right follow-up is *window-shared* selection — every W consecutive
queries share one selection — which approximates per-query much better
while keeping the gather dense enough for SDPA.

Artifacts:
- `results/raw/adaptive_attention_*_attn_block64_topk8_residual_gather_shared_*.json`
- `results/raw/adaptive_attention_*_attn_block64_topk4_residual_gather_shared_*.json` (quality-cliff artifact)

## 2026-04-19 — Bound-screen selector (A2) rescues the quality floor at low top_k

Setup: per block r, compute center c_r (mean of keys) and radius ρ_r
(max L2 distance of any key to the center). For a query q, the certified
upper bound on any key's attention logit within block r is

    U_r(q) = q·c_r + ||q||·ρ_r .

`BoundScreenSelector` picks top-k by `U_r` (with configurable margin δ on the
radius term; δ=0 ≡ FixedTopK, δ=1 uses the full Cauchy-Schwarz bound).
Correctness is enforced by `tests/test_attention_bound_screen.py`: for any
random (q, K), `U_r ≥ max_j q·k_j` holds exactly at every (query, block).

Result at residual_refine, block=64, impl=gather_shared:

| top_k | selector      | PPL     | Q      | prefill 8k | note                          |
|-------|---------------|---------|--------|------------|-------------------------------|
|   8   | fixed_topk    | 11.113  | 0.999  |  1203 ms   | earlier Pareto point           |
|   8   | bound_screen  | 11.086  | 1.001  |  2437 ms   | same quality, 2× slower       |
|   6   | bound_screen  | 11.086  | 1.001  |  1226 ms   | **same quality as k=8, -25 % blocks** |
|   4   | bound_screen  | 217.8   | 0.051  |  2438 ms   | degraded but not collapsed    |
|   4   | fixed_topk    | 13,865  | 0.001  |  2396 ms   | collapsed                     |

The headline: at block=64, the certified bound **lets us drop from top_k=8 to
top_k=6 without measurable quality loss on this calibration text** (11.086 vs
11.113 — bound_screen matches or slightly exceeds fixed_topk at 25 % fewer
exact blocks). Q values above baseline here should be read as "no measurable
loss," not "improvement" — they sit in the jitter range of a single-passage
perplexity metric, not a hardened eval.

At the aggressive top_k=4 setting that collapses FixedTopK (PPL 13,865), the
bound keeps PPL at 218 — two orders of magnitude better *at the same block
budget*, but still off the 97 % quality floor.

The right mechanistic line: the bound helps most in the regime where coarse
ranking becomes brittle. FixedTopK ranks by q·c_r alone; BoundScreen keeps
blocks whose *worst-case* exact logit could still matter. That's exactly
what protects the "hidden-important" blocks FixedTopK drops first when the
coarse summaries stop being informative.

### Pareto sweep (block=64, residual_refine, calibration PPL)

After running fixed_topk at the same k values (k=5, 6), the picture is
sharper than the first-pass read:

| top_k | fixed_topk PPL | bound_screen PPL |
|-------|----------------|------------------|
|   4   | 13,865         |   217.8          |
|   5   |  5,689         |    33.04         |
|   6   |   11.113       |    11.105        |
|   8   |   11.113       |    11.086        |

**Correction to an earlier overread:** I initially wrote that bound_screen
"lets us drop from k=8 to k=6 without quality loss." That's misleading —
fixed_topk *also* works at k=6 on this calibration metric. Both selectors
cross the quality floor somewhere between k=5 and k=6.

The honest bound_screen claim on this data:
- **Same quality floor** (both reach Q≈1.00 at k=6).
- **Graceful degradation past the cliff**: at k=5 the bound keeps PPL 33 vs
  fixed_topk's 5,689. At k=4, 217.8 vs 13,865. Two to three orders of
  magnitude better in the underprovisioned regime.

The "smaller-budget-at-same-quality" framing needs different data to hold up
— a workload where the coarse ranking is more brittle (longer context, harder
retrieval task, different block sizes) or with a tighter quality floor. The
graceful-degradation framing already holds here.

### Latency (bench_matrix v1, same-process, 10 prefill iters, std ≤ 11 ms)

| ctx  | baseline SDPA | fixed_topk k=8 | bound_screen k=6 |
|------|---------------|----------------|------------------|
| 1k   | 106 ± 2.3 ms  | 136 ± 2.1 ms   | 131 ± 2.6 ms     |
| 2k   | 217 ± 3.8 ms  | 267 ± 2.2 ms   | 245 ± 1.3 ms     |
| 4k   | 559 ± 4.3 ms  | 577 ± 3.1 ms   | **523 ± 5.5 ms**   |
| 8k   | 1596 ± 11 ms  | 1190 ± 5 ms    | **1154 ± 3 ms**    |
| 16k  | ~5610 ms*     | ~3730 ms*      | **~2461 ms***      |

\* from earlier runs in a different process — not same-session. The
cross-process ranges that were the excuse for the "1.5×–2.5×" wide error
bars on earlier tables collapse inside one process: std is 1–6 ms.

Crossover point for bound_screen k=6 vs baseline is between 2k and 4k.
At 4k bound_screen is already faster than baseline; at 8k it beats both
baseline (1.38×) and fixed_topk k=8 (1.03×). So on this hardware
bound_screen k=6 is a strict improvement over fixed_topk k=8 on both
axes (quality, latency) at context ≥ 4k.

### Tightness (bound never under-estimates, and restructures selection)

| ctx  | k  | slack_rel med | slack_abs med | set-overlap vs fixed_topk | violations |
|------|----|---------------|---------------|---------------------------|------------|
| 2k   | 4  | 9.86          | 44.5          | 0.548                     | 0          |
| 2k   | 6  | 9.86          | 44.5          | 0.639                     | 0          |
| 2k   | 8  | 9.86          | 44.5          | 0.708                     | 0          |
| 8k   | 6  | 11.43         | 43.6          | 0.449                     | 0          |

- **Zero violations across every configuration.** The certificate holds.
- The bound is loose (U_r ≈ 10× max_actual by median) but the ranking it
  induces still disagrees with coarse ranking a lot: at 8k with k=6, only
  45 % of bound's picks overlap with fixed_topk's. So the bound is
  actively restructuring the selection, not just agreeing.

Artifacts:
- `results/raw/adaptive_attention_*_attn_fixed_topk{5,6}_residual_gather_shared_*.json`
- `results/raw/adaptive_attention_*_attn_bound_screen_topk{4,5,6,8}_residual_*.json`
- `results/bench/bench_matrix_v1.json` — latency table with std
- `results/tightness/*.json` — per-layer slack distributions

## 2026-04-19 — Scaling confirms the story at long context

Extended bench (`results/bench/bench_matrix_long_ctx.json`, `bench_matrix_64k.json`):

| ctx | baseline SDPA | fixed_topk k=6 | fixed_topk k=8 | bound_screen k=6 | bound vs baseline |
|-----|---------------|----------------|----------------|-------------------|--------------------|
| 16k | 5514.7 ± 33.4 ms | 2501.6 ± 13.1 | 2628.8 ± 7.8 | **2458.4 ± 9.1 ms** | **2.24×** |
| 32k | 28 703 ± 564 ms | 5517.6 ± 10.4 | 6027.4 ± 11.6 | **5415.4 ± 14.6 ms** | **5.30×** |
| 64k | 179 763 ± 16 711 ms* | 13 825.7 ± 663 | — | **13 042.2 ± 94.5 ms** | **13.78×*** |

\*64k exceeds Qwen3's `max_position_embeddings=40960`; baseline quality past 40k is undefined, so the 13.78× figure is latency-only. The 32k point is the strongest fully-defensible speedup.

Growth is approximately O(Lk) for adaptive (driven by block scoring + gather) vs O(Lk²) for baseline SDPA even with AOTriton. Quality identical at all lengths (PPL 11.08–11.11 vs baseline 11.10).

## 2026-04-19 — Passkey retrieval exposes a block-summary-dilution failure, and it's orthogonal to selector math

Added a passkey/needle evaluator (`eval/retrieval.py`, `scripts/run_passkey_matrix.py`): inject a random 5-digit code "MAGIC=XXXXX" at depth d ∈ {0.1, 0.5, 0.9} into a long filler passage; measure teacher-forced correctness of the completion "what is the magic number? A: XXXXX" across 5 seeds. Baseline hits 100 % at every (ctx, depth) tested.

Every variant of the current adaptive method — fixed_topk or bound_screen, coarse_replace or residual_refine, top_k 4–8, at block_size 64 — scores 0 % on passkey at almost every depth/context. The one blip (`bound_screen k=8 at d=0.9, 2k`: 60 %) is too brittle to build on.

**The failure is not the selector's aggregation statistic.**

The user's upgrade ladder suggested max-over-queries as the first fix. Implemented `gather_windowed.py`: per-window selection with configurable `query_score ∈ {max, mean, mean_plus_max}`. Outcome at b=64:

| method | 2k/0.1 | 2k/0.5 | 2k/0.9 | 4k/0.5 | 4k/0.9 |
|--------|--------|--------|--------|--------|--------|
| baseline | 100 % | 100 % | 100 % | 100 % | 100 % |
| bound shared_mean (b=64) | 0 | 0 | 0 | 0 | 0 |
| bound windowed_mean (b=64) | 0 | 0 | 80 % | 0 | 0 |
| bound windowed_max (b=64) | 0 | 0 | 0 | 20 % | 0 |
| bound windowed_max δ=5 (b=64) | 0 | 0 | 0 | 0 | 0 |

Windowing helps a little (some non-zero values show up at easy depths), but the change from mean → max doesn't systematically move the numbers. Even pushing the radius weight to δ=5 doesn't rescue retrieval at b=64.

**The real failure is block-summary dilution.**

Hypothesis: at b=64, the needle is 1 token in a 64-token block. The block's mean summary is dominated by filler; its radius includes the needle contribution but that's still a single-direction perturbation averaged against 63 filler directions. The selector can't distinguish the needle block from other filler blocks.

Smaller blocks, same gather budget (so comparable attention cost inside each window):

| method | 2k/0.1 | 2k/0.5 | 2k/0.9 | 4k/0.1 | 4k/0.5 |
|--------|--------|--------|--------|--------|--------|
| bound b=32 k=12 windowed_max | 0 | 0 | 0 | 0 | 0 |
| **bound b=16 k=24 windowed_max** | **100 %** | 0 | 60 % | **100 %** | 20 % |
| bound b=16 k=24 coarse_replace | **100 %** | 0 | 40 % | **100 %** | 0 |
| bound b=8 k=48 windowed_max | 0 | 0 | 0 | 0 | 0 |

- **b=16 is a step-change** over b=64 at easy depths (d=0.1 gives 100 % at 2k and 4k).
- **b=8 is worse than b=16** — the spectrum isn't monotone. Likely: too many near-identical fragments; coarse scores become noisy and the needle's block no longer stands out against its neighbors.
- **Mid-depth failure (d=0.5) persists** even at b=16. The needle's block isn't in the top-k when it's far from the question window's causal focus.
- **Residual vs coarse_replace doesn't matter for retrieval** at b=16.

Latency of the retrieval-capable configuration:

| ctx | baseline | bound b=64 k=6 (best-latency) | bound b=16 k=24 (best-retrieval) |
|-----|----------|-------------------------------|-----------------------------------|
| 2k  | 217 ms | 245 ms | 829 ms (3.8× slower) |
| 4k  | 540 ms | 523 ms | 1896 ms (3.5× slower) |
| 8k  | 1596 ms | 1154 ms | 4572 ms (2.9× slower) |

So at b=16 k=24 we lose the latency story in the 2–8k range because residual summary count scales as Lk/b. At longer contexts the gap narrows, but we don't have numbers past 8k at b=16 yet.

**What this means for the paper as currently scoped:**
- The PPL + latency Pareto result at b=64 still stands. bound_screen k=6 is a strict improvement over fixed_topk k=8 at ctx ≥ 4k; speedup grows from 1.07× at 4k to 5.30× at 32k.
- The retrieval result does NOT stand. The current implementation cannot recover sparse single-token information from a 64-token block summary, regardless of selector tweaks.
- The most likely algorithmic fix is **multi-prototype per block**: keep the gather-friendly b=64 but store M>1 (center, radius) pairs per block so the summary isn't forced through a mean. That's Priority 4/5 on the user's roadmap.
- Until multi-prototype lands, the honest paper framing is calibration-PPL + latency only. Do not claim retrieval performance.

Artifacts:
- `results/bench/bench_matrix_long_ctx.json`, `bench_matrix_64k.json`
- `results/retrieval/passkey_small.json` — first passkey run, shared-mean b=64, 0 % everywhere
- `results/retrieval/passkey_windowed.json` — windowed + max-aggregation ablations at b=64
- `results/retrieval/passkey_block_size_sweep.json` — b=64/32/16 at windowed_max
- `results/retrieval/passkey_fine_grained.json` — b=16/8 with deeper depth sweep
- `results/retrieval/passkey_delta_and_coarse.json` — high-δ at b=64 and b=16 coarse_replace

## 2026-04-19 — Multi-prototype per block does not restore retrieval at b=64

Hypothesis from the previous entry: block-summary dilution at b=64 kills single-token retrieval, and keeping b=64 for gather while adding M>1 sub-block prototypes should recover summary resolution without the b=16 latency cost.

Implementation: `compute_block_multiproto` (fixed equal-split sub-blocks, no clustering), per-prototype bound `U_{r,m}(q) = q·c_{r,m} + ||q||·ρ_{r,m}`, block-level bound `max_m U_{r,m}`. Gather + residual summaries stay at block level (M×Nb virtual keys would wash the latency). Certified-bound test holds for every M in {1,2,4,8}.

Passkey (b=64, top_k=6, residual_refine, windowed max, 5 trials × 5 depths × 2 contexts = 50 cells per config):

| M | passkey cells > 0 | best cell | PPL (8k eval) | prefill 2k | prefill 8k |
|---|--------------------|-----------|--------------|-------------|-------------|
| 1 | 0 / 10 | 0 % | 11.082 |  503 ms |  2210 ms |
| 2 | 0 / 10 | 0 % | 11.114 |  693 ms |  3374 ms |
| 4 | 1 / 10 | 20 % @ 2k d=0.1 | 11.084 |  720 ms |  4624 ms |
| 8 | 0 / 10 | 0 % | — | — | — |
| b=16 k=24 M=1 (reference) | 6 / 10 | 100 % @ 2k/4k d=0.1 | 11.110 | 829 ms | 4572 ms |

Both thresholds from the go/no-go plan fail:
- **Retrieval not restored.** Only one out of 50 per-config cells at M=4 shows anything, and it's at the easiest depth. M=8 gives 0.
- **Latency regresses.** M=2 costs ~1.5× M=1 at 8k; M=4 costs ~2.1×. More prototypes = more scoring ops without reducing the selection granularity.

Mechanism for why this fails even though b=16 works: at b=64 M=4, each prototype does cover 16 tokens (same as b=16), so *per-prototype* summaries are equally discriminative. But selection still happens at the 64-token *block* level — 32 blocks at 2k vs. 128 at b=16. Taking `max_m` over 4 prototypes per block adds positive-side noise to every block's score, and the 64-token needle-containing block still has to outrank 31 others with 6 picks. At b=16 it's one of 128 with 24 picks — four times the selection budget at four times the granularity. Multi-prototype recovers summary resolution but not selection resolution, and the passkey needs both.

The cleaner fix is selection-at-prototype-level (gather and select in units of `b/M` instead of `b`), which collapses to "just use b=16." There is no free lunch here: if you want single-token sparse retrieval, you pay the Nb cost of finer blocks.

Paper consequence (commit to this — don't re-litigate):
- The calibration-PPL + long-context latency story stays. bound_screen k=6 at b=64 is a 1.38× speedup at 8k, 2.24× at 16k, 5.30× at 32k vs baseline SDPA, with no measurable PPL loss. That's defensible.
- The single-token passkey result becomes a *limitation*, not a contribution. Document it as a known failure mode of coarse-summary block-sparse attention, consistent with prior observations (Quest's use of element-wise min/max was specifically motivated by this).
- Do not build more variants of block-level selection to chase passkey. That's where the research swamp lives.

Artifacts:
- `results/retrieval/passkey_multiproto.json` — M ∈ {1,2,4,8} vs b=16 reference, five depths.
- `results/raw/adaptive_attention_*_attn_bound_mp{2,4}_topk6_*.json` — PPL + latency.

Latency of bound_screen is currently dominated by radius computation in fp32
(`diffs.to(torch.float32).norm(dim=-1).max(dim=-1)`). This is done at every
forward; for prefill it can be computed once per layer and cached for decode.
That's the obvious next optimization — if the fp32 radius compute is what
erases the "fewer blocks" win, caching radii recovers it.

Artifacts:
- `results/raw/adaptive_attention_*_attn_bound_screen_topk8_residual_*.json`
- `results/raw/adaptive_attention_*_attn_bound_screen_topk6_residual_*.json` ← current best Pareto candidate
- `results/raw/adaptive_attention_*_attn_bound_screen_topk4_residual_*.json`
