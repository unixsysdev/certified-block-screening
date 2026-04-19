# Outline — Certified Block Screening for Adaptive Long-Context Attention

**Near-final, frozen 2026-04-19.** Title options:
1. Certified Block Screening for Adaptive Long-Context Attention  *(safest)*
2. Graceful Under-Budget Adaptive Attention via Certified Block Screening
3. Certified Runtime Screening for Long-Context Attention

**One-sentence paper claim.** We introduce a certified runtime block-screening method for adaptive long-context attention that preserves calibration/PPL behavior while improving long-context latency, and we show that its main strength is efficient long-context modeling, not sparse single-token retrieval.

**Core thesis.** Runtime efficiency with guarantees and graceful degradation. Not a new base architecture, not a general retrieval architecture, not universal sparse attention, not a replacement for local/linear methods.

**Framing sentence (appears in intro, discussion, conclusion).** *This is a certified runtime screening method for long-context efficiency, not a general retrieval architecture.*

---

## Abstract (7 sentences, fixed shape)

1. Problem: long-context dense attention is expensive; aggressive approximate selection often fails unpredictably when exact budgets are tight.
2. Method: partition KV into fixed blocks; coarse global path + exact attention on a small subset selected using a certified upper bound on each block's best possible q·k score.
3. Theory: selector uses block center–radius metadata; bound has zero observed violations.
4. Systems: on Qwen3-0.6B, bound-screen k=6 achieves 1.07×/1.38×/2.24×/5.30× prefill speedups at 4k/8k/16k/32k over dense baseline, with no measurable calibration-PPL loss.
5. Robustness: under tight budgets, certified screening degrades substantially more gracefully than naive fixed top-k.
6. Limitation: coarse block screening at 64-token granularity fails single-token passkey retrieval; multi-prototype block summaries do not rescue it at fixed coarse selection resolution.
7. Conclusion: certified block screening is a practical runtime method for long-context efficiency, not a general retrieval architecture.

## Writing order (do NOT start with intro)

1. Method
2. Main Results
3. Limitation Analysis
4. Experimental Setup
5. Intro
6. Related Work
7. Conclusion
8. Abstract

## Section plan

### 1. Introduction (3 ¶ + contribution bullets, last paragraph pre-announces the limitation)
### 2. Related Work (four buckets)
 - 2.1 Efficient long-context attention
 - 2.2 Query-aware selective KV retrieval (Quest, MInference, LazyLLM)
 - 2.3 Local/linear alternatives (sliding/local, GPT-OSS, Gated DeltaNet, Qwen3.5)
 - 2.4 What is distinct: runtime on existing dense models, block-level exact selection, certified bound, violation-rate evaluation, under-budget degradation analysis
### 3. Method (tight; one proposition, one proof line)
 - 3.1 Setup: n, b, m = n/b, k
 - 3.2 Coarse-to-exact attention: Y = Y_coarse + Y_exact-refine
 - 3.3 Certified block screening: (c_r, ρ_r), U_r(q) = q·c_r + ||q||·ρ_r, Cauchy–Schwarz
 - 3.4 Selection policies: fixed_topk, bound_screen
 - 3.5 Systems: gather_shared, residual_refine, radius caching, per-query gather rejected
 - 3.6 Complexity: prefill O(n²/b · d + n·k·b·d), decode O((n/b + k·b)·d) — reduced-quadratic, not linear
### 4. Experimental Setup (brief; protocol hygiene visible)
### 5. Main Results
 - 5.1 Long-context latency-quality tradeoff (Table 1, Fig 2) — 1.07×/1.38×/2.24×/5.30× at 4k/8k/16k/32k
 - 5.2 Graceful degradation under tight budgets (Table 2, Fig 3) — fixed-topk k=4 collapses to 13,865; bound-screen k=4 holds at 218
 - 5.3 Bound behavior and selector restructuring (Fig 4) — 0 violations, slack_rel median 9.5–10.9, overlap with fixed_topk drops 0.64 → 0.37 with ctx
### 6. Limitation Analysis
 - 6.1 Passkey failure at b=64 (Table 3, Fig 5) — baseline 100% vs adaptive ≤10%
 - 6.2 Mechanism: 1/64-token summary dilution; selector tuning doesn't help
 - 6.3 Negative result: multi-prototype M∈{2,4,8} does not rescue retrieval and regresses latency
 - 6.4 Interpretation: effective for efficient long-context modeling under calibration/PPL objectives; insufficient for sparse single-token retrieval at this granularity
### 7. Discussion (where useful / where not / future directions — no FFN digression)
### 8. Conclusion (one paragraph)

## Figure plan

- Fig 1: method diagram — KV → blocks → summaries (c_r, ρ_r) → selector → coarse path + exact refine.
- Fig 2: prefill latency vs ctx (`paper/figures/fig2_latency_vs_ctx.png`).
- Fig 3: PPL vs exact budget k (`paper/figures/fig3_ppl_vs_k.png`).
- Fig 4: tightness + overlap scaling with ctx (`paper/figures/fig4_tightness_and_overlap.png`).
- Fig 5: passkey limitation (`paper/figures/fig5_passkey.png`).

## Table plan

- Table 1: latency vs ctx (`paper/tables/table1_latency_vs_ctx.md`).
- Table 2: PPL vs k (`paper/tables/table2_ppl_vs_k.md`).
- Table 3: passkey limitation (`paper/tables/table3_passkey.md`).

## Writing rules

**Use:** "certified screening", "graceful degradation", "runtime block screening", "long-context efficiency", "zero observed violations", "not a general retrieval architecture".

**Do not use:** "same PPL to 4 decimals", "improves intelligence", "solves retrieval", "beats all sparse attention", "novel efficient transformer architecture".

## Do-not-open boxes

- Main-paper retrieval claim beyond §6.
- Further multi-prototype / orthogonal-basis / learned-gate selectors.
- FFN compression in the body (appendix note at most).
- "Full optimization stack" framing.
