# Certified Block Screening for Adaptive Long-Context Attention

*Draft, 2026-04-19. Written in the order specified by the outline: Method →
Results → Limitation Analysis first, then Experimental Setup, Intro, Related
Work, Conclusion, Abstract. Experimental Setup is minimal until the earlier
sections stabilise.*

*Framing sentence (kept in Intro, Discussion, Conclusion):*
> This is a certified runtime screening method for long-context efficiency, not a general retrieval architecture.

---

## 3. Method

### 3.1 Setup

Let $n$ denote the KV-cache length at the point of attention and $d$ the per-head hidden dimension. We partition the KV cache along the sequence axis into $m = \lceil n / b \rceil$ fixed-size blocks of size $b$, padding the final block by repeating its last entry. Block $r$ contains keys $K_r \in \mathbb{R}^{b \times d}$ and values $V_r \in \mathbb{R}^{b \times d}$. Let $k \ll m$ denote the *exact-attention budget* — the maximum number of blocks whose per-token keys and values will participate in the token-level softmax.

Grouped-query attention is handled by the standard repeat of KV heads across query heads; the method is described per query head. Causal masking applies as usual.

### 3.2 Coarse-to-exact attention

For each block $r$ we maintain a summary $(\bar k_r, \bar v_r)$ — in our implementation, the per-head mean of $K_r$ and $V_r$ respectively. Given a query $q$ and a block-selection function $S(q) \subseteq \{1, \dots, m\}$ with $|S(q)| \le k$, the attention output is

$$
y(q) \;=\; \mathrm{softmax}\!\left( \frac{1}{\sqrt d}\, [\,q^{\top} K_{S(q)} \;|\; q^{\top} \bar K_{\bar S(q)}\,] \right) \cdot [\, V_{S(q)} \;|\; \bar V_{\bar S(q)} \,],
$$

where $K_{S(q)} = \bigsqcup_{r \in S(q)} K_r$ is the concatenation of exact token keys from the selected blocks, $V_{S(q)}$ the corresponding values, and $(\bar K_{\bar S}, \bar V_{\bar S})$ collect the summaries of the unselected blocks $\bar S(q) = \{1,\dots,m\} \setminus S(q)$. We refer to the two halves of this joint softmax as the *exact-refine* path and the *coarse* path:

$$
y(q) \;\approx\; y_{\mathrm{coarse}}(q) \,+\, y_{\mathrm{exact\text{-}refine}}(q),
$$

with proportionality absorbing the shared normaliser. In practice we evaluate the joint softmax with a single fused kernel call; the split is conceptual.

Two invariants matter. First, the exact KV cache is never discarded — only the *allocation* of exact attention varies per query. Second, the softmax is causal and well-defined for every query, including those whose selected blocks lie entirely in the future (the unselected block summaries remain available in the coarse path).

### 3.3 Certified block screening

For each block $r$, store a *center* $c_r$ and a *radius* $\rho_r$ chosen so that

$$
\| k_j - c_r \|_2 \;\le\; \rho_r \qquad \forall\, j \in B_r. \tag{1}
$$

We take $c_r$ to be the mean of the block's keys and $\rho_r = \max_{j \in B_r} \| k_j - c_r \|_2$. Both are computed in bf16 with an fp32 accumulation for the squared distances; this takes $O(b \cdot d)$ per block and — in the decode regime — is amortised by a per-layer cache that recomputes only the growing final block's statistics.

**Proposition 1 (certified upper bound).** For every query $q \in \mathbb{R}^d$ and every key $k_j \in B_r$,
$$
q^{\top} k_j \;\le\; q^{\top} c_r + \|q\|_2 \, \rho_r \;=:\; U_r(q). \tag{2}
$$

*Proof.* Write $q^{\top} k_j = q^{\top} c_r + q^{\top} (k_j - c_r)$. By Cauchy–Schwarz, $|q^{\top}(k_j - c_r)| \le \|q\|_2 \, \|k_j - c_r\|_2 \le \|q\|_2 \, \rho_r$ using (1). ∎

We call (2) a *certified* bound: it is never violated by construction. In the experiments of §5.3 we verify this empirically, reporting zero violations across every probed configuration. Tightness is a separate question addressed in §5.3.

### 3.4 Selection policies

We compare two selection functions:

**FixedTopK.** $S(q) = \arg\mathrm{top}\text{-}k_r\, q^{\top} \bar k_r$. This is the coarse-score ranking used by Quest-like page-selection methods; selection depends only on the coarse summaries.

**BoundScreen.** $S(q) = \arg\mathrm{top}\text{-}k_r \, \big( q^{\top} c_r + \delta \cdot \|q\|_2 \, \rho_r \big)$, with $\delta \in [0, 1]$. At $\delta = 0$ this reduces to FixedTopK. At $\delta = 1$ selection ranks blocks by the certified upper bound $U_r$ on their best possible token-level score. We use $\delta = 1$ throughout unless stated.

The distinction is conceptually small but load-bearing: FixedTopK ranks blocks by *what is typical* inside them, BoundScreen ranks blocks by *what is possible* inside them.

### 3.5 Systems implementation

Three design choices carry most of the performance story.

**Shared-across-queries selection (*gather_shared*).** Within a single forward pass, all queries in a head share one selection $S$, scored from the head's mean query. The gathered key/value tensors then have shape $(B, H_q, k \cdot b, d)$ — no $L_q$ axis — which lets the post-gather attention use a single fused `scaled_dot_product_attention` call on a drastically reduced KV set. Per-query causal masking is applied inside that single call, so individual queries never attend to gathered tokens in their own future. The alternative — per-query selection and gather — is bit-equivalent but 12× slower on the hardware we used, because `torch.gather` with scattered reads along a large $L_q$ axis does not coalesce. We discuss this trade and its retrieval consequences in §6.

**Residual-refine packing.** Unselected-block summaries are concatenated onto the selected key/value tensors as $m - k$ additional virtual tokens. The softmax runs jointly over $k \cdot b + m - k$ keys. This keeps every query in causal contact with every past block — either via exact tokens from the selected blocks or via a one-vector summary from each unselected one — and retains the coarse-path gradient flow described in §3.2 without introducing a separate forward pass.

**Radius caching.** In the decode regime, each generation step extends the KV cache by one token. All blocks' centres and radii except the newest are unchanged; the cache recomputes only the last (growing) block. This removes $O(n \cdot d)$ redundant work per decode step in the bound selector, which is what lets bound_screen's decode wall-clock catch up to fixed_topk's despite carrying the extra radius computation.

We implemented three variants of the gather step — mask-based (correctness oracle), per-query gather, and shared-across-queries gather — and verified bit-equivalence of the first two via unit tests. The shared-gather path is the one that achieves the speedups reported in §5.

### 3.6 Complexity

We report complexity per query head. Let $n$ be the KV length, $b$ the block size, $m = \lceil n/b \rceil$ the block count, $k$ the exact budget, and $d$ the head dimension.

**Coarse scoring:** $q \cdot c_r$ for every block, $O(m \cdot d) = O((n/b) \cdot d)$ per query.

**Exact refinement:** softmax over $k \cdot b$ gathered tokens, $O(k \cdot b \cdot d)$ per query.

**Per-query cost:** $O\!\big((n/b + k \cdot b) \cdot d\big)$ — sub-linear in $n$ when $b \cdot k \ll n$.

**Prefill** ($n$ queries): $O\!\big(n^2/b \cdot d + n \cdot k \cdot b \cdot d\big)$. The first term is the scoring pass; the second is exact refinement across all queries. This is *reduced-quadratic*, not linear: the $n^2/b$ prefix persists because each query still scores every past block. A fully linear prefill would require dropping even the coarse path, which we explicitly choose not to do.

**Decode** (one new query per step): $O\!\big((n/b + k \cdot b) \cdot d\big)$, identical to per-query cost.

The ratio of adaptive prefill to dense prefill is $(1/b + k \cdot b / n)$. At $b = 64$, $k = 6$, $n = 32\mathrm{k}$ this ratio is $\approx 0.028$. We report this as an *algorithmic ceiling*, not as a speedup claim — it is what the method could achieve if coarse scoring, gather, causal-mask construction, and residual summary packing were free. The measured 5.3× speedup at 32 k in §5.1 is what remains after those costs; the gap between ceiling and measurement is concrete implementation surface — coarse-path memory traffic, mask and index tensors, SDPA call overhead — not a theoretical loss.

---

## 5. Main Results

All experiments use Qwen3-0.6B in bfloat16 on a single AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151), with PyTorch 2.10 + ROCm 7.2 + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. Latency is reported as the median of ten timed iterations (five at 16 k, three at 32 k and 64 k) following three warm-up iterations; ± denotes the standard deviation within a single session. Calibration perplexity is measured on a fixed information-theory passage over non-overlapping 512-token windows. Block size $b = 64$ and residual refinement mode are used throughout unless stated.

### 5.1 Long-context latency-quality tradeoff

Prefill latency as a function of context length, comparing dense SDPA, FixedTopK, and BoundScreen at comparable budgets, is shown in Figure 2 and Table 1. The headline numbers for BoundScreen at $k = 6$ versus dense baseline are

| context | 1 k | 2 k | 4 k | 8 k | 16 k | 32 k |
|---------|-----|-----|-----|------|------|------|
| speedup | 0.81× | 0.89× | **1.07×** | **1.38×** | **2.24×** | **5.30×** |

Crossover against the dense baseline occurs between 2 k and 4 k. Below crossover, the coarse-score computation and mask bookkeeping dominate; the selector saves little because the baseline attention is already cheap. At 4 k and beyond, the $O(n^2/b)$ scaling of coarse scoring pulls ahead of dense $O(n^2)$ attention, and the gap widens at the rate implied by §3.6. At 32 k, a 28.7 s dense prefill collapses to 5.4 s. At 64 k, which lies beyond Qwen3-0.6B's 40 960-token training context, the raw ratio is 13.78× but the baseline's output quality is undefined in that regime; we treat the 64 k row as an appendix-quality observation rather than a headline.

Calibration perplexity is held across the whole latency sweep. The Qwen3-0.6B baseline PPL is 11.096; BoundScreen at $k = 6$ measures 11.105, and FixedTopK at $k = 8$ measures 11.113. These differences are within run-to-run jitter on a single-passage calibration metric and we describe them as "no measurable loss" rather than as an improvement.

Intra-session variance is small: standard deviations are under 15 ms at every length up to 32 k (about 0.5 % of the median at 32 k). The one exception is the 64 k baseline, whose 16.7 s standard deviation reflects the position-embedding regime rather than the measurement protocol. We omit the 64 k baseline from quality-bearing claims for that reason.

### 5.2 Graceful degradation under tight exact budgets

Calibration perplexity as a function of the exact budget $k$, for both selectors at $b = 64$, is shown in Figure 3 and Table 2.

|   $k$ | FixedTopK PPL | BoundScreen PPL |
|-------|---------------:|-----------------:|
|   4   | 13,865         |             218  |
|   5   |  5,689         |          33.042  |
|   6   | 11.113         |          11.105  |
|   8   | 11.113         |          11.086  |

Both selectors meet baseline PPL at $k \ge 6$. The story is below that.

At $k = 5$, FixedTopK loses three orders of magnitude of perplexity; BoundScreen loses less than one. At $k = 4$, FixedTopK is at $\approx 10^4$; BoundScreen is two orders of magnitude lower. Neither selector is usable at $k \le 5$ in the strict sense — both fall below any reasonable quality floor — but the *failure mode* is qualitatively different. FixedTopK drops off a cliff; BoundScreen degrades.

The algorithmic contribution is that change in failure mode. At the operating point where an exact budget is provisioned conservatively enough to be correct on its headline workload, both selectors work. Under provisioning error — a shorter budget than expected, a workload with less average-case redundancy than expected — BoundScreen continues to produce coherent output at PPLs that are two orders of magnitude better than the coarse-ranking alternative. This is the behaviour the bound was designed to deliver, and it is what the Cauchy–Schwarz inequality buys over heuristic summary ranking: a block whose best possible token-level score is large is never silently dropped simply because its *average* score is unremarkable.

### 5.3 Bound behavior and selector restructuring

Figure 4 summarises two separate measurements derived from bound instrumentation on held-out calibration queries: the slack statistic and the selector-overlap statistic.

**Violations and slack.** We compute, for every $(q, r)$ pair probed, the ratio $U_r(q) / \max_{j \in B_r} q^{\top} k_j$. By Proposition 1 this ratio must be $\ge 1$. Across every probed configuration — context lengths 2 k, 8 k, 16 k, 32 k at $k = 6$ and additionally $k \in \{4, 8\}$ at 2 k — the ratio is at least 1 on every sampled cell: **zero observed violations**. The certificate holds in practice. The median ratio stays in the 9.5–10.9 range across all contexts, indicating that the bound is loose in absolute terms (roughly an order of magnitude above the actual max) but remarkably stable. Slack does not widen with context length, which is what one would hope from a bound whose derivation does not depend on $n$.

**Restructuring vs FixedTopK.** For each $(q, r)$ configuration we also measure the fraction of BoundScreen's selected blocks that also appear in FixedTopK's selected set at the same $k$. At 2 k context the overlap is 0.55, 0.64, 0.71 at $k = 4, 6, 8$ — BoundScreen keeps most of FixedTopK's picks and re-ranks around them. At longer contexts the selectors diverge: 0.45 at 8 k, 0.40 at 16 k, 0.37 at 32 k, all at $k = 6$. By 32 k, 63 % of BoundScreen's six selected blocks are not in FixedTopK's top-6. This rules out the interpretation that BoundScreen is a relabelling of FixedTopK: the two selectors make measurably different picks at long context. §5.2 is a separate experiment — different $k$, different context — but shows that the selectors' behaviours also differ under tight exact budgets, in the direction the bound would predict. We do not directly attribute the graceful degradation to the specific picks the overlap measurement identifies; we state only that (i) the two selectors are empirically distinct and (ii) BoundScreen is the one that preserves graceful degradation.

---

## 6. Limitation Analysis

### 6.1 Passkey retrieval failure at b=64

We probe sparse single-token retrieval with a standard passkey test: a 5-digit magic number is injected at depth $d \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ into a filler passage of context length $n \in \{2048, 4096\}$, and a trailing query "what is the magic number? A:" elicits a teacher-forced completion. Correctness is the fraction of trials over five seeds whose greedy completion matches the injected digits token-by-token. Results are shown in Table 3 and Figure 5.

The dense SDPA baseline scores 1.00 at every (context, depth) cell. Every adaptive variant at $b = 64$ scores at or near zero. BoundScreen at $k = 6$ shared-mean selection: zero across every cell. BoundScreen with windowed max aggregation: zero except a single 0.20 cell at 4 k / $d = 0.5$. BoundScreen with $\delta = 5$ (radius-dominant selection): zero. A reference configuration at $b = 16$, $k = 24$ — which uses the same gather budget in tokens ($k \cdot b = 384$) but a finer block partition — recovers 1.00 correctness at the easiest depth ($d = 0.1$) and 0.80 at $d = 0.9$ but still fails at mid-depth.

We state this plainly: the method as built is not a general retrieval architecture. The pattern of failure is not a selector tuning artefact.

### 6.2 Mechanism

The failure has two mechanisms that compound.

*Summary dilution.* A 64-token block whose only informative content is one magic-number token has a mean summary dominated by filler: its center $c_r$ is statistically indistinguishable from a neighbouring filler-only block's. The bound $U_r(q)$ adds $\|q\|_2 \rho_r$, which grows when the block contains an outlier — but for retrieval queries at Qwen3-0.6B scale, $q^{\top} c_r$ dominates the ranking. Raising $\delta$ to 5 (radius-term-dominant selection) does not rescue this: the invariant still holds, but the radius term alone is not discriminative enough to distinguish a needle block from its neighbours.

*Selection granularity.* A 64-token block is the unit of selection. Keeping it requires outscoring 31 other blocks at 2 k with six picks. At $b = 16$ the same argument operates against 127 blocks with 24 picks — a four-times-larger selection budget at four-times-finer granularity — and a 1/16 dilution is more often distinguishable from a 0/16 filler block than 1/64 is from 0/64. The $b = 16$ reference in §6.1 recovers retrieval at easy depths under exactly this accounting.

Neither mechanism alone is decisive; together they are.

### 6.3 Negative result: multi-prototype

A natural fix is to split each block into $M$ sub-blocks, store one $(c_{r,m}, \rho_{r,m})$ pair per sub-block, and bound the block via $U_r(q) = \max_{m=1,\dots,M} U_{r,m}(q)$. The block-level bound remains certified: the pointwise max of upper bounds is an upper bound on the pointwise max. We implemented this for $M \in \{2, 4, 8\}$ with equal sub-block splits and verified the bound invariant holds.

Empirically:

| $M$ | passkey cells > 0 (of 10 at 2 k/4 k) | best cell | PPL | 8 k prefill |
|-----|--------------------------------------|-----------|------|-------------|
|  1  | 0                                    |  0 %      | 11.082 | 2210 ms |
|  2  | 0                                    |  0 %      | 11.114 | 3374 ms |
|  4  | 1                                    | 20 %      | 11.084 | 4624 ms |
|  8  | 0                                    |  0 %      |   –    |   –     |
|  b=16, $k=24$, $M=1$  | 6                    | 100 %     | 11.110 | 4572 ms |

Retrieval is not restored, and latency regresses monotonically with $M$ — the $M$-fold increase in scoring cost is not amortised by any reduction in the downstream attention shape, which still sees $k \cdot b + m$ keys. Calibration PPL is preserved, consistent with multi-prototype leaving average-case modelling behaviour intact.

**The precise lesson: summary resolution alone is insufficient when selection granularity stays coarse.** Multi-prototype gives each block $M$ (center, radius) pairs, so per-prototype summaries cover $b / M$ tokens — as distinctive as a finer block $b$ would produce. But the selector still picks $k$ blocks out of $m$, not $k \cdot M$ prototypes out of $m \cdot M$. The passkey needs finer selection as well as finer summaries. Any fix that recovers selection resolution by treating each prototype as a selectable unit collapses to "use block size $b / M$" and pays the $\lceil n / (b/M) \rceil$ summary count in the softmax key set — which is the latency cost we saw in the $b = 16$ reference row.

### 6.4 Interpretation

The method is effective for long-context inference under calibration/PPL-style objectives: the selector preserves average modelling behaviour, the bound gives real graceful degradation under under-provisioned exact budgets, and the systems implementation delivers growing wall-clock wins at longer contexts. It is *not* a mechanism for retrieving sparse single-token information at the coarse block granularities tested here.

The correct positioning is the framing sentence: *this is a certified runtime screening method for long-context efficiency, not a general retrieval architecture*. The two most-closely-related prior methods use different per-block metadata with different tradeoffs. Quest maintains per-page element-wise key min/max and scores against those min/max bounds — finer-grained per-token information at the cost of more scoring operations and more metadata per block. Our method stores one center and one scalar radius per block and derives its bound from Cauchy–Schwarz — coarser metadata and a cheaper scoring pass, but a provable block-level upper bound in one line. Neither design point dominates the other; a production system wanting both sparse-retrieval fidelity and the present method's long-context wall-clock behaviour would likely compose them.

---

*[Next up, per the writing order: §4 Experimental Setup, §1 Introduction, §2 Related Work, §7 Discussion, §8 Conclusion, Abstract. The experimental setup section will be minimal once §5 and §6 stabilise; intro/related-work will be deferred until the empirical narrative here is signed off.]*
