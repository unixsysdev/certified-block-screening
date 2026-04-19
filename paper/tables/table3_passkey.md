# Table 3 — Passkey retrieval correctness (5 trials × depth × ctx)

Entries: fraction of trials whose greedy completion matches the injected passkey token-by-token.
Baseline is dense SDPA; all adaptive variants use residual_refine.

| method | ctx=2048 d=0.1 | ctx=2048 d=0.3 | ctx=2048 d=0.5 | ctx=2048 d=0.7 | ctx=2048 d=0.9 | ctx=4096 d=0.1 | ctx=4096 d=0.3 | ctx=4096 d=0.5 | ctx=4096 d=0.7 | ctx=4096 d=0.9 | ctx=8192 d=0.1 | ctx=8192 d=0.3 | ctx=8192 d=0.5 | ctx=8192 d=0.7 | ctx=8192 d=0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (dense SDPA) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bound_screen k=6 b=64 (headline config) | 0.00 | — | 0.00 | — | 0.00 | 0.00 | — | 0.00 | — | 0.00 | — | — | — | — | — |
| bound_screen k=6 b=64 windowed+max | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | — | — | — | — | — |
| multi-prototype M=4 b=64 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — | — | — | — | — |
| b=16 k=24 reference | 1.00 | 1.00 | 0.00 | 0.00 | 0.80 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
