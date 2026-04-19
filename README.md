# Certified Block Screening for Adaptive Long-Context Attention

Runtime adaptive-attention method with a provable per-block upper bound.
Implementation, experiments, and paper source. Qwen3-0.6B target model,
ROCm / CUDA compatible.

> **Framing sentence.** This is a certified runtime screening method for
> long-context efficiency, not a general retrieval architecture.

## Headline results (Qwen3-0.6B, bf16, AOTriton SDPA)

| context | dense baseline | BoundScreen k=6 | speedup |
|---------|---------------:|----------------:|--------:|
| 4 k     |   559 ms       |   520 ms        | **1.07×** |
| 8 k     |  1596 ms       |  1154 ms        | **1.38×** |
| 16 k    |  5515 ms       |  2458 ms        | **2.24×** |
| 32 k    | 28 703 ms      |  5415 ms        | **5.30×** |

Calibration perplexity preserved within run-to-run jitter
(baseline 11.096, BoundScreen k=6 11.105). Zero observed bound
violations across every probed configuration. Graceful degradation
under tight budgets: at `k=5`, fixed top-k collapses to PPL 5,689;
bound screening holds at 33. Full tables and figures live in
[`paper/`](paper/).

Known limitation: coarse block screening at `b=64` does not recover
single-token sparse retrieval (e.g. passkey). Multi-prototype extensions
do not rescue it; selection granularity, not summary granularity, is the
binding constraint. See `paper/main.tex` §6 for the full negative-result
analysis.

## Repo layout

```
src/adaptive_llm_speed/      # Python package
  methods/
    ffn_lowrank/             # (appendix / future work)
    adaptive_attention/      # method core: layer.py, gather_shared.py,
                             # gather_windowed.py, bounds.py, selectors.py
  models/loaders.py          # HF loader + torchvision-stub shim
  eval/                      # perplexity, latency, retrieval, tightness
configs/                     # YAML configs, one per method variant
scripts/                     # profile, run_eval, bench_matrix,
                             # measure_bound_tightness, make_paper_artifacts,
                             # run_passkey_matrix
results/                     # raw JSON artefacts (bench, tightness, retrieval)
paper/                       # main.tex, references.bib, figures, tables, data, outline
tests/                       # unit tests, mostly CPU-only
tb                           # wrapper that runs commands in the rocm-7.2 toolbox
```

## Reproducing

Everything runs inside a ROCm / CUDA toolbox via the `./tb` wrapper
(see `tb` at repo root).

Baseline profile:
```
./tb python3 scripts/profile_baseline.py \
  --config configs/base/model_qwen3_0p6b.yaml \
  --ctx-lens 1024 2048 4096 8192 16384 32768
```

Bound-screen evaluation:
```
./tb python3 scripts/run_eval.py \
  --model configs/base/model_qwen3_0p6b.yaml \
  --method configs/attention/attn_bound_screen_topk6_residual.yaml \
  --baseline-result results/raw/baseline_model_qwen3_0p6b_d7cc62251aa8.json \
  --ctx-lens 2048 8192 16384
```

Tightness probe:
```
./tb python3 scripts/measure_bound_tightness.py \
  --model configs/base/model_qwen3_0p6b.yaml \
  --block-size 64 --top-k 6 --delta 1.0 --ctx-len 8192
```

Regenerate every paper table and figure from the JSON artefacts:
```
./tb python3 scripts/make_paper_artifacts.py
```

## Build the paper

The paper is authored in `paper/main.tex` with references in
`paper/references.bib`. A GitHub Actions workflow builds `main.pdf` on
every push to `main` (see `.github/workflows/paper.yml`).

Locally:
```
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Tests

27+ unit tests cover the factorisation primitives, summary/selector
correctness, bound invariants, gather impl equivalences, and
multi-prototype certification.

```
./tb pytest -q tests/
```

## Hardware notes

Primary target: AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151) with ROCm 7.2.
The method is hardware-agnostic; CUDA targets work with a standard
`torch.cuda` install. See `paper/main.tex` §4 for the exact software stack.
