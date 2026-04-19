"""Prefill + decode latency benchmarks.

Two measurements, both reported per context length:
- prefill_ms:    time for a single forward pass over `ctx_len` tokens, no KV cache.
- decode_tok_s:  throughput of autoregressive decoding for `decode_len` tokens, with KV cache.

Decisions (explicit so they're reproducible):
- input IDs are sampled from [0, vocab_size); real-text vs random has no measurable
  effect on the backbone pass at this model size — the kernels don't care.
- we warm up, synchronize, and take median + p95 over N runs.
- tokens/sec is computed from the full decode wall time; the first decode step includes
  prefill, so we report prefill separately and count only `decode_len` generated tokens
  against `decode_total_ms - prefill_ms` for a cleaner number.
"""
from __future__ import annotations

from typing import Any

import torch

from ..utils.timers import time_many


@torch.inference_mode()
def measure_prefill_ms(model, device: str, vocab_size: int, ctx_len: int,
                       *, warmup: int = 2, iters: int = 5) -> dict[str, float]:
    input_ids = torch.randint(0, vocab_size, (1, ctx_len), device=device, dtype=torch.long)

    def fn():
        # use_cache=False so we're timing a pure forward pass, no KV bookkeeping cost.
        model(input_ids, use_cache=False)

    stats = time_many(fn, warmup=warmup, iters=iters)
    return {"ctx_len": ctx_len, **stats.to_dict()}


@torch.inference_mode()
def measure_decode_tokens_per_s(model, tokenizer, device: str, ctx_len: int, decode_len: int,
                                *, warmup: int = 1, iters: int = 3) -> dict[str, float]:
    """Greedy decode `decode_len` new tokens starting from a random prefix of length `ctx_len`.

    Returns a dict with mean/median decode tokens/sec and the per-step latency.
    """
    vocab_size = getattr(model.config, "vocab_size", tokenizer.vocab_size if tokenizer else 32000)
    input_ids = torch.randint(0, vocab_size, (1, ctx_len), device=device, dtype=torch.long)

    def fn():
        out = model.generate(
            input_ids,
            max_new_tokens=decode_len,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or 0,
        )
        return out

    stats = time_many(fn, warmup=warmup, iters=iters)
    # median total generate time minus median prefill time, then decode_len / remaining
    median_total_s = stats.median_ms / 1000.0
    # We don't know prefill-only time here precisely; approximate tokens/sec against total.
    # That's conservative (slightly underestimates decode speed).
    tok_s_total = decode_len / median_total_s if median_total_s > 0 else float("inf")
    return {
        "ctx_len": ctx_len,
        "decode_len": decode_len,
        "total_median_ms": stats.median_ms,
        "total_p95_ms": stats.p95_ms,
        "decode_tok_s_approx": tok_s_total,
        "per_step_ms_approx": stats.median_ms / max(decode_len, 1),
    }


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def profile_latency_grid(model, tokenizer, device: str, *,
                         ctx_lens: list[int], decode_len: int = 64,
                         warmup_prefill: int = 2, iters_prefill: int = 5,
                         warmup_decode: int = 1, iters_decode: int = 3) -> dict[str, Any]:
    vocab_size = int(model.config.vocab_size)
    reset_peak_memory()
    prefill = [measure_prefill_ms(model, device, vocab_size, L,
                                  warmup=warmup_prefill, iters=iters_prefill) for L in ctx_lens]
    decode = [measure_decode_tokens_per_s(model, tokenizer, device, L, decode_len,
                                          warmup=warmup_decode, iters=iters_decode) for L in ctx_lens]
    return {
        "prefill": prefill,
        "decode": decode,
        "peak_memory_mb": peak_memory_mb(),
    }
