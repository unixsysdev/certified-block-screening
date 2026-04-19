"""Passkey / needle-in-haystack retrieval evaluation.

Given a long passage of filler text with a short "magic number" injected at a
chosen depth, measure whether the model can retrieve it. We use a teacher-forced
log-probability instead of generation so the result doesn't depend on the model
being instruction-tuned.

Scoring per trial:
    prompt    = "<preamble><filler_prefix>MAGIC=<key>.<filler_suffix>Q: what is the magic number? A: "
    completion= "<key>"
    score     = sum over completion tokens of log P(token_i | prompt, token_<i>)
    correct   = greedy argmax over completion tokens matches `<key>` token-by-token

Reported:
    avg log-prob per token
    argmax-correct fraction
    broken out by depth (0 = near start, 1 = near end)

This is the sparse-far-dependency workload the research plan's A2 selector is
supposed to help on. A failure mode of aggressive block-sparse attention is
dropping the block that happens to contain the needle — we measure exactly that.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch


_FILLER_LINE = (
    "The grass is green and the sky is blue. Researchers often walk to the canteen for lunch. "
    "The weather forecast says it may rain tomorrow, but probably not. Cats like to nap in the sun. "
    "An old proverb claims that a rolling stone gathers no moss, though nobody really checked. "
)


@dataclass
class PasskeyResult:
    ctx_len: int
    depth: float
    key: str
    logprob_sum: float
    logprob_per_token: float
    correct: bool
    completion_tokens: int


def _build_prompt(tokenizer, ctx_tokens: int, depth: float, key: str) -> tuple[str, str]:
    """Return (prompt, completion) strings such that tokenise(prompt+completion) ≈ ctx_tokens."""
    q_marker = f"\n\nMAGIC={key}.\n\n"
    opening = "Read the following passage carefully. A short magic number is hidden inside.\n"
    trailer = "\nQ: What is the magic number that was hidden above? A: "
    # Tokenise fixed pieces once.
    tok_q = tokenizer(q_marker, add_special_tokens=False)["input_ids"]
    tok_op = tokenizer(opening, add_special_tokens=False)["input_ids"]
    tok_tr = tokenizer(trailer, add_special_tokens=False)["input_ids"]
    reserved = len(tok_op) + len(tok_q) + len(tok_tr) + 8  # completion slack
    filler_budget = max(16, ctx_tokens - reserved)

    # Build filler up to that many tokens.
    filler = ""
    while True:
        filler += _FILLER_LINE
        n = len(tokenizer(filler, add_special_tokens=False)["input_ids"])
        if n >= filler_budget:
            break
    # Trim to exact budget.
    toks = tokenizer(filler, add_special_tokens=False)["input_ids"][:filler_budget]
    filler = tokenizer.decode(toks)

    # Split filler at `depth` fraction.
    cut = int(len(toks) * depth)
    prefix_toks = toks[:cut]
    suffix_toks = toks[cut:]
    prefix = tokenizer.decode(prefix_toks)
    suffix = tokenizer.decode(suffix_toks)

    prompt = opening + prefix + q_marker + suffix + trailer
    completion = key
    return prompt, completion


@torch.inference_mode()
def score_passkey(model, tokenizer, device: str, *,
                  ctx_len: int, depth: float, key: str) -> PasskeyResult:
    prompt, completion = _build_prompt(tokenizer, ctx_len, depth, key)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    # Forward, compute per-token logprobs for the completion positions.
    out = model(full_ids, use_cache=False)
    # out.logits shape: (1, L, vocab). The logits at position i predict token at position i+1.
    logits = out.logits[0]                                           # (L, vocab)
    # Completion tokens occupy the last completion_ids.numel() positions of `full_ids`.
    start = prompt_ids.shape[1]                                      # index of first completion token
    pred_logits = logits[start - 1 : start - 1 + completion_ids.shape[1]]   # (C, vocab)
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    token_idx = completion_ids[0]                                    # (C,)
    tok_logprobs = log_probs.gather(1, token_idx.unsqueeze(1)).squeeze(1)  # (C,)
    argmax_tokens = pred_logits.argmax(dim=-1)                       # (C,)
    correct = bool(torch.equal(argmax_tokens, token_idx))
    logprob_sum = float(tok_logprobs.sum().item())
    return PasskeyResult(
        ctx_len=ctx_len,
        depth=depth,
        key=key,
        logprob_sum=logprob_sum,
        logprob_per_token=logprob_sum / max(completion_ids.shape[1], 1),
        correct=correct,
        completion_tokens=int(completion_ids.shape[1]),
    )


def run_passkey_suite(model, tokenizer, device: str, *,
                      ctx_lens: list[int], depths: list[float],
                      n_trials_per_cell: int = 5, seed: int = 0) -> dict[str, Any]:
    """Run passkey retrieval across a grid of (ctx_len × depth). Returns aggregated stats."""
    rng = random.Random(seed)
    rows = []
    for L in ctx_lens:
        for d in depths:
            results: list[PasskeyResult] = []
            for _ in range(n_trials_per_cell):
                key = f"{rng.randint(10_000, 99_999)}"
                r = score_passkey(model, tokenizer, device, ctx_len=L, depth=d, key=key)
                results.append(r)
            correct_frac = sum(r.correct for r in results) / len(results)
            mean_lpt = sum(r.logprob_per_token for r in results) / len(results)
            rows.append({
                "ctx_len": L,
                "depth": d,
                "n_trials": n_trials_per_cell,
                "correct_frac": correct_frac,
                "mean_logprob_per_token": mean_lpt,
                "per_trial": [
                    {"key": r.key, "correct": r.correct,
                     "logprob_per_token": r.logprob_per_token,
                     "completion_tokens": r.completion_tokens}
                    for r in results
                ],
            })
    return {"rows": rows}
