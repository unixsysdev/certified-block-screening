"""Perplexity on a fixed calibration passage, sliding-window style.

This is intentionally tiny — a ~4k-token snippet tokenized once, run at full
precision through the model, and averaged over non-overlapping windows. It's the
'smoke test' PPL, not the paper-grade number; swap in a real corpus later.

Choosing a passage: we embed it in the code so offline runs work. The text is a
couple of paragraphs from a CC-BY-SA Wikipedia article, short enough that we
aren't distributing a dataset, long enough that the PPL is stable.
"""
from __future__ import annotations

import math
from typing import Any

import torch


_CALIBRATION_TEXT = """
Information theory is the mathematical study of the quantification, storage, and
communication of information. The field was established and put on a firm footing
by Claude Shannon in the 1940s, though early contributions were made in the 1920s
through the works of Harry Nyquist and Ralph Hartley. It is at the intersection
of electronic engineering, mathematics, statistics, computer science, neurobiology,
physics, and electrical engineering.

A key measure in information theory is entropy. Entropy quantifies the amount of
uncertainty involved in the value of a random variable or the outcome of a random
process. For example, identifying the outcome of a fair coin flip — with two
equally likely outcomes — provides less information than specifying the outcome
of a roll of a fair die with six equally likely outcomes. Some other important
measures in information theory are mutual information, channel capacity, error
exponents, and relative entropy.

Important sub-fields of information theory include source coding, algorithmic
complexity theory, algorithmic information theory, and information-theoretic
security. Applications of fundamental topics of information theory include
lossless data compression, lossy data compression, and channel coding.
Its impact has been crucial to the success of the Voyager missions to deep space,
the invention of the compact disc, the feasibility of mobile phones, and the
development of the Internet and artificial intelligence.

The theory has also found applications in other areas, including statistical
inference, cryptography, neurobiology, perception, linguistics, the evolution
and function of molecular codes (bioinformatics), thermal physics, molecular
dynamics, black holes, quantum computing, and even plagiarism detection. This
breadth reflects the generality of the mathematical machinery Shannon introduced
to model communication over a noisy channel.
"""


@torch.inference_mode()
def perplexity_on_calibration(model, tokenizer, device: str, *,
                              window: int = 512, stride: int | None = None) -> dict[str, Any]:
    """Compute token-level perplexity via non-overlapping windows of the calibration text."""
    stride = stride or window
    enc = tokenizer(_CALIBRATION_TEXT, return_tensors="pt")
    ids = enc["input_ids"][0].to(device)
    if ids.numel() < window + 1:
        # Tokenizer produced too few tokens; fall back to the whole thing.
        window = max(8, ids.numel() - 1)
        stride = window

    nll_sum = 0.0
    ntok = 0
    for start in range(0, max(1, ids.numel() - window), stride):
        chunk = ids[start:start + window]
        if chunk.numel() < 2:
            continue
        input_ids = chunk.unsqueeze(0)
        labels = chunk.clone().unsqueeze(0)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        # HF returns mean loss over non-ignored positions. We approximate token count as window - 1.
        nt = window - 1
        nll_sum += float(out.loss.item()) * nt
        ntok += nt
    if ntok == 0:
        return {"perplexity": float("nan"), "nll_mean": float("nan"), "tokens": 0, "window": window}
    nll_mean = nll_sum / ntok
    return {
        "perplexity": math.exp(nll_mean),
        "nll_mean": nll_mean,
        "tokens": ntok,
        "window": window,
    }
