"""Composite quality score Q(M) = sum_i w_i * (s_i(M) / s_i(B)).

For metrics where lower is better (perplexity), we invert: ratio = s_B / s_M so
higher is always better in the final Q.
"""
from __future__ import annotations

from typing import Any

# Known metric names that are "lower is better". Everything else is "higher is better".
LOWER_IS_BETTER = {"perplexity", "nll_mean"}

# Default weights — placeholder until we have capability + long-context evals wired up.
# Once those land, weights go in configs/base/eval_base.yaml instead.
DEFAULT_WEIGHTS: dict[str, float] = {
    "perplexity": 1.0,
}


def normalize(metric: str, value: float, baseline: float) -> float:
    if baseline is None or baseline == 0:
        return float("nan")
    if metric in LOWER_IS_BETTER:
        return baseline / value if value and value > 0 else 0.0
    return value / baseline


def composite_q(metrics: dict[str, float], baseline: dict[str, float],
                weights: dict[str, float] | None = None) -> dict[str, Any]:
    w = dict(weights or DEFAULT_WEIGHTS)
    # Drop metrics we don't have in both sides.
    usable = {k: v for k, v in w.items() if k in metrics and k in baseline}
    if not usable:
        return {"q": float("nan"), "components": {}, "weights": w, "reason": "no overlapping metrics"}
    total_w = sum(usable.values())
    components = {k: normalize(k, metrics[k], baseline[k]) for k in usable}
    q = sum(usable[k] * components[k] for k in usable) / total_w
    return {"q": q, "components": components, "weights": usable}
