from __future__ import annotations

from typing import Any


def passthrough_patch(model, cfg: dict) -> dict[str, Any]:
    """No-op patcher. Used as the baseline control."""
    return {"method": "baseline", "touched_layers": 0}
