from __future__ import annotations

import datetime as _dt
import json
import socket
from pathlib import Path
from typing import Any


def save_result(out_dir: str | Path, run_id: str, payload: dict[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_id = run_id.replace("/", "_")
    path = out / f"{safe_id}.json"
    payload = {
        "run_id": run_id,
        "created_at": _dt.datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        **payload,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
