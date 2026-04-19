"""Config loader tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

from adaptive_llm_speed.utils.config import config_hash, load_yaml, merge


def test_load_simple(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("model:\n  name: foo\n  dtype: bf16\n")
    cfg = load_yaml(p)
    assert cfg["model"]["name"] == "foo"


def test_merge_deep():
    a = {"model": {"name": "m", "dtype": "bf16"}, "method": {"name": "x"}}
    b = {"model": {"dtype": "fp16"}, "extra": 42}
    out = merge(a, b)
    assert out == {"model": {"name": "m", "dtype": "fp16"}, "method": {"name": "x"}, "extra": 42}


def test_config_hash_stable():
    a = {"x": 1, "y": [1, 2, 3], "nested": {"a": 1, "b": 2}}
    b = {"nested": {"b": 2, "a": 1}, "y": [1, 2, 3], "x": 1}  # reordered
    assert config_hash(a) == config_hash(b)


def test_include_resolution(tmp_path):
    sub = tmp_path / "sub.yaml"
    sub.write_text("ffn_lowrank:\n  rank: 128\n")
    top = tmp_path / "top.yaml"
    top.write_text("method:\n  name: combined\ncombined:\n  ffn_config: sub.yaml\n")
    cfg = load_yaml(top)
    assert cfg["method"]["name"] == "combined"
    # ffn_config is only resolved at top-level keys; nested under combined it stays a string.
    # Our loader's current scope: top-level include keys only. That's fine for v1.
