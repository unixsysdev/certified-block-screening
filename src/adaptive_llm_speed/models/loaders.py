from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


@dataclass
class LoadedModel:
    model: "torch.nn.Module"
    tokenizer: Any
    name: str
    dtype: torch.dtype
    device: str


def load_model(cfg: dict) -> LoadedModel:
    """Load a causal LM from config.

    Expected config shape (from configs/base/model_*.yaml):
        model:
          name: "Qwen/Qwen3-0.6B"
          dtype: "bfloat16"
          device: "cuda"
          attn_impl: "sdpa"   # or "eager" / "flash_attention_2"
          max_seq_len: 40960
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = cfg["model"]
    name = mcfg["name"]
    dtype = _DTYPE_MAP[mcfg.get("dtype", "bfloat16")]
    device = mcfg.get("device", "cuda")
    attn_impl = mcfg.get("attn_impl", "sdpa")

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=mcfg.get("trust_remote_code", False))
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=mcfg.get("trust_remote_code", False),
    )
    model.eval()
    model.to(device)
    return LoadedModel(model=model, tokenizer=tok, name=name, dtype=dtype, device=device)


def model_info(loaded: LoadedModel) -> dict[str, Any]:
    m = loaded.model
    cfg = m.config
    n_params = sum(p.numel() for p in m.parameters())
    info = {
        "name": loaded.name,
        "dtype": str(loaded.dtype),
        "device": loaded.device,
        "architectures": getattr(cfg, "architectures", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "total_params": n_params,
    }
    return info
