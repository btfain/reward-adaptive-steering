"""Base + reward model loading, hooked generation with residual-stream steering.

Environment-agnostic on purpose (CLAUDE.md rule 6): nothing here knows about
bandits, turns, or rewards beyond "score this (prompt, response) pair".
"""

import contextlib
import json
import resource
import time
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "configs" / "base.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    if cfg["device"] != "auto":
        return torch.device(cfg["device"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_base(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=getattr(torch, cfg["dtype"])
    ).to(device)
    model.eval()
    return model, tok


def load_rm(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg["reward_model"])
    rm = AutoModelForSequenceClassification.from_pretrained(
        cfg["reward_model"],
        torch_dtype=getattr(torch, cfg["dtype"]),
        num_labels=1,
    ).to(device)
    rm.eval()
    return rm, tok


@contextlib.contextmanager
def steering_hook(model, layer_idx, vector, alpha):
    """Add alpha * vector to the residual stream at layer_idx for every
    forward position (prompt and generated tokens alike)."""
    if vector is None or alpha == 0.0:
        yield
        return
    vec = vector.to(next(model.parameters()).device, torch.float32)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + alpha * vec.to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate(model, tok, prompt, cfg, system=None, vector=None, alpha=0.0):
    """Chat-templated generation with optional steering. Returns response text."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    gen = cfg["generation"]
    with steering_hook(model, cfg["steer_layer"], vector, alpha):
        out = model.generate(
            **inputs,
            max_new_tokens=gen["max_new_tokens"],
            do_sample=gen["do_sample"],
            temperature=gen["temperature"],
            top_p=gen["top_p"],
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def rm_score(rm, rm_tok, prompt, response):
    """Skywork-Reward-V2 usage: chat-templated (user, assistant) pair -> scalar logit."""
    conv = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    inputs = rm_tok.apply_chat_template(conv, return_tensors="pt", return_dict=True).to(
        rm.device
    )
    return rm(**inputs).logits[0][0].item()


def log_cost(stage, event, wall_s, device, notes=""):
    """Append a measured cost record (CLAUDE.md rule 3: measured, never estimated)."""
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    mps_gb = (
        torch.mps.driver_allocated_memory() / 1e9
        if torch.backends.mps.is_available()
        else None
    )
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": stage,
        "event": event,
        "wall_s": round(wall_s, 2),
        "peak_rss_gb": round(peak_rss_gb, 3),
        "mps_driver_gb": round(mps_gb, 3) if mps_gb is not None else None,
        "device": str(device),
        "notes": notes,
    }
    path = REPO_ROOT / "results" / "cost_log.jsonl"
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
