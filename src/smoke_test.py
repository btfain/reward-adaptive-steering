"""Stage A step 1 smoke test: base model generates, RM scores, on 5 UltraFeedback
prompts. Also verifies the steering hook is mechanically sound (a zero vector must
not change anything; a random vector at large alpha must change the output)."""

import json
import time

import torch
from datasets import load_dataset

from models import (
    REPO_ROOT,
    generate,
    load_base,
    load_config,
    load_rm,
    log_cost,
    resolve_device,
    rm_score,
)


def main():
    t0 = time.time()
    cfg = load_config()
    device = resolve_device(cfg)
    torch.manual_seed(cfg["seed"])
    print(f"device={device}")

    model, tok = load_base(cfg, device)
    n_layers = model.config.num_hidden_layers
    print(f"base={cfg['base_model']} layers={n_layers} hidden={model.config.hidden_size}")
    assert cfg["steer_layer"] < n_layers, "steer_layer out of range"

    rm, rm_tok = load_rm(cfg, device)
    print(f"rm={cfg['reward_model']}")

    d = cfg["data"]
    stream = load_dataset(d["prompt_dataset"], split=d["prompt_split"], streaming=True)
    prompts = [row[d["prompt_field"]] for _, row in zip(range(5), stream)]

    results = []
    for i, prompt in enumerate(prompts):
        response = generate(model, tok, prompt, cfg)
        score = rm_score(rm, rm_tok, prompt, response)
        results.append({"prompt": prompt, "response": response, "rm_score": score})
        print(f"\n--- prompt {i + 1} ---\n{prompt[:200]}")
        print(f"--- response (rm={score:.3f}) ---\n{response[:400]}")

    # Hook mechanical check at fixed seed: zero vector is a no-op, random vector isn't.
    hidden = model.config.hidden_size
    torch.manual_seed(cfg["seed"])
    base_out = generate(model, tok, prompts[0], cfg)
    torch.manual_seed(cfg["seed"])
    zero_out = generate(model, tok, prompts[0], cfg, vector=torch.zeros(hidden), alpha=1.0)
    rand_vec = torch.randn(hidden)
    rand_vec = rand_vec / rand_vec.norm()
    torch.manual_seed(cfg["seed"])
    rand_out = generate(model, tok, prompts[0], cfg, vector=rand_vec, alpha=8.0)
    assert zero_out == base_out, "zero-vector steering changed output"
    assert rand_out != base_out, "large random steering vector did not change output"
    print("\nhook check: zero-vector no-op OK, random-vector perturbs OK")

    out_path = REPO_ROOT / "results" / "smoke_test.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    scores = [r["rm_score"] for r in results]
    print(f"\nrm scores: {[round(s, 3) for s in scores]}")
    print(log_cost("A", "smoke_test", time.time() - t0, device))


if __name__ == "__main__":
    main()
