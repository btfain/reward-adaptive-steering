"""Analytic GOLD judge (Qwen) for the mentor v2 rubric: emits the five category scores in one pass; the
overall is summed in code as C1+C2+C3+C4+C5 (0-21; C5 already on its 0/3/6/9 weighted scale). Held out from
discovery (Qwen never selects moves) and rubric-blind everywhere except here. Reused by M0 (headroom +
validity) and later by the discovery/eval stages.
"""

import re

import torch

_SYS = ("You are a strict, fair grading assistant. Score a programming tutor's reply to a student using the "
        "provided analytic rubric, judging only against the rubric. Output the five category scores in the "
        "exact format requested and nothing else.")

_TASK = """Analytic rubric (assign each category one of its listed point values):
{rubric}

Student message:
{prompt}

Tutor reply to score:
{response}

Output EXACTLY these five lines and then stop:
C1: <0, 1, 2, or 3>
C2: <0, 1, 2, or 3>
C3: <0, 1, 2, or 3>
C4: <0, 1, 2, or 3>
C5: <0, 3, 6, or 9>"""

_PAT = {c: re.compile(rf"{c}\s*[:=]\s*(\d+)") for c in ["C1", "C2", "C3", "C4", "C5"]}


def score_analytic(mdl, tok, items, rubric, max_new_tokens=160, batch=8):
    """items: list of (prompt_str, response_str). Returns list of {C1..C5, overall} dicts (None if unparsed
    or out of range). overall = C1+C2+C3+C4+C5."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": _SYS},
             {"role": "user", "content": _TASK.format(rubric=rubric, prompt=p, response=r)}],
            add_generation_prompt=True, tokenize=False) for p, r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        for row in gen:
            txt = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            vals, ok = {}, True
            for c, pat in _PAT.items():
                m = pat.search(txt)
                if not m:
                    ok = False; break
                vals[c] = int(m.group(1))
            valid = ok and all(vals[c] <= 3 for c in ["C1", "C2", "C3", "C4"]) and vals["C5"] in (0, 3, 6, 9)
            if valid:
                vals["overall"] = vals["C1"] + vals["C2"] + vals["C3"] + vals["C4"] + vals["C5"]
                out.append(vals)
            else:
                out.append(None)
    return out
