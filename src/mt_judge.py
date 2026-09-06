"""Prometheus-2 as the reward for the multi-turn controller: score an assistant reply a2 for
CONTEXT-APPROPRIATENESS under our rubric (configs/moves_multiturn_v1.yaml), given the dialogue (u1,a1,u2)
and, optionally, the logged a2 as the score-5 reference. Absolute grading 1-5 (Prometheus's native format).

Prometheus is a generative judge (writes feedback then '[RESULT] n'), so this loads a CausalLM and parses n.
"""

import re
from pathlib import Path

import torch
import yaml

from models import REPO_ROOT

_ABS_SYSTEM = "You are a fair judge assistant tasked with providing clear, objective feedback based on " \
    "specific criteria, ensuring each assessment reflects the absolute standards set for performance."

_ABS_TASK = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, {ref_clause}and a score rubric \
representing an evaluation criterion are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score \
rubric, not evaluating in general.
2. After writing the feedback, write a score that is an integer between 1 and 5. You should refer to the \
score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an \
integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}
{ref_block}###Score Rubrics:
{rubric}

###Feedback: """


def load_rubric(moves_config):
    return yaml.safe_load(open(REPO_ROOT / moves_config))["rubric"]


def load_judge(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return mdl, tok


def _instruction(ctx):
    """ctx = [{'role','content'} for u1,a1,u2]. Frame as: the dialogue so far + 'give the next reply'."""
    lines = []
    for m in ctx:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    convo = "\n".join(lines)
    return (f"Below is a conversation. Provide the assistant's next reply to the user's most recent "
            f"message.\n\n{convo}")


def _prompt(tok, ctx, response, rubric, reference):
    ref_clause = "a reference answer that would score 5, " if reference else ""
    ref_block = f"\n###Reference Answer (Score 5):\n{reference}\n" if reference else ""
    body = _ABS_TASK.format(ref_clause=ref_clause, instruction=_instruction(ctx),
                            response=response, ref_block=ref_block, rubric=rubric)
    return tok.apply_chat_template([{"role": "system", "content": _ABS_SYSTEM},
                                    {"role": "user", "content": body}],
                                   add_generation_prompt=True, tokenize=False)


_RESULT = re.compile(r"\[RESULT\]\s*([1-5])")


def score_batch(mdl, tok, items, rubric, max_new_tokens=512, batch=8):
    """items: list of (ctx, response, reference|None). Returns list of int scores in 1..5 (or None if unparsed)."""
    tok.padding_side = "left"
    out = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        texts = [_prompt(tok, c, r, rubric, ref) for c, r, ref in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        for row, w in zip(gen, [enc["input_ids"].shape[1]] * len(chunk)):
            txt = tok.decode(row[w:], skip_special_tokens=True)
            mobj = _RESULT.search(txt)
            out.append(int(mobj.group(1)) if mobj else None)
    return out
