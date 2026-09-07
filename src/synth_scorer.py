"""Deterministic checks + prompt templates for the synthetic notation positive-control. Each context TYPE
has 4 programmatically-checkable constraints; a response is scored ONLY against its prompt's type. Also
builds the per-type move bundles (system prompts) and the `full_rubric` baseline prompt from the config.

No judge, no GPU: score(response, type) -> fraction of that type's constraints satisfied, in [0,1].
"""

import re
import yaml
from pathlib import Path

from models import REPO_ROOT

_AMERICAN = ["color", "behavior", "organize", "center", "favorite", "analyze", "catalog", "honor",
             "labor", "flavor", "neighbor", "recognize", "traveled", "modeling"]


def _sentences(r):
    return [s for s in re.split(r"(?<=[.!?])\s+", r.strip()) if s.strip()]


def _code_lines(r):
    out, inb = [], False
    for ln in r.splitlines():
        if ln.strip().startswith("```"):
            inb = not inb; continue
        if inb:
            out.append(ln)
    return out


def _bullets(r):
    return [ln.strip() for ln in r.splitlines() if re.match(r"^\s*[-*•—]\s+", ln)]


def _last(r):
    ls = [ln for ln in r.strip().splitlines() if ln.strip()]
    return ls[-1] if ls else ""


CHECKS = {
    "has_code_block":   lambda r: "```" in r,
    "code_tabs":        lambda r: (any(_code_lines(r)) and
                                   not any(re.match(r"^ +\S", ln) for ln in _code_lines(r))),
    "no_code_comments": lambda r: not any(("#" in ln or "//" in ln) for ln in _code_lines(r)),
    "end_complexity":   lambda r: bool(re.search(r"(?i)complexity:\s*O\(", _last(r))),
    "starts_answer":    lambda r: r.strip().startswith("Answer:"),
    "max_3_sentences":  lambda r: len(_sentences(r)) <= 3,
    "brackets_numbers": lambda r: not re.search(r"(?<![\[\w])\d+(?!\])", r),
    "british_spelling": lambda r: not any(re.search(rf"\b{w}\b", r, re.I) for w in _AMERICAN),
    "exactly_5_bullets": lambda r: len(_bullets(r)) == 5,
    "dash_bullets":     lambda r: (len(_bullets(r)) > 0 and all(b.startswith("—") for b in _bullets(r))),
    "bullet_max_12_words": lambda r: all(len(b.split()) <= 13 for b in _bullets(r)) and len(_bullets(r)) > 0,
    "no_word_very":     lambda r: not re.search(r"\bvery\b", r, re.I),
    "ends_summary":     lambda r: bool(re.match(r"(?i)^—\s*summary:", _last(r))),
    "no_lists":         lambda r: not any(re.match(r"^\s*([-*•—]\s+|\d+[.)]\s+)", ln) for ln in r.splitlines()),
    "math_solution_start": lambda r: r.strip().startswith("Solution:"),
    "has_equation":     lambda r: "=" in r,
    "final_answer":     lambda r: bool(re.match(r"(?i)^final answer:", _last(r))),
    "email_subject":    lambda r: r.strip().startswith("Subject:"),
    "email_dear":       lambda r: bool(re.search(r"\bDear\b", r)),
    "email_regards":    lambda r: "Best regards," in r,
}


def load(cfg_path):
    return yaml.safe_load(open(REPO_ROOT / cfg_path))


def types(cfg):
    return list(cfg["types"].keys())


def score(cfg, response, typ):
    cons = cfg["types"][typ]["constraints"]
    if not response.strip():
        return 0.0
    return sum(bool(CHECKS[c["key"]](response)) for c in cons) / len(cons)


def move_bundle(cfg, typ):
    """System prompt enforcing exactly one type's constraints."""
    rules = " ".join(c["text"] for c in cfg["types"][typ]["constraints"])
    return f"Follow these formatting rules exactly in your reply: {rules}"


def full_rubric(cfg):
    """The baseline-to-beat: ALL conditional rules in one system prompt (the model must self-route)."""
    parts = ["Apply the formatting rules that match the request type, and only those rules."]
    for t, d in cfg["types"].items():
        rules = " ".join(c["text"] for c in d["constraints"])
        parts.append(f"If the request is {d['label']}: {rules}")
    return " ".join(parts)


# ---- typed prompt generation (known type => ground-truth oracle move) ----
_TOPICS = {
    "code": ["reverse a linked list", "check if a string is a palindrome", "merge two sorted lists",
             "compute the nth Fibonacci number", "find duplicates in an array", "validate an email address",
             "implement binary search", "count word frequencies in a paragraph"],
    "factual": ["the capital of Australia", "who wrote Pride and Prejudice", "the speed of light",
                "the boiling point of water in Celsius", "which planet is largest",
                "the year the Berlin Wall fell", "the chemical symbol for gold", "the tallest mountain on Earth"],
    "howto": ["change a flat bicycle tyre", "brew pour-over coffee", "set up a Python virtual environment",
              "propagate a succulent", "write a cover letter", "back up a phone",
              "start composting at home", "tie a bowline knot"],
    "prose": ["a foggy harbour at dawn", "the feeling of finishing a long journey", "an old bookshop",
              "a thunderstorm over the plains", "the last day of summer", "a city waking up",
              "a lighthouse keeper's routine", "the smell of rain on hot pavement"],
    "math": ["A train travels 60 km in 45 minutes; find its speed in km/h",
             "If 3x + 7 = 22, solve for x", "A shirt costs 40 dollars after a 20% discount; find the original",
             "Find the area of a circle with radius 5", "Sum the first 10 positive even numbers",
             "A recipe for 4 needs 300g flour; how much for 7", "Factor x^2 - 9",
             "What is 15% of 240"],
    "email": ["request a deadline extension from a professor", "thank a mentor after an internship",
              "ask a colleague to review a document", "decline a meeting politely",
              "follow up on a job application", "apologise for a shipping delay to a customer",
              "invite a team to a planning session", "request time off from a manager"],
}
_TEMPL = {
    "code": "Write a Python function to {t}.",
    "factual": "What is {t}?",
    "howto": "How do I {t}?",
    "prose": "Write a short descriptive paragraph about {t}.",
    "math": "{t}.",
    "email": "Draft an email to {t}.",
}


def prompts(cfg):
    """Returns list of {ci, type, prompt}; n_per_type per type, cycling topics."""
    import itertools
    n = cfg["data"]["n_per_type"]; out = []
    for t in types(cfg):
        tops = _TOPICS[t]
        for i in range(n):
            out.append({"type": t, "prompt": _TEMPL[t].format(t=tops[i % len(tops)])})
    for i, p in enumerate(out):
        p["ci"] = i
    return out
