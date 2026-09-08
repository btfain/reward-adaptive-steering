"""Deterministic checks + prompt templates for the synthetic notation positive-control. Each context TYPE
has a set of programmatically-checkable constraints; a response is scored ONLY against its prompt's type.
Also builds the per-type move bundles (system prompts) and the `full_rubric` baseline prompt from the config.

No judge, no GPU: score(response, type) -> fraction of that type's constraints satisfied, in [0,1].
Checks are shared across types where sensible (cross-cutting compliance rules), which also stresses the
full_rubric baseline. Constraints (key -> instruction text) live in the config; check fns live here (CHECKS).
"""

import re
import yaml
from pathlib import Path

from models import REPO_ROOT

_AMERICAN = ["color", "behavior", "organize", "center", "favorite", "analyze", "catalog", "honor",
             "labor", "flavor", "neighbor", "recognize", "traveled", "modeling", "defense", "license",
             "gray", "meter", "liter", "apologize"]


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


def _numbered(r):
    return [ln.strip() for ln in r.splitlines() if re.match(r"^\s*\d+[.)]\s+", ln)]


def _speakers(r):
    return [ln for ln in r.splitlines() if re.match(r"^[A-Z][A-Za-z0-9 ]{0,18}:\s", ln.strip())]


def _last(r):
    ls = [ln for ln in r.strip().splitlines() if ln.strip()]
    return ls[-1] if ls else ""


def _nwords(s):
    return len(re.sub(r"^\s*(\d+[.)]|[-*•—])\s*", "", s).split())   # words after a bullet/number marker


CHECKS = {
    # --- cross-cutting ---
    "british_spelling": lambda r: not any(re.search(rf"\b{w}\b", r, re.I) for w in _AMERICAN),
    "no_word_very":     lambda r: not re.search(r"\bvery\b", r, re.I),
    "no_word_just":     lambda r: not re.search(r"\bjust\b", r, re.I),
    "no_exclamation":   lambda r: "!" not in r,
    "no_contractions":  lambda r: not re.search(r"\b\w+['’]\w+\b", r),
    # --- HARD cross-cutting (v3): individually taxing, jointly overloading ---
    "no_comma":         lambda r: "," not in r,
    "no_word_the":      lambda r: not re.search(r"\bthe\b", r, re.I),
    "no_ly_adverbs":    lambda r: not re.search(r"\b[A-Za-z]+ly\b", r, re.I),
    "ends_period":      lambda r: r.strip().endswith("."),
    "no_lists":         lambda r: not any(re.match(r"^\s*([-*•—]\s+|\d+[.)]\s+)", ln) for ln in r.splitlines()),
    "brackets_numbers": lambda r: not re.search(r"(?<![\[\w])\d+(?!\])", r),
    "max_2_sentences":  lambda r: len(_sentences(r)) <= 2,
    "max_3_sentences":  lambda r: len(_sentences(r)) <= 3,
    "max_8_sentences":  lambda r: len(_sentences(r)) <= 8,
    "min_3_sentences":  lambda r: len(_sentences(r)) >= 3,
    "min_4_sentences":  lambda r: len(_sentences(r)) >= 4,
    "min_5_sentences":  lambda r: len(_sentences(r)) >= 5,
    "word_count_max_60": lambda r: len(r.split()) <= 60,
    # --- starts_with prefixes ---
    "starts_answer":     lambda r: r.strip().startswith("Answer:"),
    "starts_steps":      lambda r: r.strip().startswith("Steps:"),
    "starts_solution":   lambda r: r.strip().startswith("Solution:"),
    "starts_summary":    lambda r: r.strip().startswith("Summary:"),
    "starts_definition": lambda r: r.strip().startswith("Definition:"),
    "starts_comparison": lambda r: r.strip().startswith("Comparison:"),
    "starts_list":       lambda r: r.strip().startswith("List:"),
    "starts_scene":      lambda r: r.strip().startswith("Scene:"),
    "starts_review":     lambda r: r.strip().startswith("Review:"),
    # --- ends / contains ---
    "end_complexity":  lambda r: bool(re.search(r"(?i)complexity:\s*O\(", _last(r))),
    "ends_summary":    lambda r: bool(re.match(r"(?i)^—\s*summary:", _last(r))),
    "ends_rating":     lambda r: bool(re.search(r"(?i)rating:\s*\d\s*/\s*5", _last(r))),
    "final_answer":    lambda r: bool(re.match(r"(?i)^final answer:", _last(r))),
    "has_equation":    lambda r: "=" in r,
    "contains_def":    lambda r: "def " in r,
    # --- code ---
    "has_code_block":   lambda r: "```" in r,
    "code_tabs":        lambda r: (any(_code_lines(r)) and
                                   not any(re.match(r"^ +\S", ln) for ln in _code_lines(r))),
    "no_code_comments": lambda r: not any(("#" in ln or "//" in ln) for ln in _code_lines(r)),
    # --- bullets ---
    "exactly_3_bullets": lambda r: len(_bullets(r)) == 3,
    "exactly_5_bullets": lambda r: len(_bullets(r)) == 5,
    "dash_bullets":      lambda r: (len(_bullets(r)) > 0 and all(b.startswith("—") for b in _bullets(r))),
    "bullet_max_12_words": lambda r: len(_bullets(r)) > 0 and all(_nwords(b) <= 12 for b in _bullets(r)),
    "bullet_max_15_words": lambda r: len(_bullets(r)) > 0 and all(_nwords(b) <= 15 for b in _bullets(r)),
    # --- numbered list ---
    "numbered_list":   lambda r: len(_numbered(r)) >= 1,
    "exactly_7_items": lambda r: len(_numbered(r)) == 7,
    "item_max_8_words": lambda r: len(_numbered(r)) > 0 and all(_nwords(n) <= 8 for n in _numbered(r)),
    # --- email ---
    "email_subject": lambda r: r.strip().startswith("Subject:"),
    "email_dear":    lambda r: bool(re.search(r"\bDear\b", r)),
    "email_regards": lambda r: "Best regards," in r,
    # --- dialogue ---
    "has_speakers": lambda r: len(_speakers(r)) >= 1,
    "min_4_turns":  lambda r: len(_speakers(r)) >= 4,
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
    rules = " ".join(c["text"] for c in cfg["types"][typ]["constraints"])
    return f"Follow these formatting rules exactly in your reply: {rules}"


def full_rubric(cfg):
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
    "factual": ["the capital of Australia", "who wrote Pride and Prejudice", "the speed of light in a vacuum",
                "the boiling point of water in Celsius", "which planet is the largest",
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
             "A recipe for 4 needs 300g flour; how much for 7", "Factor x squared minus nine",
             "What is 15 percent of 240"],
    "email": ["request a deadline extension from a professor", "thank a mentor after an internship",
              "ask a colleague to review a document", "decline a meeting politely",
              "follow up on a job application", "apologise for a shipping delay to a customer",
              "invite a team to a planning session", "request time off from a manager"],
    "summary": ["The industrial revolution began in Britain in the late eighteenth century and shifted "
                "production from hand tools to machines, reshaping cities and labour.",
                "Photosynthesis lets plants convert sunlight, water, and carbon dioxide into glucose and "
                "oxygen, forming the base of most food chains.",
                "The printing press, introduced by Gutenberg, made books cheaper and faster to produce and "
                "helped spread literacy across Europe.",
                "Vaccines train the immune system by exposing it to a harmless form of a pathogen so it can "
                "respond quickly to a real infection.",
                "Plate tectonics describes how large sections of the Earth's crust move slowly, causing "
                "earthquakes, volcanoes, and mountain ranges.",
                "The internet grew from a research network into a global system connecting billions of "
                "devices and reshaping commerce and communication.",
                "Supply and demand describe how the price of a good tends to settle where the quantity buyers "
                "want equals the quantity sellers offer.",
                "The water cycle moves water through evaporation, condensation, and precipitation, "
                "continuously recycling it between the oceans, air, and land."],
    "definition": ["osmosis", "inflation", "recursion", "entropy", "democracy", "photosynthesis",
                   "an algorithm", "gravity"],
    "comparison": ["Python and JavaScript", "cats and dogs", "coffee and tea", "trains and planes",
                   "renting and buying a home", "solar and wind power", "email and instant messaging",
                   "libraries and bookshops"],
    "list": ["seven benefits of regular exercise", "seven common HTML tags", "seven tips for better sleep",
             "seven staple pantry ingredients", "seven ways to save energy at home",
             "seven common programming languages", "seven steps to plan a trip", "seven famous rivers"],
    "dialogue": ["a customer returns a faulty phone to a shop assistant",
                 "two friends decide which film to watch", "a student asks a professor for an extension",
                 "a traveller asks a stranger for directions", "a job candidate answers a tough interview "
                 "question", "a chef explains a dish to a curious diner",
                 "two coworkers plan a surprise party", "a parent helps a child with homework"],
    "review": ["a fictional science-fiction film", "a cosy neighbourhood cafe", "a budget wireless headphone",
               "a new fantasy novel", "a small family-run restaurant", "a productivity mobile app",
               "a weekend camping tent", "an indie video game"],
}
_TEMPL = {
    "code": "Write a Python function to {t}.",
    "factual": "What is {t}?",
    "howto": "How do I {t}?",
    "prose": "Write a short descriptive paragraph about {t}.",
    "math": "{t}.",
    "email": "Draft an email to {t}.",
    "summary": "Summarise this passage: {t}",
    "definition": "Define {t} in your own words.",
    "comparison": "Compare {t}.",
    "list": "List {t}.",
    "dialogue": "Write a short dialogue in which {t}.",
    "review": "Write a review of {t}.",
}


def prompts(cfg):
    n = cfg["data"]["n_per_type"]; out = []
    for t in types(cfg):
        tops = _TOPICS[t]
        for i in range(n):
            out.append({"type": t, "prompt": _TEMPL[t].format(t=tops[i % len(tops)])})
    for i, p in enumerate(out):
        p["ci"] = i
    return out
