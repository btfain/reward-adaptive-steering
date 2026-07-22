"""Cheap lexical proxies, one per axis. Signed so higher = more of the pos pole.

These quantify monotonicity; they do not adjudicate it (qualitative-first rule
in PLAN.md). Rates are per-100-words to decorrelate from length — except the
length axis itself, where length is the point.
"""

import re

HEDGE = [
    "might", "may", "perhaps", "possibly", "likely", "probably", "could",
    "seems", "appears", "i think", "i believe", "not sure", "uncertain",
    "arguably", "somewhat", "it depends", "hard to say",
]
PUSHBACK = [
    "however", "actually", "in fact", "that said", "not necessarily",
    "not quite", "incorrect", "mistaken", "flawed", "questionable",
    "misconception", "i disagree", "on the contrary", "worth questioning",
]
AGREEMENT = [
    "great question", "you're right", "you are right", "absolutely",
    "definitely", "of course", "exactly", "great idea", "good point",
    "i agree", "certainly",
]
CAVEAT = [
    "keep in mind", "note that", "be careful", "consult", "may not apply",
    "double-check", "at your own risk", "depending on", "edge case",
    "caution", "warning", "risk", "disclaimer", "professional advice",
]
# extended 2026-07-22: compliance samples were genuinely warm ("hello, i'm
# thrilled...") but used words the original lexicon lacked
WARMTH = [
    "glad", "happy to", "hope this helps", "hope that helps", "feel free",
    "wonderful", "good luck", "you're welcome", "thanks for", "i'm here",
    "don't hesitate", "great to", "thrilled", "excited", "delighted",
    "love to", "amazing", "fantastic", "hello", "hi there", "we take pride",
]
CASUAL = [
    "gonna", "wanna", "kinda", "sorta", "yeah", "yep", "cool", "awesome",
    "stuff", "hey", "btw", "lol", "pretty much", "no worries",
]
CLARIFY = [
    "could you", "can you tell me", "do you mean", "what is your",
    "what's your", "would you like", "let me know", "clarify", "which one",
    "more details", "more information about your",
]

CONTRACTION_RE = re.compile(r"\b\w+'(?:t|re|ve|ll|d|m)\b", re.IGNORECASE)


def _words(text):
    return max(len(text.split()), 1)


def _rate(text, lexicon):
    """Occurrences of lexicon phrases per 100 words."""
    low = text.lower()
    hits = sum(low.count(phrase) for phrase in lexicon)
    return 100.0 * hits / _words(text)


def _question_count(text):
    """Sentences ending in '?'. (v2 required second-person wording, which
    undercounted real clarifying questions like 'What is the company's
    industry?' — dropped 2026-07-22.)"""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return sum(1 for s in sentences if s.rstrip().endswith("?"))


def challenge_accommodate(text):
    return _rate(text, PUSHBACK) - _rate(text, AGREEMENT)


def hedge_assert(text):
    return _rate(text, HEDGE)


def elaborate_concise(text):
    return float(len(text.split()))


def formal_casual(text):
    casualness = _rate(text, CASUAL) + 100.0 * (
        len(CONTRACTION_RE.findall(text)) + text.count("!")
    ) / _words(text)
    return -casualness


def cautious_direct(text):
    return _rate(text, CAVEAT)


def warm_neutral(text):
    # '!' term removed 2026-07-22: exclamations also feed the casualness proxy,
    # and the shared term manufactured spurious warm<->formal cross-steering.
    # Exclamations stay with casualness (register), warmth is lexicon-only.
    return _rate(text, WARMTH)


def inquire_proceed(text):
    return 100.0 * _question_count(text) / _words(text)


# Per-axis pair filters for extraction (option 1, approved 2026-07-22): keep
# only pairs where the pos completion exhibits the behavior at all. Both poles
# of a retained pair enter the mean difference, so the contrast stays matched.
PAIR_FILTERS = {
    "inquire_proceed": lambda pos_completion: "?" in pos_completion,
    # added 2026-07-22 after challenge failed the cross-steering prong: keep
    # only pairs whose challenge completion contains at least one pushback marker
    "challenge_accommodate": lambda pos_completion: _rate(pos_completion, PUSHBACK) > 0,
}
MIN_RETAINED_PAIRS = 40  # pre-registered floor; below this, stop and rebrief

PROXIES = {
    "challenge_accommodate": challenge_accommodate,
    "hedge_assert": hedge_assert,
    "elaborate_concise": elaborate_concise,
    "formal_casual": formal_casual,
    "cautious_direct": cautious_direct,
    "warm_neutral": warm_neutral,
    "inquire_proceed": inquire_proceed,
}
