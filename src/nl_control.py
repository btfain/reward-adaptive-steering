"""M2 control modality: natural-language imperatives delivered as a system message.

The imperative is the ACTION, never shown to the reward model — exactly like an
M1 steering vector. Poles mirror the M1 sign convention: `pos` pushes toward the
axis's positive pole (the proxy's higher direction: hedge / elaborate / formal /
cautious / warm / inquire), `neg` toward the opposite. Two strengths (mod, strong)
mirror M1's |alpha| in {0.1, 0.2}. Vocabulary lives in configs/headroom.yaml so it
can be edited without touching code.
"""

POLES = ("pos", "neg")
STRENGTHS = ("mod", "strong")


def validate_imperatives(imperatives, names):
    """Every active axis needs both poles at both strengths."""
    missing = []
    for n in names:
        for pole in POLES:
            for s in STRENGTHS:
                if not imperatives.get(n, {}).get(pole, {}).get(s):
                    missing.append(f"{n}.{pole}.{s}")
    if missing:
        raise RuntimeError(f"headroom.yaml imperatives incomplete: {missing}")


def m2_conditions(imperatives, names):
    """One condition per (axis, pole, strength); system = the imperative text.
    id encodes sign so it lines up with M1: m2:{axis}{+/-}{0.1|0.2}."""
    validate_imperatives(imperatives, names)
    conds = []
    for n in names:
        for pole in POLES:
            sign = "+" if pole == "pos" else "-"
            for s, mag in (("mod", "0.1"), ("strong", "0.2")):
                conds.append({
                    "id": f"m2:{n}{sign}{mag}",
                    "arm": "m2",
                    "axis": n,
                    "pole": pole,
                    "strength": s,
                    "system": imperatives[n][pole][s],
                })
    return conds
