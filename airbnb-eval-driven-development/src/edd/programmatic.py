import re
from dataclasses import dataclass
from typing import Any

# Terms that should never leak into a guest-facing answer.
INTERNAL_TERMS = [
    "excerpt",
    "retrieved context",
    "vector",
    "embedding",
    "qdrant",
    "system prompt",
    "as an ai",
    "language model",
]

# The exact refusal string the system prompt instructs the model to use, plus a
# couple of looser fallbacks so a lightly reworded refusal still counts.
REFUSAL_MARKERS = [
    "i don't have a policy that covers this",
    "i do not have a policy that covers this",
    "i can't say",
    "i cannot say",
    "i don't have enough information",
]

MIN_CHARS = 15
MAX_CHARS = 1500


@dataclass
class Check:
    name: str
    score: int  # 1 pass, 0 fail
    reason: str


def _looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def run_checks(answer: str, label: int) -> list[Check]:
    """Run every programmatic check for a single (answer, human label) pair."""
    answer = answer or ""
    stripped = answer.strip()
    low = stripped.lower()

    checks: list[Check] = []

    # 1. Presence
    checks.append(
        Check(
            "non_empty",
            int(bool(stripped)),
            "ok" if stripped else "empty output",
        )
    )

    # 2. Length bounds
    n = len(stripped)
    within = MIN_CHARS <= n <= MAX_CHARS
    checks.append(
        Check(
            "within_length",
            int(within),
            "ok" if within else f"length {n} outside [{MIN_CHARS}, {MAX_CHARS}]",
        )
    )

    # 3. No internal terminology leaked
    leaked = [t for t in INTERNAL_TERMS if t in low]
    checks.append(
        Check(
            "no_internal_terms",
            int(not leaked),
            "ok" if not leaked else f"leaked internal terms: {', '.join(leaked)}",
        )
    )

    # 4. Refusal behavior — the behavioral gate that makes clean negatives useful.
    #    label 0 -> the assistant SHOULD refuse; label 1 -> it should NOT refuse.
    refused = _looks_like_refusal(stripped)
    if label == 0:
        ok = refused
        reason = (
            "correctly refused an unanswerable question"
            if refused
            else "FABRICATED an answer to an unanswerable question"
        )
    else:
        ok = not refused
        reason = (
            "over-refused an answerable question"
            if refused
            else "answered an answerable question"
        )
    checks.append(Check("refusal_behavior", int(ok), reason))

    return checks


def run_production_checks(answer: str) -> list[Check]:
    """Label-free subset for live monitoring, where ground-truth is unknown.

    Drops ``refusal_behavior`` (which needs the human label) and keeps the
    purely text-based checks that are valid on unlabeled production traffic.
    """
    return [c for c in run_checks(answer, label=1) if c.name != "refusal_behavior"]


def is_refusal(answer: str) -> bool:
    return _looks_like_refusal((answer or "").strip())


def checks_to_scores(checks: list[Check]) -> dict[str, dict[str, Any]]:
    return {
        check.name: {
            "value": float(check.score),
            "reason": check.reason,
            "scoring_failed": False,
        }
        for check in checks
    }


def passed_all(checks: list[Check]) -> bool:
    return all(check.score == 1 for check in checks)


# Exposed so the eval layer can report a consistent column order.
CHECK_NAMES = ["non_empty", "within_length", "no_internal_terms", "refusal_behavior"]
