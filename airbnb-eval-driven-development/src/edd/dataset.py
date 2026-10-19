"""The golden dataset — the single most important artifact in EDD.

Per the blog: "Include bad examples in your Gold Set. You can't test discernment
without them." So every row carries a human ``label``:

    label = 1  answerable from the policy corpus (a positive)
    label = 0  a clean negative — NOT covered by any policy; the assistant is
               expected to refuse honestly rather than fabricate an answer.

Schema (one JSON object per line):
    input           the guest's question
    expected_output the ground-truth answer lifted from the corpus (positives)
                    or a plausible-but-unsupported answer (clean negatives)
    metadata.source the policy file the answer comes from (or null)
    label           1 or 0 (human ground truth)
    critique        why it is grounded, or why it is unanswerable
"""

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"input", "expected_output", "metadata", "label", "critique"}


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open() as dataset_file:
        return [json.loads(line) for line in dataset_file if line.strip()]


def validate_items(items: list[dict[str, Any]]) -> None:
    if not items:
        raise ValueError("Golden dataset is empty.")
    positives = sum(1 for item in items if item.get("label") == 1)
    negatives = sum(1 for item in items if item.get("label") == 0)

    for position, item in enumerate(items, start=1):
        missing = REQUIRED_FIELDS.difference(item)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Golden item {position} is missing: {fields}")
        if not isinstance(item["input"], str) or not isinstance(
            item["expected_output"], str
        ):
            raise ValueError(
                f"Golden item {position}: input and expected_output must be strings"
            )
        if not isinstance(item["metadata"], dict):
            raise ValueError(f"Golden item {position}: metadata must be an object")
        if item["label"] not in (0, 1):
            raise ValueError(f"Golden item {position}: label must be 0 or 1")
        if not isinstance(item["critique"], str):
            raise ValueError(f"Golden item {position}: critique must be a string")

    # The whole point of a gold set is discernment — enforce both classes.
    if positives == 0 or negatives == 0:
        raise ValueError(
            "Golden dataset must contain BOTH positives (label 1) and clean "
            f"negatives (label 0). Found {positives} positive(s), "
            f"{negatives} negative(s)."
        )


def load_validated_items(path: Path) -> list[dict[str, Any]]:
    items = load_items(path)
    validate_items(items)
    return items
