import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"input", "expected_output", "label", "critique"}


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
