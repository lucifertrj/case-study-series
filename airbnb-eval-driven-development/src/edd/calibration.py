from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from openai import OpenAI as OpenAIClient

from .app import build_app
from .config import Settings, get_settings
from .dataset import load_validated_items
from .evals import GOLDEN_PATH

CALIBRATION_RUBRIC = (
    "You are calibrating a support assistant's ability to tell answerable "
    "questions from unanswerable ones. You are given a guest QUESTION and the "
    "policy EXCERPTS that were retrieved for it. Decide whether the excerpts "
    "actually contain the information needed to answer the question.\n\n"
    "Return answerable=1 only if a correct, specific answer can be supported by "
    "the excerpts. Return answerable=0 if the excerpts do not cover the question "
    "(the assistant should refuse rather than guess).\n\n"
    'Respond ONLY as JSON: {"answerable": <0 or 1>, "reason": "<short>"}'
)


@dataclass
class CalibrationResult:
    dimension: str
    n: int
    agreement: float
    cohens_kappa: float
    confusion: dict[str, int]  # tp, tn, fp, fn (positive = answerable)
    disagreements: list[dict[str, Any]]
    target_met: bool


def _judge_answerable(
    client: OpenAIClient, model: str, question: str, contexts: list[str]
) -> tuple[int, str]:
    excerpts = "\n\n".join(
        f"[excerpt {i + 1}] {c}" for i, c in enumerate(contexts)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CALIBRATION_RUBRIC},
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nEXCERPTS:\n{excerpts}",
            },
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        return int(parsed.get("answerable", 0)), str(parsed.get("reason", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0, f"unparseable judge output: {raw[:120]}"


def _cohens_kappa(tp: int, tn: int, fp: int, fn: int) -> float:
    """Two-rater, two-class Cohen's kappa."""
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    # Marginals: rater A = human, rater B = judge; positive = answerable.
    p_yes = ((tp + fn) / n) * ((tp + fp) / n)
    p_no = ((tn + fp) / n) * ((tn + fn) / n)
    pe = p_yes + p_no
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def calibrate(
    settings: Settings | None = None,
    golden_path: Path = GOLDEN_PATH,
    target: float = 0.85,
) -> CalibrationResult:
    settings = settings or get_settings()
    items = load_validated_items(golden_path)
    app = build_app(settings)
    client = OpenAIClient(api_key=settings.openai_api_key or None)

    tp = tn = fp = fn = 0
    agree = 0
    disagreements: list[dict[str, Any]] = []

    for item in items:
        human = 1 if item["label"] == 1 else 0  # human: answerable?
        contexts = app.retrieve_plain(item["input"])
        judge, reason = _judge_answerable(
            client, settings.eval_model, item["input"], contexts
        )

        if human == 1 and judge == 1:
            tp += 1
        elif human == 0 and judge == 0:
            tn += 1
        elif human == 0 and judge == 1:
            fp += 1
        else:
            fn += 1

        if human == judge:
            agree += 1
        else:
            disagreements.append(
                {
                    "input": item["input"],
                    "human_label": human,
                    "judge_label": judge,
                    "judge_reason": reason,
                    "critique": item["critique"],
                }
            )

    n = len(items)
    agreement = agree / n if n else 0.0
    kappa = _cohens_kappa(tp, tn, fp, fn)

    return CalibrationResult(
        dimension="answerability",
        n=n,
        agreement=agreement,
        cohens_kappa=kappa,
        confusion={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        disagreements=disagreements,
        target_met=agreement >= target,
    )


def format_calibration(result: CalibrationResult, target: float = 0.85) -> str:
    lines = [
        f"Calibration — dimension: {result.dimension}",
        f"  items:          {result.n}",
        f"  agreement:      {result.agreement:.1%}  (target >= {target:.0%})",
        f"  Cohen's kappa:  {result.cohens_kappa:.3f}",
        f"  confusion:      {result.confusion}  (positive = answerable)",
        f"  verdict:        {'TRUSTWORTHY' if result.target_met else 'NOT YET — refine rubric & few-shots, then re-run'}",
    ]
    if result.disagreements:
        lines.append("\n  Disagreements to analyze:")
        for d in result.disagreements:
            lines.append(
                f"   - human={d['human_label']} judge={d['judge_label']}: "
                f"{d['input']}"
            )
            lines.append(f"     judge reason: {d['judge_reason']}")
    return "\n".join(lines)
