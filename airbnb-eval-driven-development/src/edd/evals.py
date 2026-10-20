import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from trulens.apps.app import TruApp
from trulens.core import TruSession

from .app import build_app
from .config import Settings, get_settings
from .dataset import load_validated_items
from .judges import JUDGE_NAMES, get_virtual_judges
from .programmatic import CHECK_NAMES, checks_to_scores, run_checks

APP_NAME = "support-policy-assistant"
GOLDEN_PATH = Path("evals/gold_set.jsonl")


@dataclass
class EvaluationResult:
    experiment_name: str
    app_id: str
    aggregated_scores: dict[str, dict[str, float]]
    experiment_scores: dict[str, dict[str, Any]]
    cases: list[dict[str, Any]]


def _retrieve_feedback_df(recording: Any, session: TruSession) -> Any:
    """Get the records+feedback DataFrame, tolerating TruLens API variants."""
    retriever = getattr(recording, "retrieve_feedback_results", None)
    if callable(retriever):
        return retriever(timeout=600)
    record_ids = [r.record_id for r in getattr(recording, "records", [])]
    df, _ = session.get_records_and_feedback(record_ids=record_ids)
    return df


def _count_completed_scores(
    session: TruSession, record_ids: list[str], feedback_names: list[str]
) -> int:
    """How many (record, judge) scores have landed in the store so far."""
    try:
        df, _ = session.get_records_and_feedback(record_ids=record_ids)
    except Exception:
        return 0
    done = 0
    for name in feedback_names:
        if name in df.columns:
            done += int(df[name].notna().sum())
    return done


def _await_feedback_with_progress(
    recording: Any,
    session: TruSession,
    record_ids: list[str],
    feedback_names: list[str],
    poll_seconds: float = 2.0,
) -> Any:
    """Block on TruLens judge scoring while showing live per-judge progress.

    ``retrieve_feedback_results`` is a single opaque blocking call, so we run it
    off-thread and poll the store for how many judge scores have actually landed
    — that's the only honest progress signal (each judge cell goes non-null when
    that judge finishes for that record).
    """
    holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            holder["df"] = _retrieve_feedback_df(recording, session)
        except Exception as exc:  # re-raised after the bar is torn down
            holder["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    total = len(record_ids) * len(feedback_names)
    with tqdm(total=total, desc="Scoring (judges)", unit="score") as bar:
        while worker.is_alive():
            bar.n = min(
                _count_completed_scores(session, record_ids, feedback_names), total
            )
            bar.refresh()
            time.sleep(poll_seconds)
        worker.join()
        # Final read: reflect reality — a short bar here means some judges failed.
        bar.n = min(_count_completed_scores(session, record_ids, feedback_names), total)
        bar.refresh()

    if "error" in holder:
        raise holder["error"]
    return holder["df"]


def _cell(df: Any, col: str, idx: int) -> float | None:
    if col not in df.columns or idx >= len(df):
        return None
    val = df[col].iloc[idx]
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _reason(df: Any, col: str, idx: int) -> str:
    calls_col = f"{col}_calls"
    if calls_col in df.columns and idx < len(df):
        calls = df[calls_col].iloc[idx]
        if isinstance(calls, list) and calls:
            meta = calls[0].get("meta", {})
            return str(meta.get("reason") or meta.get("reasons") or "")
    return ""


def run_evaluation(
    settings: Settings | None = None,
    experiment_name: str | None = None,
    golden_path: Path = GOLDEN_PATH,
) -> EvaluationResult:
    settings = settings or get_settings()
    exp_name = (
        experiment_name
        or f"{settings.chat_model}-k{settings.retrieval_k}"
    )

    session = TruSession(database_url=settings.database_url)
    items = load_validated_items(golden_path)
    judges = get_virtual_judges(settings)
    app = build_app(settings)

    tru_app = TruApp(
        app,
        app_name=APP_NAME,
        app_version=exp_name,
        feedbacks=judges,
        main_method=app.query,
        connector=session.connector,
        metadata={
            "chat_model": settings.chat_model,
            "retrieval_k": settings.retrieval_k,
            "eval_model": settings.eval_model,
            "decision": "pending",  # filled in by the loop once decided
        },
    )

    print(f"\n[eval] Experiment '{exp_name}' over {len(items)} golden items.")
    outputs: list[str] = []
    with tru_app as recording:
        for item in tqdm(items, desc="Generating answers", unit="q"):
            outputs.append(app.query(item["input"]))

    print("[eval] Awaiting virtual-judge scores from TruLens...")
    session.force_flush()
    record_ids = [r.record_id for r in getattr(recording, "records", [])]
    feedback_names = [j.name for j in judges]
    records_df = _await_feedback_with_progress(
        recording, session, record_ids, feedback_names
    )
    print("[eval] Scoring complete.\n")

    # --- Per-case: merge Layer-1 programmatic checks + Layer-2 judge scores ---
    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        answer = outputs[idx] if idx < len(outputs) else ""
        case_scores: dict[str, dict[str, Any]] = {}

        for judge in JUDGE_NAMES:
            value = _cell(records_df, judge, idx)
            case_scores[judge] = {
                "value": value,
                "reason": _reason(records_df, judge, idx),
                "scoring_failed": value is None,
            }

        checks = run_checks(answer, item["label"])
        case_scores.update(checks_to_scores(checks))

        cases.append(
            {
                "dataset_item_id": idx + 1,
                "input": item["input"],
                "expected_output": item["expected_output"],
                "label": item["label"],
                "output": answer,
                "scores": case_scores,
            }
        )

    aggregated_scores = _aggregate(cases, JUDGE_NAMES + CHECK_NAMES)
    experiment_scores = _experiment_metrics(cases)

    return EvaluationResult(
        experiment_name=exp_name,
        app_id=tru_app.app_id,
        aggregated_scores=aggregated_scores,
        experiment_scores=experiment_scores,
        cases=cases,
    )


def _aggregate(
    cases: list[dict[str, Any]], names: list[str]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in names:
        vals = [
            c["scores"][name]["value"]
            for c in cases
            if name in c["scores"]
            and not c["scores"][name]["scoring_failed"]
            and isinstance(c["scores"][name]["value"], (int, float))
        ]
        if vals:
            out[name] = {
                "mean": float(np.mean(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }
        else:
            out[name] = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return out


def _experiment_metrics(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Metrics that only make sense across the whole run."""
    negatives = [c for c in cases if c["label"] == 0]
    correct_refusals = sum(
        1
        for c in negatives
        if c["scores"].get("refusal_behavior", {}).get("value") == 1.0
    )
    clean_negative_recall = (
        correct_refusals / len(negatives) if negatives else 0.0
    )

    programmatic_pass = sum(
        1
        for c in cases
        if all(
            c["scores"].get(name, {}).get("value") == 1.0 for name in CHECK_NAMES
        )
    )
    return {
        "clean_negative_recall": {
            "value": clean_negative_recall,
            "reason": (
                f"{correct_refusals}/{len(negatives)} clean negatives honestly "
                "refused."
            ),
        },
        "programmatic_pass_rate": {
            "value": programmatic_pass / len(cases) if cases else 0.0,
            "reason": f"{programmatic_pass}/{len(cases)} passed all Layer-1 checks.",
        },
    }
