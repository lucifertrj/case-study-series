from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .results import RESULTS_DIR, load_result

# Every metric in this project is higher-is-better on a [0, 1] scale.
LOWER_IS_BETTER: set[str] = set()
PLATEAU_WINDOW = 2


def normalize_metric_name(name: str) -> str:
    return name.lower().strip().replace(" ", "_").removesuffix("_metric")


def _metric_mean(record: dict[str, Any], metric: str) -> float | None:
    target = normalize_metric_name(metric)
    for name, stats in record.get("aggregated_scores", {}).items():
        if normalize_metric_name(name) == target:
            return stats.get("mean")
    for name, score in record.get("experiment_scores", {}).items():
        if normalize_metric_name(name) == target:
            return score.get("value")
    return None


@dataclass
class MetricComparison:
    metric: str
    current: float | None
    baseline: float | None
    delta: float | None
    direction: str
    regressed: bool


@dataclass
class GateResult:
    """An absolute ship gate — 'what must be true before you ship' (blog)."""

    metric: str
    value: float | None
    threshold: float
    passed: bool


@dataclass
class Verdict:
    experiment: str
    baseline: str | None
    decision: str  # "keep" | "reject" | "no_baseline"
    comparisons: list[MetricComparison]
    plateaued: bool
    reason: str
    gates: list[GateResult]
    gates_passed: bool
    shippable: bool


def evaluate_gates(
    current: dict[str, Any], gates: dict[str, float] | None
) -> tuple[list[GateResult], bool]:
    """Check the experiment's metrics against the upfront absolute gates."""
    results: list[GateResult] = []
    for metric, threshold in (gates or {}).items():
        value = _metric_mean(current, metric)
        passed = value is not None and value >= threshold
        results.append(GateResult(metric, value, threshold, passed))
    return results, all(r.passed for r in results)


def compare(
    current: dict[str, Any], baseline: dict[str, Any] | None
) -> list[MetricComparison]:
    required = current.get("acceptance", {}).get("required_metrics", [])
    comparisons = []
    for metric in required:
        current_value = _metric_mean(current, metric)
        baseline_value = _metric_mean(baseline, metric) if baseline else None
        direction = (
            "lower"
            if normalize_metric_name(metric) in LOWER_IS_BETTER
            else "higher"
        )
        delta = (
            round(current_value - baseline_value, 4)
            if current_value is not None and baseline_value is not None
            else None
        )
        regressed = bool(
            delta is not None
            and (delta < 0 if direction == "higher" else delta > 0)
        )
        comparisons.append(
            MetricComparison(
                metric=metric,
                current=current_value,
                baseline=baseline_value,
                delta=delta,
                direction=direction,
                regressed=regressed,
            )
        )
    return comparisons


def _is_flat(comparisons: list[MetricComparison]) -> bool:
    return bool(comparisons) and all(
        c.delta is not None and c.delta == 0 for c in comparisons
    )


def _recent_flat_streak(
    experiments: list[dict[str, Any]], up_to_name: str
) -> int:
    streak = 0
    for entry in reversed(experiments):
        if entry["name"] == up_to_name:
            continue
        result = entry.get("result")
        if result is None:
            break
        if result.get("decision") == "keep" and result.get("flat"):
            streak += 1
        else:
            break
    return streak


def decide(
    experiment_name: str,
    experiments: list[dict[str, Any]],
    gates: dict[str, float] | None = None,
    results_dir: Path = RESULTS_DIR,
) -> Verdict:
    entry = next((e for e in experiments if e["name"] == experiment_name), None)
    if entry is None:
        raise ValueError(
            f"No entry named {experiment_name!r} in experiment_plan.json"
        )

    current = load_result(experiment_name, results_dir)
    gate_results, gates_passed = evaluate_gates(current, gates)
    gate_note = _gate_note(gate_results)
    baseline_name = entry.get("acceptance", {}).get("compare_against")

    if baseline_name is None:
        return Verdict(
            experiment=experiment_name,
            baseline=None,
            decision="no_baseline",
            comparisons=[],
            plateaued=False,
            reason="No compare_against set - treated as the new baseline. " + gate_note,
            gates=gate_results,
            gates_passed=gates_passed,
            shippable=gates_passed,
        )

    baseline = load_result(baseline_name, results_dir)
    comparisons = compare(current, baseline)
    regressions = [c.metric for c in comparisons if c.regressed]
    flat = _is_flat(comparisons)

    if regressions:
        decision = "reject"
        reason = f"Regressed on: {', '.join(regressions)}."
    else:
        decision = "keep"
        reason = (
            "Flat or improved on every required metric."
            if flat
            else "Improved."
        )

    plateau_streak = _recent_flat_streak(experiments, experiment_name)
    plateaued = (
        decision == "keep" and flat and plateau_streak + 1 >= PLATEAU_WINDOW
    )

    # Relative keep/reject narrows the candidate pool; the absolute gates decide
    # whether the kept candidate is actually shippable.
    shippable = decision == "keep" and gates_passed

    return Verdict(
        experiment=experiment_name,
        baseline=baseline_name,
        decision=decision,
        comparisons=comparisons,
        plateaued=plateaued,
        reason=f"{reason} {gate_note}",
        gates=gate_results,
        gates_passed=gates_passed,
        shippable=shippable,
    )


def _gate_note(gate_results: list[GateResult]) -> str:
    if not gate_results:
        return ""
    failed = [g.metric for g in gate_results if not g.passed]
    if failed:
        return f"Ship gates NOT met: {', '.join(failed)}."
    return "All ship gates met."


def worst_cases(
    record: dict[str, Any], n: int = 5, metric: str | None = None
) -> list[dict[str, Any]]:
    required = (
        [metric]
        if metric
        else record.get("acceptance", {}).get("required_metrics", [])
    )
    required_normalized = {normalize_metric_name(m) for m in required}

    def case_score(case: dict[str, Any]) -> float:
        values = [
            score["value"]
            for score_name, score in case.get("scores", {}).items()
            if (
                not required
                or normalize_metric_name(score_name) in required_normalized
            )
            and not score.get("scoring_failed")
            and isinstance(score.get("value"), (int, float))
        ]
        return min(values) if values else float("inf")

    ranked = sorted(record.get("cases", []), key=case_score)
    return ranked[:n]


def format_report(
    record: dict[str, Any], n: int = 5, metric: str | None = None
) -> str:
    lines = [
        f"Experiment: {record['name']} ({record.get('change', '')})",
        "Aggregated scores (virtual judges + programmatic checks):",
    ]
    for name, stats in record.get("aggregated_scores", {}).items():
        lines.append(
            f"  {name}: mean={stats['mean']:.3f} "
            f"min={stats['min']:.3f} max={stats['max']:.3f}"
        )
    lines.append("Experiment-level metrics:")
    for name, score in record.get("experiment_scores", {}).items():
        lines.append(f"  {name}: {score['value']:.3f} ({score.get('reason', '')})")

    lines.append(f"\nWorst {n} cases:")
    for case in worst_cases(record, n, metric):
        lines.append(f"\n- input: {case['input']}")
        lines.append(f"  label: {case['label']}")
        lines.append(f"  expected_output: {case['expected_output']}")
        lines.append(f"  actual output: {case['output']}")
        for score_name, score in case.get("scores", {}).items():
            lines.append(
                f"  [{score_name}] {score['value']} - {score.get('reason')}"
            )
    return "\n".join(lines)


def load_and_format(name: str, n: int = 5, metric: str | None = None) -> str:
    return format_report(load_result(name), n=n, metric=metric)
