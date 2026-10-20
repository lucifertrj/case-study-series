from dataclasses import dataclass, field
import json
import random
from pathlib import Path
from typing import Any

from openai import OpenAI as OpenAIClient

from .app import build_app
from .calibration import _judge_answerable
from .config import Settings, get_settings
from .programmatic import is_refusal, passed_all, run_production_checks

TRAFFIC_PATH = Path("evals/production_traffic.jsonl")


@dataclass
class MonitorReport:
    sampled: int
    total: int
    flagged: list[dict[str, Any]] = field(default_factory=list)

    @property
    def flag_rate(self) -> float:
        return len(self.flagged) / self.sampled if self.sampled else 0.0


def _load_traffic(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"No traffic log at {path}. Add one de-identified question per line "
            'as JSON, e.g. {"input": "how long do refunds take?"}'
        )
    queries = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                queries.append(json.loads(line)["input"])
    return queries


def monitor(
    settings: Settings | None = None,
    traffic_path: Path = TRAFFIC_PATH,
    sample_rate: float = 0.2,
    seed: int = 7,
) -> MonitorReport:
    settings = settings or get_settings()
    queries = _load_traffic(traffic_path)

    rng = random.Random(seed)
    k = max(1, round(len(queries) * sample_rate))
    sample = rng.sample(queries, min(k, len(queries)))

    app = build_app(settings)
    client = OpenAIClient(api_key=settings.openai_api_key or None)

    report = MonitorReport(sampled=len(sample), total=len(queries))
    for question in sample:
        contexts = app.retrieve_plain(question)
        answer = app.generate_plain(question, contexts)

        reasons: list[str] = []
        checks = run_production_checks(answer)
        if not passed_all(checks):
            reasons += [c.reason for c in checks if c.score == 0]

        judge_label, judge_reason = _judge_answerable(
            client, settings.eval_model, question, contexts
        )
        if judge_label == 0 and not is_refusal(answer):
            reasons.append(
                f"possible fabrication: judge says unanswerable ({judge_reason}) "
                "but the assistant did not refuse"
            )

        if reasons:
            report.flagged.append(
                {
                    "input": question,
                    "output": answer,
                    "reasons": reasons,
                }
            )

    return report


def format_monitor(report: MonitorReport) -> str:
    lines = [
        "Production monitoring (sampled live traffic):",
        f"  sampled:    {report.sampled} / {report.total}",
        f"  flagged:    {len(report.flagged)} ({report.flag_rate:.0%})",
    ]
    for item in report.flagged:
        lines.append(f"\n- input: {item['input']}")
        lines.append(f"  output: {item['output']}")
        for reason in item["reasons"]:
            lines.append(f"  ! {reason}")
    if not report.flagged:
        lines.append("\n  Nothing flagged in this sample.")
    return "\n".join(lines)
