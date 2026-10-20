"""`edd` — the Eval-Driven Development command line.

Commands mirror the stages of the blog:

    index        Index the support-policy corpus into Qdrant Cloud
    golden       Validate the gold set (positives + clean negatives)
    explore      "Look at your data" — read outputs & traces before building evals
    evaluate     Run the three-layer evaluation on the current config
    run <name>   Run one experiment from the plan, decide keep/reject + ship gates
    report <name>  Inspect the worst-scoring cases + judge reasons
    calibrate    Calibrate the virtual judge against human labels (Cohen's kappa)
    monitor      Mirror the evals on sampled production traffic
    ask <q>      Ask the assistant a single question
    dashboard    Launch the TruLens dashboard
"""

import argparse
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="trulens")
logging.getLogger("trulens.core.utils.imports").setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval-Driven Development for a Haystack + Qdrant + TruLens "
        "support-policy RAG assistant."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="Index the policy corpus into Qdrant Cloud")
    sub.add_parser("golden", help="Validate the gold set (positives + clean negatives)")

    explore_p = sub.add_parser(
        "explore", help="Read outputs & contexts for the golden questions"
    )
    explore_p.add_argument("--limit", type=int, default=0, help="0 = all")

    sub.add_parser("evaluate", help="Run the three-layer eval on the baseline")

    run_p = sub.add_parser(
        "run", help="Run one experiment, persist it, decide keep/reject/revert"
    )
    run_p.add_argument("name")
    run_p.add_argument("--plan", default="evals/experiment_plan.json")

    report_p = sub.add_parser(
        "report", help="Show the worst-scoring cases for an experiment"
    )
    report_p.add_argument("name")
    report_p.add_argument("--worst", type=int, default=5)
    report_p.add_argument("--metric", default=None)

    cal_p = sub.add_parser(
        "calibrate", help="Calibrate the virtual judge against human labels"
    )
    cal_p.add_argument("--target", type=float, default=0.85)

    mon_p = sub.add_parser("monitor", help="Mirror evals on sampled traffic")
    mon_p.add_argument("--sample-rate", type=float, default=0.2)
    mon_p.add_argument("--traffic", default="evals/production_traffic.jsonl")

    ask_p = sub.add_parser("ask", help="Ask the assistant a single question")
    ask_p.add_argument("question")

    sub.add_parser("dashboard", help="Launch the TruLens dashboard UI")

    args = parser.parse_args()

    if args.command == "index":
        from .indexing import index_corpus

        print(f"Indexed {index_corpus()} chunks into Qdrant.")

    elif args.command == "golden":
        from .dataset import load_validated_items

        items = load_validated_items(Path("evals/gold_set.jsonl"))
        positives = sum(1 for i in items if i["label"] == 1)
        negatives = len(items) - positives
        print(
            f"Gold set valid: {len(items)} items "
            f"({positives} positives, {negatives} clean negatives)."
        )

    elif args.command == "explore":
        _explore(args.limit)

    elif args.command == "evaluate":
        from .evals import run_evaluation

        result = run_evaluation()
        _print_scores(result.aggregated_scores, result.experiment_scores)

    elif args.command == "run":
        from .loop import run_iteration

        verdict = run_iteration(args.name, Path(args.plan))
        print(f"\n{verdict.decision.upper()}: {verdict.reason}")
        for c in verdict.comparisons:
            print(
                f"  {c.metric}: {c.baseline} -> {c.current} (delta {c.delta})"
            )
        if verdict.gates:
            print("  Ship gates (what must be true before shipping):")
            for g in verdict.gates:
                mark = "PASS" if g.passed else "FAIL"
                print(f"    [{mark}] {g.metric}: {g.value} (>= {g.threshold})")
            print(f"  Shippable: {verdict.shippable}")
        if verdict.plateaued:
            print(
                "\nPlateaued: no movement over the last few kept runs. "
                "Time for a structural change (reranking, chunking, hybrid search)."
            )

    elif args.command == "report":
        from .report import load_and_format

        print(load_and_format(args.name, n=args.worst, metric=args.metric))

    elif args.command == "calibrate":
        from .calibration import calibrate, format_calibration

        result = calibrate(target=args.target)
        print(format_calibration(result, target=args.target))

    elif args.command == "monitor":
        from .monitor import format_monitor, monitor

        report = monitor(
            traffic_path=Path(args.traffic), sample_rate=args.sample_rate
        )
        print(format_monitor(report))

    elif args.command == "ask":
        from .app import build_app

        result = build_app().answer(args.question)
        print(result["answer"])

    elif args.command == "dashboard":
        _dashboard()


def _print_scores(aggregated: dict, experiment: dict) -> None:
    print("Aggregated (virtual judges + programmatic checks):")
    for metric, stats in aggregated.items():
        print(f"  {metric}: mean={stats['mean']:.3f}")
    print("Experiment-level:")
    for metric, score in experiment.items():
        print(f"  {metric}: {score['value']:.3f}")


def _explore(limit: int) -> None:
    from .app import build_app
    from .dataset import load_validated_items
    from .programmatic import run_checks

    items = load_validated_items(Path("evals/gold_set.jsonl"))
    if limit > 0:
        items = items[:limit]
    app = build_app()
    for item in items:
        result = app.answer(item["input"])
        checks = run_checks(result["answer"], item["label"])
        print("=" * 72)
        print(f"Q (label={item['label']}): {item['input']}")
        print(f"A: {result['answer']}")
        print(f"retrieved {len(result['contexts'])} excerpt(s)")
        for c in checks:
            flag = "ok " if c.score else "!! "
            print(f"  [{flag}{c.name}] {c.reason}")


def _dashboard() -> None:
    try:
        from trulens.dashboard import run_dashboard
    except ImportError:
        print("trulens-dashboard is not installed. pip install trulens-dashboard")
        return
    from trulens.core import TruSession

    session = TruSession()
    print("Starting TruLens dashboard on http://localhost:8501 ...")
    proc = run_dashboard(session)
    try:
        proc.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
