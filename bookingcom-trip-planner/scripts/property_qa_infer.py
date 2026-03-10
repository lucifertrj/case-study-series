"""
Property QA Inference Script

Runs hybrid retrieval + grounded QA generation from the property QA RAG index.

Examples:
  python -m scripts.property_qa_infer --question "Which property in Bali is best for a honeymoon with sunset views?"
  python -m scripts.property_qa_infer --question "Find highly rated options in Tokyo with gym access" --top-k 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from property_qa import answer_property_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Property QA hybrid RAG inference")
    parser.add_argument("--question", required=True, help="Traveler question for QA")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved sources")
    args = parser.parse_args()

    result = answer_property_question(question=args.question, top_k=args.top_k)

    print("\n=== Property QA Answer ===")
    print(result.get("answer", "").strip())
    print("\n=== Sources ===")
    print(json.dumps(result.get("sources", []), indent=2))


if __name__ == "__main__":
    main()
