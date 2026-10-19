"""Eval-Driven Development (EDD) for a support-policy RAG assistant.

Stack: Haystack 3.x (orchestration) + Qdrant Cloud (vector search) +
TruLens (virtual judges, tracing, dashboard).

This is a faithful replication of the three-layer evaluation stack from
Airbnb's "Eval-driven development" post:

    Layer 1  Programmatic & heuristic checks   (edd.programmatic)
    Layer 2  Virtual judges / LLM-as-judge     (edd.judges)
    Layer 3  Human evaluation & calibration    (edd.calibration)

IMPORTANT: TruLens only records OTEL spans (and therefore only produces
non-empty feedback scores) when OTEL tracing is enabled. This flag MUST be set
*before* anything from ``trulens`` is imported, so we set it here at package
import time, which is the first thing to run for every ``edd`` entry point.
"""

import os

os.environ.setdefault("TRULENS_OTEL_TRACING", "1")
