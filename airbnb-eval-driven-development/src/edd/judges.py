"""Layer 2 — Virtual judges (LLM-as-judge).

A small, sharp panel of judges, one per correctness dimension (no "God
evaluator"), following the blog's five rules:

    * one evaluator per dimension
    * a *different / stronger* model than the generator (``eval_model``)
    * chain-of-thought reasons on every score
    * an explicit scoring contract (TruLens' *_with_cot_reasons providers)

The RAG triad covers the dimensions that matter for a grounded support bot:

    groundedness       is the answer supported by the retrieved policy excerpts?
    answer_relevance   does the answer actually address the question?
    context_relevance  did retrieval surface relevant policy excerpts?

plus one product-specific judge from the walkthrough:

    conciseness        is the answer tight, or padded/verbose?

Each selector is keyed to the exact OTEL span attribute emitted in ``edd.app``,
so the judge inputs and the app instrumentation always reference the same key.
"""

import numpy as np
from trulens.core import Metric, Selector
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import OpenAI

from .config import Settings

# Judge names — kept in one place so every layer agrees on the column order.
GROUNDEDNESS = "groundedness"
ANSWER_RELEVANCE = "answer_relevance"
CONTEXT_RELEVANCE = "context_relevance"
CONCISENESS = "conciseness"
JUDGE_NAMES = [GROUNDEDNESS, ANSWER_RELEVANCE, CONTEXT_RELEVANCE, CONCISENESS]

# Which judges a keep/reject gate should default to requiring.
TRIAD = [GROUNDEDNESS, ANSWER_RELEVANCE, CONTEXT_RELEVANCE]


def get_provider(settings: Settings) -> OpenAI:
    return OpenAI(
        model_engine=settings.eval_model,
        api_key=settings.openai_api_key or None,
    )


def _retrieved_contexts(collect_list: bool) -> Selector:
    return Selector(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
        collect_list=collect_list,
    )


def _query_text() -> Selector:
    return Selector(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
    )


def get_virtual_judges(settings: Settings) -> list[Metric]:
    provider = get_provider(settings)

    f_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons_consider_answerability,
        name=GROUNDEDNESS,
        selectors={
            "source": _retrieved_contexts(collect_list=True),
            "statement": Selector.select_record_output(),
            "question": Selector.select_record_input(),
        },
    )

    f_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name=ANSWER_RELEVANCE,
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    f_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name=CONTEXT_RELEVANCE,
        selectors={
            "question": _query_text(),
            "context": _retrieved_contexts(collect_list=False),
        },
        agg=np.mean,
    )

    judges = [f_groundedness, f_answer_relevance, f_context_relevance]

    # Conciseness is a product-specific extra, not part of the required gate.
    # Guard it so an API-name change in the provider can never take down the
    # whole panel (and therefore `edd evaluate` / `edd run`).
    conciseness_impl = getattr(
        provider, "conciseness_with_cot_reasons", None
    ) or getattr(provider, "conciseness", None)
    if conciseness_impl is not None:
        judges.append(
            Metric(
                implementation=conciseness_impl,
                name=CONCISENESS,
                selectors={"text": Selector.select_record_output()},
            )
        )
    else:
        print("[judges] conciseness not available in this TruLens provider; skipping.")

    return judges
