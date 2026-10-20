import numpy as np
from trulens.core import Metric, Selector
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.openai import OpenAI

from .config import Settings

# Judge names — kept in one place so every layer agrees on the column order.
ANSWER_RELEVANCE = "answer_relevance"
CONTEXT_RELEVANCE = "context_relevance"
CONCISENESS = "conciseness"
JUDGE_NAMES = [ANSWER_RELEVANCE, CONTEXT_RELEVANCE, CONCISENESS]

# Which judges a keep/reject gate should default to requiring.
TRIAD = [ANSWER_RELEVANCE, CONTEXT_RELEVANCE]


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

    judges = [f_answer_relevance, f_context_relevance]

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
