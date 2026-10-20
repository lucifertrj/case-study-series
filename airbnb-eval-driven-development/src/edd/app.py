from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
)
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from .config import Settings, get_settings
from .store import get_document_store

# The generation prompt is the single most-tuned "one variable" in the EDD loop.
# Keep it here so a prompt experiment is a small, reviewable diff.
SYSTEM_PROMPT = (
    "You are a support assistant for a travel platform. Answer the guest's "
    "question using ONLY the policy excerpts provided. Be concise and specific: "
    "quote exact timeframes, percentages, and conditions when the policy states "
    "them. If the policy excerpts do not contain the answer, reply exactly: "
    '"I don\'t have a policy that covers this, so I can\'t say." Never invent '
    "a policy, a number, or a timeframe that is not in the excerpts."
)


def _format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(
        f"[excerpt {i + 1}] {chunk}" for i, chunk in enumerate(contexts)
    )


class SupportPolicyRAG:
    """Instrumented retrieve-then-generate assistant over the policy corpus."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        api_key = (
            Secret.from_token(self.settings.openai_api_key)
            if self.settings.openai_api_key
            else Secret.from_env_var("OPENAI_API_KEY")
        )
        store = get_document_store(self.settings)
        self._embedder = OpenAITextEmbedder(
            api_key=api_key, model=self.settings.embedding_model
        )
        self._retriever = QdrantEmbeddingRetriever(
            document_store=store, top_k=self.settings.retrieval_k
        )
        self._generator = OpenAIChatGenerator(
            api_key=api_key, model=self.settings.chat_model
        )

    # --- Plain implementations (no TruLens span) -------------------------
    # The instrumented methods below wrap these. Code paths that run *outside*
    # a recording context (edd ask / explore / calibrate / monitor) call the
    # _impl methods directly, so they never depend on an active TruSession.

    def _retrieve_impl(self, query: str) -> list[str]:
        embedding = self._embedder.run(text=query)["embedding"]
        documents = self._retriever.run(
            query_embedding=embedding, top_k=self.settings.retrieval_k
        )["documents"]
        return [doc.content for doc in documents if doc.content]

    def _generate_impl(self, query: str, contexts: list[str]) -> str:
        messages = [
            ChatMessage.from_system(SYSTEM_PROMPT),
            ChatMessage.from_user(
                f"Policy excerpts:\n{_format_contexts(contexts)}\n\n"
                f"Guest question: {query}"
            ),
        ]
        replies = self._generator.run(messages=messages)["replies"]
        return replies[0].text if replies else ""

    # --- Instrumented surface (recorded by TruLens during evaluation) ----

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list[str]:
        return self._retrieve_impl(query)

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, contexts: list[str]) -> str:
        return self._generate_impl(query, contexts)

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        contexts = self.retrieve(query)
        return self.generate(query, contexts)

    # --- Uninstrumented helpers for non-recording CLI commands -----------

    def retrieve_plain(self, query: str) -> list[str]:
        return self._retrieve_impl(query)

    def generate_plain(self, query: str, contexts: list[str]) -> str:
        return self._generate_impl(query, contexts)

    def answer(self, query: str) -> dict:
        """Uninstrumented path for the ``edd ask`` CLI command."""
        contexts = self._retrieve_impl(query)
        return {
            "question": query,
            "answer": self._generate_impl(query, contexts),
            "contexts": contexts,
        }


def build_app(settings: Settings | None = None) -> SupportPolicyRAG:
    # A fresh app per experiment: each experiment may vary chat_model /
    # retrieval_k, so we must not reuse a cached instance across experiments.
    return SupportPolicyRAG(settings or get_settings())
