"""Index the support-policy corpus into Qdrant Cloud.

A standard Haystack 3.x indexing pipeline:

    converter -> cleaner -> splitter -> embedder -> writer

Re-run this whenever ``chunk_size``, ``chunk_overlap``, the embedding model, or
the source documents change.
"""

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from .config import Settings, get_settings
from .store import get_document_store


def _source_files(settings: Settings) -> list[str]:
    if not settings.corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found at {settings.corpus_dir}. "
            "Add the support-policy markdown files there."
        )
    files = sorted(str(p) for p in settings.corpus_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No .md files found in {settings.corpus_dir}."
        )
    return files


def index_corpus(settings: Settings | None = None) -> int:
    """Index every markdown policy into Qdrant and return the chunk count."""
    settings = settings or get_settings()
    store = get_document_store(settings, recreate=True)

    pipeline = Pipeline()
    pipeline.add_component("converter", MarkdownToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter",
        DocumentSplitter(
            split_by="word",
            split_length=settings.chunk_size,
            split_overlap=settings.chunk_overlap,
        ),
    )
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=Secret.from_token(settings.openai_api_key)
            if settings.openai_api_key
            else Secret.from_env_var("OPENAI_API_KEY"),
            model=settings.embedding_model,
        ),
    )
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE),
    )

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    result = pipeline.run({"converter": {"sources": _source_files(settings)}})
    return int(result["writer"]["documents_written"])
