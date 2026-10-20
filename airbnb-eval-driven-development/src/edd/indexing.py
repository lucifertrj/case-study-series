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

    sources = _source_files(settings)
    converter = MarkdownToDocument()
    cleaner = DocumentCleaner()
    splitter = DocumentSplitter(
        split_by="word",
        split_length=settings.chunk_size,
        split_overlap=settings.chunk_overlap,
    )
    embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_token(settings.openai_api_key)
        if settings.openai_api_key
        else Secret.from_env_var("OPENAI_API_KEY"),
        model=settings.embedding_model,
    )
    writer = DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE)

    docs = converter.run(sources=sources)["documents"]
    docs = cleaner.run(documents=docs)["documents"]
    chunks = splitter.run(documents=docs)["documents"]

    print(
        f"Embedding {len(chunks)} chunks from {len(sources)} policies "
        f"with {settings.embedding_model} (this is the slow part) ..."
    )

    embedded = embedder.run(documents=chunks)["documents"]
    written = writer.run(documents=embedded)["documents_written"]
    return int(written)
