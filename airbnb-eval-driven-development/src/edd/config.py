from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Everything a single EDD experiment needs to run.

    An "experiment" in the Airbnb EDD loop varies exactly ONE of these knobs at
    a time (a model, a retrieval_k, a prompt) so that any metric movement can be
    attributed to that single change.
    """

    # --- Corpus & vector store ---
    corpus_dir: Path = Path("data/support_policies")
    collection_name: str = "support_policies"
    chunk_size: int = 150  # words per chunk (Haystack DocumentSplitter)
    chunk_overlap: int = 30

    # --- Models ---
    embedding_model: str = "text-embedding-3-small"
    # The generator under test.
    chat_model: str = "gpt-4o-mini"
    # A *stronger / different* model judges the generator's output, per the
    # blog's "different judge than generator" rule.
    eval_model: str = "gpt-4o"

    # --- Retrieval ---
    retrieval_k: int = 4

    # --- Credentials (from .env) ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    # Optional: point the TruLens store somewhere other than ./default.sqlite.
    database_url: str | None = os.getenv("TRULENS_DATABASE_URL", None)


def get_settings() -> Settings:
    return Settings()
