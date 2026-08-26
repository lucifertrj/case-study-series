import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    model: str
    grading_temperature: float
    chat_temperature: float
    grading_max_output_tokens: int
    chat_max_output_tokens: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    vector_size: int


@dataclass(frozen=True)
class VectorConfig:
    collection_name: str
    distance: str
    chunk_size: int
    chunk_stride: int
    search_limit: int


@dataclass(frozen=True)
class QdrantConfig:
    url: str | None
    api_key: str | None


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector: VectorConfig
    qdrant: QdrantConfig


config = AppConfig(
    llm=LLMConfig(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        model=os.getenv("LLM_MODEL", "gemini-3.1-flash-lite"),
        grading_temperature=_get_float("LLM_GRADING_TEMPERATURE", 0.0),
        chat_temperature=_get_float("LLM_CHAT_TEMPERATURE", 0.2),
        grading_max_output_tokens=_get_int("LLM_GRADING_MAX_OUTPUT_TOKENS", 300),
        chat_max_output_tokens=_get_int("LLM_CHAT_MAX_OUTPUT_TOKENS", 300),
    ),
    embedding=EmbeddingConfig(
        model=os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v2-small-en"),
        vector_size=_get_int("VECTOR_SIZE", 512),
    ),
    vector=VectorConfig(
        collection_name=os.getenv("VECTOR_COLLECTION_NAME", "offer_chunks"),
        distance=os.getenv("VECTOR_DISTANCE", "COSINE"),
        chunk_size=_get_int("VECTOR_CHUNK_SIZE", 1200),
        chunk_stride=_get_int("VECTOR_CHUNK_STRIDE", 1000),
        search_limit=_get_int("VECTOR_SEARCH_LIMIT", 3),
    ),
    qdrant=QdrantConfig(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    ),
)
