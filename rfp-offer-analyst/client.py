from functools import lru_cache
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from config import config


def _vector_distance() -> models.Distance:
    try:
        return models.Distance[config.vector.distance.upper()]
    except KeyError as err:
        valid = ", ".join(item.name for item in models.Distance)
        raise ValueError(f"Invalid VECTOR_DISTANCE={config.vector.distance!r}. Use one of: {valid}") from err


@lru_cache(maxsize=1)
def get_embedding_client() -> TextEmbedding:
    return TextEmbedding(model_name=config.embedding.model)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)

    if not client.collection_exists(config.vector.collection_name):
        client.create_collection(
            collection_name=config.vector.collection_name,
            vectors_config=models.VectorParams(
                size=config.embedding.vector_size,
                distance=_vector_distance(),
            ),
        )
        client.create_payload_index(
            collection_name=config.vector.collection_name,
            field_name="offer_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    return client


def get_llm_client(required: bool = True) -> Any | None:
    if not config.llm.api_key:
        if required:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Configure it in your .env to run LLM calls."
            )
        return None

    from google import genai

    return genai.Client(api_key=config.llm.api_key)
