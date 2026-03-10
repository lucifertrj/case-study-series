from __future__ import annotations
import os
from openai import OpenAI
from config import EMBEDDING_MODEL

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client



def embed(text: str) -> list[float]:
    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.replace("\n", " "),
    )
    return response.data[0].embedding



def embed_batch(texts: list[str]) -> list[list[float]]:
    cleaned = [t.replace("\n", " ") for t in texts]
    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned,
    )
    return [item.embedding for item in response.data]
