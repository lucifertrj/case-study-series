import uuid
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_HTTPS,
    QDRANT_PORT,
)


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    _client: QdrantClient | None = None
    _embedding_model: TextEmbedding | None = None

    @classmethod
    def get_client(cls) -> QdrantClient:
        """Get or create Qdrant client instance."""
        if cls._client is None:
            cls._client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY,
                https=QDRANT_HTTPS,
            )
        return cls._client

    @classmethod
    def get_embedding_model(cls) -> TextEmbedding:
        """Get or create embedding model instance."""
        if cls._embedding_model is None:
            cls._embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        return cls._embedding_model

    @classmethod
    async def initialize_collection(cls) -> None:
        """Initialize Qdrant collection if it doesn't exist."""
        client = cls.get_client()

        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(
            collection.name == QDRANT_COLLECTION_NAME for collection in collections
        )

        if not collection_exists:
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION, distance=Distance.COSINE
                ),
            )

    @classmethod
    async def add_documents(
        cls, documents: list[dict[str, Any]], file_id: str | None = None
    ) -> int:
        """
        Add documents to Qdrant with embeddings.

        Args:
            documents: List of dicts with 'text' and optional metadata
            file_id: Optional file ID to associate documents with

        Returns:
            Number of documents added
        """
        client = cls.get_client()
        embedding_model = cls.get_embedding_model()

        # Ensure collection exists
        await cls.initialize_collection()

        # Generate embeddings for all documents
        texts = [doc["text"] for doc in documents]
        embeddings = list(embedding_model.embed(texts))

        # Create points for upsert
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id = doc.get("id") or str(uuid.uuid4())
            payload = {
                "text": doc["text"],
                "file_id": file_id,
                **{k: v for k, v in doc.items() if k not in ["text", "id"]},
            }

            points.append(
                PointStruct(id=i, vector=embedding.tolist(), payload=payload)
            )

        # Upsert to Qdrant
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

        return len(points)

    @classmethod
    async def search(
        cls, query: str, limit: int = 5, file_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents in Qdrant.

        Args:
            query: Search query text
            limit: Number of results to return
            file_id: Optional file ID to filter results

        Returns:
            List of search results with text and score
        """
        client = cls.get_client()
        embedding_model = cls.get_embedding_model()

        # Ensure collection exists
        await cls.initialize_collection()

        # Generate query embedding
        query_embedding = list(embedding_model.embed([query]))[0]

        # Build filter if file_id is provided
        query_filter = None
        if file_id:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="file_id", match=MatchValue(value=file_id)
                    )
                ]
            )

        # Search
        results = client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        # Format results
        formatted_results = []
        for point in results.points:
            formatted_results.append(
                {
                    "text": point.payload.get("text", ""),
                    "score": point.score,
                    "metadata": {
                        k: v
                        for k, v in point.payload.items()
                        if k not in ["text"]
                    },
                }
            )

        return formatted_results

    @classmethod
    async def delete_by_file_id(cls, file_id: str) -> bool:
        """
        Delete all documents associated with a file ID.

        Args:
            file_id: The file ID to delete

        Returns:
            True if deletion was successful
        """
        client = cls.get_client()

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Delete points with matching file_id
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file_id", match=MatchValue(value=file_id)
                    )
                ]
            ),
        )

        return True

    @classmethod
    async def clear_all(cls) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if clear was successful
        """
        client = cls.get_client()

        # Delete the collection and recreate it
        client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        await cls.initialize_collection()

        return True
