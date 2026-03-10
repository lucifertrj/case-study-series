import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv("../.env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIM: int = 1536
INTENT_MODEL: str = "gpt-4o-mini"

QDRANT_ENDPOINT: str = os.getenv("QDRANT_ENDPOINT", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME: str = "properties"
PROPERTY_QA_COLLECTION_NAME: str = "property_qa"
PROPERTY_QA_DENSE_VECTOR_NAME: str = "dense"
PROPERTY_QA_SPARSE_VECTOR_NAME: str = "sparse"
PROPERTY_QA_SPARSE_MODEL: str = "Qdrant/bm25"
PROPERTY_QA_DENSE_DIM: int = EMBEDDING_DIM

PROPERTIES_FILE: Path = PROJECT_ROOT / "data" / "properties.json"

WEIGHTS = {
    "semantic": 0.40,
    "filter":   0.30,
    "quality":  0.15,
    "memory":   0.15,
}

QDRANT_TOP_K: int = 30
FINAL_TOP_K: int = 10
