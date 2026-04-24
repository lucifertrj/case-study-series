import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_TITLE = "HOME-TAKER API"
API_VERSION = "1.0.0"

# CORS Configuration
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# OpenAI Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500

# Data Storage Configuration
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PREFERENCES_FILE = DATA_DIR / "preferences.json"

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HTTPS = os.getenv("QDRANT_HTTPS", "true").lower() == "true"

# Embedding Configuration
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
EMBEDDING_DIMENSION = 1024
