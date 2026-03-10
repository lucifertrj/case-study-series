"""
Create Payload Indexes for Qdrant Collection
Adds keyword indexes for filterable fields.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables before importing config
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

sys.path.insert(0, str(project_root / "backend"))

from qdrant_client.http.models import PayloadSchemaType
from qdrant_client_singleton import get_qdrant
from config import COLLECTION_NAME

def create_indexes():
    client = get_qdrant()

    # Create indexes for filterable fields
    indexes_to_create = [
        ("city", PayloadSchemaType.KEYWORD),
        ("type", PayloadSchemaType.KEYWORD),
        ("available", PayloadSchemaType.BOOL),
        ("price_per_night", PayloadSchemaType.FLOAT),
    ]

    for field_name, field_type in indexes_to_create:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_type,
                wait=True,
            )
            print(f"✅ Created index for '{field_name}' ({field_type})")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️  Index for '{field_name}' already exists")
            else:
                print(f"❌ Failed to create index for '{field_name}': {e}")

    print("\n✅ All indexes created successfully!")

if __name__ == "__main__":
    create_indexes()
