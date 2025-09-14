import os
import json
from qdrant_client import QdrantClient
from fastapi.responses import JSONResponse
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_2 = os.getenv("QDRANT_COLLECTION_2")

def fetch_tickets(limit=30):
    """
    Fetches ticket points from the Qdrant collection QDRANT_COLLECTION_2 (Bulk_Tickets-2)
    Returns results to frontend"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION_2,
        limit=limit,
        with_payload=True
    )

    results = []
    for p in points:
        payload = p.payload or {}
        results.append({
            "id": payload.get("id", ""),
            "subject": payload.get("subject", ""),
            "body": payload.get("body", ""),
            "priority": payload.get("priority", ""),
            "topics": payload.get("topics", ""),
            "keywords": payload.get("keywords", ""),
            "sentiment": payload.get("sentiment", ""),
            "created_at": payload.get("created_at", "")
        })

    return results
