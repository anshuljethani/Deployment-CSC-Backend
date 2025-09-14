import os
import uuid
import datetime
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, Field, PointStruct
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_COLLECTION_2 = os.getenv("QDRANT_COLLECTION_2")
QDRANT_COLLECTION_3 = os.getenv("QDRANT_COLLECTION_3")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME")


COLLECTION_NAME = QDRANT_COLLECTION_2
print(COLLECTION_NAME)
print(QDRANT_URL)
print(QDRANT_API_KEY)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
def push_ticket_point(
    ticket_id: str,
    subject: str,
    body: str,
    priority: str,
    topics: str,
    keywords: str,
    sentiment: str,
    created_at: datetime.datetime,
    vector: list[float]
):
    """
    Pushes a ticket point to the Qdrant collection. In this case the collection is QDRANT_COLLECTION_2 (BULK TICKETS STORAGE)
    Input parameters:
    - ticket_id: str   
    - subject: str
    - body: str
    - priority: str
    - topics: str
    - keywords: str
    - sentiment: str
    - created_at: datetime
    - vector: list of floats (embedding vector)
    """
    print(f"Pushing point with ticket_id: {ticket_id}")
    
    point_id = str(uuid.uuid4())

    point_payload = {
        "id": ticket_id,
        "subject": subject,
        "body": body,
        "priority": priority,
        "topics": topics,
        "keywords": keywords,
        "sentiment": sentiment,
        "created_at": created_at.isoformat()  
    }
    
    point_to_insert = PointStruct(
        id=point_id,  
        vector=vector,
        payload=point_payload
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[point_to_insert],
    )
    print(f"Successfully pushed point for ticket_id: {ticket_id}")