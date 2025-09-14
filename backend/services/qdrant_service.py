import datetime
import os
import json
import traceback
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import FieldCondition, PayloadSchemaType
from qdrant_client import models
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_COLLECTION_2 = os.getenv("QDRANT_COLLECTION_2")
QDRANT_COLLECTION_3 = os.getenv("QDRANT_COLLECTION_3")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME")


VECTOR_NAME = QDRANT_VECTOR_NAME
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

top_k = 3

def embed_text(text: str) -> List[float]:
    """
    Generates an embedding vector for the given text using OpenAI text-3-small embedding model. 
    vector size = 1536
    """
    
    resp = openai_client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text)
    
    return resp.data[0].embedding

def search_text(query: str, k: int) -> List[Dict[str, Any]]:
    """
    Searches the Qdrant collection for the top-k(5 in our case) most similar documents to the query text.
    k is set so as to not overload the LLM with too many documents.
    Searched in QDRANT_COLLECTION (Primary collection for RAG Pipeline) ie atlan-web-docs-3
    """
    try:
        if not query:
            raise ValueError("Query text is required")

        
        top_k = k 

        
        vec = embed_text(query)

        
        hits = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            using=VECTOR_NAME,
            limit=top_k,
            with_payload=True
        ).points

        
        results: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            results.append({
                "id": str(h.id),
                "score": float(h.score) if h.score is not None else 0.0,
                "text": payload.get("text", ""),
                "url": payload.get("url", ""),
                "url_id": payload.get("url_id", ""),
                "parent_id": payload.get("parent_id", "")
            })

        return results

    except Exception as e:
        print(f"Error in search function: {e}")
        return []



def insert_point(client: QdrantClient, user_id: str, input_text: str, llm_response: str):
    """
    Inserts a new point into the Qdrant collection for chat history.
    Each point contains user_id, input_text, llm_response, and created_at timestamp.
    Pushes into qdrant collection QDRANT_COLLECTION_3 (chat-history-with-llm-per-user)"""
    print(f"Inserting chat history for user_id: {user_id}")

    
    point_id = str(uuid.uuid4())
    vector = embed_text(input_text)

    payload = {
        "user_id": user_id,                
        "input_text": input_text,          
        "llm_response": llm_response,      
        "created_at": datetime.datetime.utcnow().isoformat()
    }

    point = PointStruct(id=point_id, vector=vector, payload=payload)

    client.upsert(
        collection_name=QDRANT_COLLECTION_3,
        wait=True,      
        points=[point],
    )

    print(f"Inserted chat history for user_id: {user_id}, point_id: {point_id}")

def retrieve_llm_responses_by_user(client: QdrantClient, user_id: str, input_text: str) -> List[str]:
    """
    Retrieves the top 3 most relevant LLM responses for a given user_id based on the input_text.
    Searches in QDRANT_COLLECTION_3 (chat-history-with-llm-per-user)"""
    query_vector = embed_text(input_text)

    results = client.search(
        collection_name=QDRANT_COLLECTION_3,
        query_vector=query_vector,
        limit=3,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id)
            )]
        )
    )

    return [r.payload["llm_response"] for r in results if "llm_response" in r.payload]

