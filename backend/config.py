import os
from dotenv import load_dotenv

load_dotenv()
# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_COLLECTION_2 = os.getenv("QDRANT_COLLECTION_2")
QDRANT_COLLECTION_3 = os.getenv("QDRANT_COLLECTION_3")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", 3))

# CORS (frontend origin)
VECTOR_NAME = "doc-dense-vector"
FRONTEND_ORIGIN=os.getenv("FRONTEND_ORIGIN")