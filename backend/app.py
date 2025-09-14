import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
from qdrant_client import QdrantClient
from services.llm_service import generate_llm_response
from services.qdrant_service import search_text, insert_point, retrieve_llm_responses_by_user,embed_text
from pipeline.ai_pipeline import ai_pipeline
from pipeline.ml_processing import _pipelines
from utils.fetch import fetch_tickets
import os
from dotenv import load_dotenv
load_dotenv()
# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_COLLECTION_2 = os.getenv("QDRANT_COLLECTION_2")
QDRANT_COLLECTION_3 = os.getenv("QDRANT_COLLECTION_3")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", 3))

# CORS (frontend origin)
VECTOR_NAME = QDRANT_VECTOR_NAME
FRONTEND_ORIGIN=os.getenv("FRONTEND_ORIGIN")

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Qdrant client with error handling
try:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
except Exception as e:
    print(f"Warning: Could not initialize Qdrant client: {e}")
    qdrant_client = None

@app.route('/input', methods=['POST'])
def handle_input():
    """
    Handles incoming JSON data from the frontend, processes it through the AI pipeline, and returns the results.
    Used when the application calls the /input endpoint with a POST request containing JSON data.
    The "+ Upload Json" button on the frontend triggers this endpoint.
    """
    try:
        data = request.json
        if not isinstance(data, list):
            return jsonify({"error": "Invalid JSON format. Expected a list of objects."}), 400
            
        processed_data = ai_pipeline(data)
        
        if any("error" in item for item in processed_data):
            # If any item failed to process, return a 500
            return jsonify(processed_data), 500
        
        return jsonify(processed_data), 200

    except Exception as e:
        # Catch any exceptions during request parsing or pipeline execution
        print("An error occurred:")
        traceback.print_exc() # This will print the full traceback to your terminal
        return jsonify({"error": "Internal server error. Check the server logs for details."}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handles chat requests from the frontend, interacts with the LLM and Qdrant to generate responses.
    Expects a JSON payload with 'text' and 'user_id'.
    Returns the LLM response and cited URLs.
    """
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        user_id = data.get("user_id", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        top_k = 3
        search_results = search_text(query=text, k=top_k)
        # print(search_results) -> for debugging

        try : 
            previous_responses = retrieve_llm_responses_by_user(client=qdrant_client,user_id=user_id,input_text=text)
            # print("we got the prev responses") -> for debugging
        except Exception as e:
            previous_responses=[]
            # print("cant load prev responses endpoint error:", e)

        if previous_responses:
            llm_output = generate_llm_response(
                user_text=text,
                responses=previous_responses,
                results=search_results
            )
        else:
            
            llm_output = generate_llm_response(
                user_text=text,
                responses=[],  
                results=search_results,
            )

        llm_response = llm_output.get("LLM_Response", "")
        cited_urls = llm_output.get("Cited_URLs", [])

        metadata = {
                "user_id": user_id,
                "user_query": text,
                "llm_response": llm_response,
                "cited_urls": cited_urls,
                "timestamp": str(uuid.uuid1().time) 
            }
        
        try:
            insert_point(
                client=qdrant_client,
                user_id=user_id,
                input_text=text,
                llm_response=llm_response
            )
        except Exception as insert_error:
            print("Failed to insert Qdrant point:", traceback.format_exc())
            print("Insert error details:", str(insert_error))

        response_data = {
            "user_id": user_id,
            "LLM_Response": llm_response,
            "Cited_URLs": cited_urls
        }
        return jsonify(response_data)
        
    except Exception as e:
        print("Chat endpoint error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=['GET'])
def health_check():
    """
    Health check endpoint to verify that the server is running.
    Often helps in debugging. Returns a simple JSON response."""
    return jsonify({"status": "healthy", "message": "Server is running"})

@app.route("/fetch",  methods=['GET', 'POST'])
def get_tickets(limit: int = 30):
    """
    Fetches ticket points from the Qdrant collection and returns them to the frontend.
    /fetch endpoint can be called with a GET or POST request.
    Accepts an optional 'limit' query parameter to specify the number of tickets to fetch (default is 30).
    """
    try:
        tickets = fetch_tickets(limit)
        return jsonify(tickets)
    except Exception as e:
        print(f"Error fetching tickets: {e}")

if __name__ == '__main__':
    """
    Backend runs on port 8081
    """
    app.run(port=8081, debug=True)


