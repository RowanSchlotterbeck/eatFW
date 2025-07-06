from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import chromadb
import os
import json


# ---THIS CHUNCK OF CODE WAS IS THE SOLUTION TO THE PATH PROBLEM PREVIOUSLY ENCOUNTERED---
# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Use absolute paths to ensure the script can be run from anywhere
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(SCRIPT_DIR, "chroma_db"))
print(f"Using ChromaDB path: {CHROMA_PATH}")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
COLLECTION_NAME = "restaurants"


# FastAPI init
app = FastAPI()

# Define the origins that are allowed to access the API
# For development, this will be the Next.js frontend 
# I think this will evenutally get changed once put on the cloud
origins = [
    "http://localhost:3000",
]

# Add CORS middleware to the application
# Allows for the frontend to connect to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# Initialize clients once when the server starts
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    restaurant_collection = chroma_client.get_collection(COLLECTION_NAME)
    print("Successfully connected to ChromaDB and loaded collection.")
except Exception as e:
    print(f"Error connecting to ChromaDB or getting collection: {e}")
    print("Please ensure you have run ingest.py successfully before starting the server.")
    # Exit if we can't connect to the database
    exit()


# Define the data model for the request body
class Query(BaseModel):
    text: str

@app.post("/api/ask")
async def ask_question(query: Query):
    """
    Receives a question, retrieves relevant context from ChromaDB,
    and returns an AI-generated answer.
    """
    print(f"Received query: {query.text}")

    # 1. Generate an embedding for the user's query
    query_embedding = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=query.text
    )['embedding']

    # 2. Retrieve the most relevant documents from ChromaDB
    results = restaurant_collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Get the top 3 most relevant results
    )
    context_documents = results['documents'][0]
    context = "\\n\\n".join(context_documents)

    # 3. Augment the prompt with the retrieved context
    prompt_template = (
        "You are an expert on restaurants in Fort Worth, Texas specialising in Fort Worth dining.\\n\\n"
        "Using ONLY the information from the context provided below, return EXACTLY three restaurant recommendations.\\n"
        "FORMAT YOUR ENTIRE RESPONSE AS A VALID JSON ARRAY (no markdown or code fences) where each element has the keys: name, shortDescription, address, matchHighlights.\\n"
        "• Description >= 100 words and should naturally weave in relevant parts of the user's request.\\n"
        "• matchHighlights summarises which parts of the context matched the user's query.\\n\\n"
        "If the context is insufficient, do your best with the available information but still comply with the JSON format.\\n\\n"
        f"Context:\\n{context}\\n\\n"
        f"User's Question: {query.text}"
    )

    print("--- Augmented Prompt ---")
    print(prompt_template)
    print("------------------------")

    # 4. Generate the final response using the LLM
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a helpful restaurant assistant.'},
                {'role': 'user', 'content': prompt_template}
            ]
        )

        # Attempt to parse the LLM response as JSON so the frontend receives structured data
        try:
            restaurants = json.loads(response['message']['content'])
        except json.JSONDecodeError as json_err:
            print(f"JSON decode error: {json_err}")
            restaurants = [{
                "name": "Unknown",
                "shortDescription": response['message']['content'],
                "address": "",
                "matchHighlights": ""
            }]
    except Exception as e:
        print(f"Error calling Ollama chat API: {e}")
        restaurants = [{
            "name": "Error",
            "shortDescription": "Sorry, I'm having trouble connecting to the AI model right now.",
            "address": "",
            "matchHighlights": ""
        }]

    return {"restaurants": restaurants} 