import ollama
import chromadb
import json
import os

# ---THIS CHUNCK OF CODE WAS IS THE SOLUTION TO THE PATH PROBLEM PREVIOUSLY ENCOUNTERED---
# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Use absolute paths to ensure the script can be run from anywhere
CHROMA_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
DATA_PATH = os.path.join(SCRIPT_DIR, "resturant-data.json")
COLLECTION_NAME = "restaurants"
EMBEDDING_MODEL = "nomic-embed-text"

def ingest_data():
    """
    Reads restaurant data from a JSON file, generates embeddings,
    and stores them in a ChromaDB collection.
    """
    print("Starting data ingestion...")

    # Init ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Get or create the collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Load restaurant data
    try:
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DATA_PATH}. Please check for syntax errors.")
        return

    # Process and store each restaurant
    for i, restaurant in enumerate(data):

        # Create a single document for embedding
        document = (
            f"Name: {restaurant.get('name', 'N/A')}\\n"
            f"Cuisine: {restaurant.get('cuisine', 'N/A')}\\n"
            f"Summary: {restaurant.get('summary', 'N/A')}"
        )
        
        # Generate embedding
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=document
        )
        embedding = response["embedding"]
        
        # Store in Chroma
        collection.add(
            ids=[str(i)],  # IDs must be strings
            embeddings=[embedding],
            documents=[document],
            metadatas=[{"name": restaurant.get('name', 'N/A')}]
        )
        print(f"  - Ingested '{restaurant.get('name', 'Unknown Restaurant')}'")

    print("\\nData ingestion complete.")
    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    # Ensure Ollama is running
    try:
        ollama.list()
        print("Ollama is running.")
        ingest_data()
    except Exception:
        print("Error: Ollama is not running. Please start Ollama and try again.")
        print("You can start Ollama by running 'ollama serve' in your terminal.") 