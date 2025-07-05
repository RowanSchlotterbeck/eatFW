import ollama
import chromadb
import json
import os
import argparse
import sys

# ---THIS CHUNCK OF CODE WAS IS THE SOLUTION TO THE PATH PROBLEM PREVIOUSLY ENCOUNTERED---
# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Use absolute paths to ensure the script can be run from anywhere
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(SCRIPT_DIR, "chroma_db"))
print(f"Using ChromaDB path: {CHROMA_PATH}")
DATA_PATH = os.path.join(SCRIPT_DIR, "dfw_restaurants.json")
COLLECTION_NAME = "restaurants"
EMBEDDING_MODEL = "nomic-embed-text"


def query_restaurants(query: str, top_k: int = 5) -> None:
    """
    Performs a semantic search over the restaurant collection and prints the
    top_k most relevant restaurants for the provided query.
    """
    # Init ChromaDB client and collection
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("The collection is empty. Run the script without --query to ingest data first.")
        return

    # Generate embedding for the query
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    query_embedding = response["embedding"]

    # Search the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nTop {top_k} results for query: '{query}'\n" + "-" * 60)
    for idx, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        print(f"{idx}. {meta.get('name', 'Unknown Restaurant')} (distance: {dist:.4f})")
        print(doc)
        print("-" * 60)


def ingest_data():
    """
    Reads restaurant data from a JSON file, generates embeddings,
    and stores them in a ChromaDB collection.
    """
    print("Starting data ingestion...")

    # Init ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Get or create the collection (wipe if it already exists to avoid stale data)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {collection.count()} documents – deleting to ingest fresh data…")
        # Delete and recreate the collection so IDs reset and old docs disappear
        client.delete_collection(name=COLLECTION_NAME)
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

        # Create a richer document for embedding
        reviews = restaurant.get('reviews', [])
        avg_rating = (
            round(sum(r.get('rating', 0) for r in reviews) / len(reviews), 2)
            if reviews else "N/A"
        )

        menu_items = restaurant.get('menu', [])
        menu_highlights = (
            ", ".join([f"{item.get('item')} (${'{:.2f}'.format(item.get('price', 0))})" for item in menu_items[:5]])
            if menu_items else "N/A"
        )

        document = (
            f"Name: {restaurant.get('name', 'N/A')}\n"
            f"Cuisine: {restaurant.get('cuisine', 'N/A')}\n"
            f"Price Range: {restaurant.get('price_range', 'N/A')}\n"
            f"Address: {restaurant.get('address', 'N/A')}\n"
            f"Average Rating: {avg_rating}\n"
            f"Menu Highlights: {menu_highlights}\n"
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
    parser = argparse.ArgumentParser(
        description="Ingest restaurant data into ChromaDB or query it for relevant restaurants.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Natural language query to search for relevant restaurants.",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=5,
        help="Number of top results to return when querying (default: 5).",
    )
    args = parser.parse_args()

    # Ensure Ollama is running
    try:
        ollama.list()
    except Exception:
        print("Error: Ollama is not running. Please start Ollama and try again.")
        print("You can start Ollama by running 'ollama serve' in your terminal.")
        sys.exit(1)

    if args.query:
        query_restaurants(args.query, args.top_k)
    else:
        ingest_data() 