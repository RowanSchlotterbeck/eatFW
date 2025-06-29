from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Create the FastAPI app instance
app = FastAPI()

# Define the origins that are allowed to access the API
# For development, this will be your Next.js frontend
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

# Define the data model for the request body
class Query(BaseModel):
    text: str

@app.post("/api/ask")
async def ask_question(query: Query):
    """
    Receives a question from the user, processes it,
    and returns an AI-generated answer.
    """
    # For now, we'll just echo the question back
    # This is where the RAG will go 
    print(f"Received query: {query.text}")
    return {"answer": f"You asked: '{query.text}'. The AI response will go here."} 