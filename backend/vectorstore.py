"""
Vector Store Manager
---------------------
Handles storing document chunks as embeddings in ChromaDB.
Uses LangChain's Google embedding — compatible with the latest API.
"""

import os
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── ChromaDB client (persistent = saves to disk) ──────────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")

# LangChain embedding — uses the correct google-genai API
embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def get_or_create_collection(session_id: str):
    """Each session gets its own collection in ChromaDB."""
    return client.get_or_create_collection(name=f"session_{session_id}")


def add_documents(session_id: str, chunks: list[dict]):
    """Store document chunks with their embeddings in ChromaDB."""
    collection = get_or_create_collection(session_id)

    texts     = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    ids       = [f"{session_id}_{i}" for i in range(len(chunks))]

    # Generate embeddings using LangChain
    embeddings = embedder.embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    return len(chunks)


def search_documents(session_id: str, query: str, n_results: int = 5) -> list[dict]:
    """Search for the most relevant chunks for a given query."""
    collection = get_or_create_collection(session_id)
    count = collection.count()
    if count == 0:
        return []

    n_results = min(n_results, count)

    # Generate query embedding
    query_embedding = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text":   doc,
            "source": results["metadatas"][0][i]["source"],
            "page":   results["metadatas"][0][i]["page"],
        })
    return chunks


def delete_session(session_id: str):
    """Clean up a session's collection from ChromaDB."""
    try:
        client.delete_collection(f"session_{session_id}")
    except Exception:
        pass


def list_sessions() -> list[str]:
    """List all active sessions."""
    collections = client.list_collections()
    return [c.name.replace("session_", "") for c in collections]