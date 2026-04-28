"""
RAG Pipeline
-------------
Retrieval-Augmented Generation using LangChain + Google Gemini.

RAG = Retrieve relevant document chunks → Augment the prompt with them
      → Generate an answer grounded in the documents.

This is what makes it different from a basic chatbot:
the AI can only answer from YOUR documents, not hallucinate.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from backend.vectorstore import search_documents

load_dotenv()

# ── Gemini model ──────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,   # Lower = more factual, less creative
)

SYSTEM_PROMPT = """You are DocuMind, an intelligent document analysis assistant.

Your job is to answer questions based ONLY on the provided document excerpts.

Rules:
- Only use information from the provided context
- Always cite your sources (filename and page number)
- If the answer is not in the documents, say so clearly
- Be concise but thorough
- Format your response clearly with the answer first, then citations
"""


def answer_question(
    session_id: str,
    question: str,
    chat_history: list[dict]
) -> dict:
    """
    Main RAG function:
    1. Search ChromaDB for relevant chunks
    2. Build a prompt with context + chat history
    3. Send to Gemini
    4. Return answer with citations

    chat_history format: [{"role": "user/assistant", "content": "..."}]
    """

    # Step 1: Retrieve relevant chunks
    chunks = search_documents(session_id, question, n_results=5)

    if not chunks:
        return {
            "answer": "No documents have been uploaded yet. Please upload documents first.",
            "citations": [],
            "chunks_used": 0
        }

    # Step 2: Format context from chunks
    context_parts = []
    citations = []

    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
        citations.append({
            "source": chunk["source"],
            "page":   chunk["page"],
            "excerpt": chunk["text"][:150] + "..."
        })

    context = "\n\n".join(context_parts)

    # Step 3: Build chat history context
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # Last 3 exchanges
        for msg in recent:
            role = "User" if msg["role"] == "user" else "DocuMind"
            history_text += f"{role}: {msg['content']}\n"

    # Step 4: Build full prompt
    user_prompt = f"""
Previous conversation:
{history_text if history_text else "None"}

Document excerpts:
{context}

Question: {question}

Please answer based on the documents above. Cite your sources.
"""

    # Step 5: Send to Gemini
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    return {
        "answer":      response.content,
        "citations":   citations,
        "chunks_used": len(chunks)
    }


def generate_summary(session_id: str, filename: str) -> str:
    """
    Auto-generate an executive summary for an uploaded document.
    Searches for broad content and asks Gemini to summarise.
    """
    chunks = search_documents(session_id, "main topic summary overview", n_results=8)
    relevant = [c for c in chunks if c["source"] == filename]

    if not relevant:
        relevant = chunks[:5]

    context = "\n\n".join([c["text"] for c in relevant])

    prompt = f"""Based on these excerpts from '{filename}', write a concise executive summary (3-5 bullet points) covering:
- Main topic/purpose
- Key findings or arguments  
- Important conclusions

Excerpts:
{context}"""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content


def compare_documents(session_id: str, topic: str) -> str:
    """
    Compare multiple documents on a specific topic.
    """
    chunks = search_documents(session_id, topic, n_results=8)

    # Group by source
    by_source = {}
    for chunk in chunks:
        src = chunk["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(chunk["text"])

    if len(by_source) < 2:
        return "Need at least 2 documents to compare. Please upload more documents."

    context_parts = []
    for source, texts in by_source.items():
        context_parts.append(f"**{source}:**\n" + "\n".join(texts[:2]))

    context = "\n\n".join(context_parts)

    prompt = f"""Compare these documents on the topic of '{topic}':

{context}

Provide a structured comparison showing:
- Similarities
- Differences  
- Which document covers the topic more thoroughly"""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content