"""
DocuMind FastAPI Backend
-------------------------
REST API that the Streamlit frontend talks to.
Handles document upload, Q&A, summarisation, and comparison.
"""

import os
import io
import uuid
import pypdf
from docx import Document as DocxDocument
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.vectorstore import add_documents, delete_session, list_sessions
from backend.rag import answer_question, generate_summary, compare_documents
from backend.exporter import generate_report
from fastapi.responses import Response

load_dotenv()

app = FastAPI(title="DocuMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes, filename: str) -> list[dict]:
    reader = pypdf.PdfReader(io.BytesIO(content))
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            # Split page into ~500 word chunks
            words = text.split()
            for j in range(0, len(words), 500):
                chunk_text = " ".join(words[j:j+500])
                if chunk_text.strip():
                    chunks.append({
                        "text":   chunk_text,
                        "source": filename,
                        "page":   i + 1
                    })
    return chunks


def extract_text_from_docx(content: bytes, filename: str) -> list[dict]:
    doc = DocxDocument(io.BytesIO(content))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), 500):
        chunk_text = " ".join(words[i:i+500])
        if chunk_text.strip():
            chunks.append({
                "text":   chunk_text,
                "source": filename,
                "page":   (i // 500) + 1
            })
    return chunks


def extract_text_from_txt(content: bytes, filename: str) -> list[dict]:
    text = content.decode("utf-8", errors="ignore")
    words = text.split()
    chunks = []
    for i in range(0, len(words), 500):
        chunk_text = " ".join(words[i:i+500])
        if chunk_text.strip():
            chunks.append({
                "text":   chunk_text,
                "source": filename,
                "page":   (i // 500) + 1
            })
    return chunks


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "DocuMind API is running", "version": "1.0.0"}


@app.post("/upload/{session_id}")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename

    if filename.endswith(".pdf"):
        chunks = extract_text_from_pdf(content, filename)
    elif filename.endswith(".docx"):
        chunks = extract_text_from_docx(content, filename)
    elif filename.endswith(".txt"):
        chunks = extract_text_from_txt(content, filename)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    count = add_documents(session_id, chunks)
    return {"message": f"Uploaded {filename}", "chunks": count, "filename": filename}


class QuestionRequest(BaseModel):
    question: str
    chat_history: list[dict] = []


@app.post("/ask/{session_id}")
def ask_question(session_id: str, request: QuestionRequest):
    result = answer_question(session_id, request.question, request.chat_history)
    return result


@app.get("/summary/{session_id}/{filename}")
def get_summary(session_id: str, filename: str):
    summary = generate_summary(session_id, filename)
    return {"summary": summary, "filename": filename}


class CompareRequest(BaseModel):
    topic: str


@app.post("/compare/{session_id}")
def compare(session_id: str, request: CompareRequest):
    result = compare_documents(session_id, request.topic)
    return {"comparison": result}


class ExportRequest(BaseModel):
    session_name: str
    qa_pairs: list[dict]


@app.post("/export")
def export_report(request: ExportRequest):
    pdf_bytes = generate_report(request.session_name, request.qa_pairs)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=documind_report.pdf"}
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    delete_session(session_id)
    return {"message": "Session cleared"}