"""
DocuMind — Intelligent Document Intelligence Platform
------------------------------------------------------
Streamlit frontend that talks to the FastAPI backend.
Upload documents, ask questions, compare, summarise,
and export a full PDF report.
"""

import streamlit as st
import requests
import uuid
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
)

BACKEND_URL = "http://localhost:8000"

# ── SESSION STATE ─────────────────────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction.
# st.session_state persists variables across reruns — like memory.

if "session_id"     not in st.session_state:
    st.session_state.session_id    = str(uuid.uuid4())[:8]
if "chat_history"   not in st.session_state:
    st.session_state.chat_history  = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "qa_pairs"       not in st.session_state:
    st.session_state.qa_pairs      = []

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🧠 DocuMind")
st.caption("Intelligent Document Intelligence Platform — powered by Gemini + LangChain + ChromaDB")
st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Manager")
    st.caption(f"Session ID: `{st.session_state.session_id}`")

    # File uploader
    uploaded = st.file_uploader(
        "Upload documents:",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload one or more PDF, Word, or text files"
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {f.name}..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/upload/{st.session_state.session_id}",
                            files={"file": (f.name, f.getvalue(), f.type)}
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.uploaded_files.append(f.name)
                            st.success(f"✅ {f.name} — {data['chunks']} chunks indexed")
                        else:
                            st.error(f"❌ Failed to upload {f.name}")
                    except Exception as e:
                        st.error(f"❌ Backend not running: {e}")

    # Uploaded files list
    if st.session_state.uploaded_files:
        st.subheader("📄 Indexed Documents")
        for fname in st.session_state.uploaded_files:
            st.write(f"• {fname}")

    st.divider()

    # Clear session
    if st.button("🗑️ Clear Session", type="secondary"):
        try:
            requests.delete(f"{BACKEND_URL}/session/{st.session_state.session_id}")
        except Exception:
            pass
        st.session_state.session_id     = str(uuid.uuid4())[:8]
        st.session_state.chat_history   = []
        st.session_state.uploaded_files = []
        st.session_state.qa_pairs       = []
        st.rerun()

    st.divider()
    st.caption("Built with FastAPI · LangChain · ChromaDB · Gemini")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Ask Questions",
    "📋 Auto Summary",
    "⚖️ Compare Documents",
    "📊 Export Report"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Q&A CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask questions across your documents")
    st.caption("DocuMind answers from your documents only — every answer includes source citations.")

    if not st.session_state.uploaded_files:
        st.info("👈 Upload documents in the sidebar to get started.")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🧠"):
                    st.write(msg["content"])
                    if msg.get("citations"):
                        with st.expander("📎 Sources"):
                            seen = set()
                            for c in msg["citations"]:
                                key = f"{c['source']} — Page {c['page']}"
                                if key not in seen:
                                    st.caption(f"• {key}")
                                    seen.add(key)

        # Question input
        question = st.chat_input("Ask a question about your documents...")

        if question:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user", "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/ask/{st.session_state.session_id}",
                            json={
                                "question": question,
                                "chat_history": st.session_state.chat_history[:-1]
                            }
                        )
                        data = response.json()
                        answer = data["answer"]
                        citations = data.get("citations", [])

                        st.write(answer)

                        if citations:
                            with st.expander("📎 Sources"):
                                seen = set()
                                for c in citations:
                                    key = f"{c['source']} — Page {c['page']}"
                                    if key not in seen:
                                        st.caption(f"• {key}")
                                        seen.add(key)

                        # Save to history and qa_pairs
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations
                        })
                        st.session_state.qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "citations": citations
                        })

                    except Exception as e:
                        st.error(f"❌ Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AUTO SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Auto-generate executive summaries")
    st.caption("Select a document and get an instant structured summary.")

    if not st.session_state.uploaded_files:
        st.info("👈 Upload documents in the sidebar first.")
    else:
        selected = st.selectbox(
            "Select a document to summarise:",
            st.session_state.uploaded_files
        )

        if st.button("📋 Generate Summary", type="primary"):
            with st.spinner("Reading document and generating summary..."):
                try:
                    response = requests.get(
                        f"{BACKEND_URL}/summary/{st.session_state.session_id}/{selected}"
                    )
                    data = response.json()
                    st.subheader(f"Summary: {selected}")
                    st.write(data["summary"])

                    # Add to export pairs
                    st.session_state.qa_pairs.append({
                        "question": f"Executive Summary of {selected}",
                        "answer": data["summary"],
                        "citations": []
                    })
                    st.success("✅ Summary added to export report.")

                except Exception as e:
                    st.error(f"❌ Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Compare documents on a topic")
    st.caption("Enter a topic and DocuMind will compare how your documents cover it.")

    if len(st.session_state.uploaded_files) < 2:
        st.info("👈 Upload at least 2 documents to use comparison.")
    else:
        topic = st.text_input(
            "Topic to compare:",
            placeholder="e.g. data privacy, machine learning methods, conclusions..."
        )

        if st.button("⚖️ Compare", type="primary", disabled=not topic.strip()):
            with st.spinner("Comparing documents..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/compare/{st.session_state.session_id}",
                        json={"topic": topic}
                    )
                    data = response.json()
                    st.subheader(f"Comparison: {topic}")
                    st.write(data["comparison"])

                    st.session_state.qa_pairs.append({
                        "question": f"Document Comparison on: {topic}",
                        "answer": data["comparison"],
                        "citations": []
                    })
                    st.success("✅ Comparison added to export report.")

                except Exception as e:
                    st.error(f"❌ Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Export your session as a PDF report")
    st.caption("Download a professional PDF containing all your Q&A, summaries and comparisons.")

    if not st.session_state.qa_pairs:
        st.info("Ask some questions or generate summaries first — they'll appear here for export.")
    else:
        st.write(f"**{len(st.session_state.qa_pairs)} items ready to export:**")
        for i, pair in enumerate(st.session_state.qa_pairs, 1):
            st.write(f"{i}. {pair['question'][:80]}...")

        session_name = st.text_input(
            "Report title:",
            value=f"DocuMind Report — {datetime.now().strftime('%d %b %Y')}"
        )

        if st.button("📥 Download PDF Report", type="primary"):
            with st.spinner("Generating PDF..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/export",
                        json={
                            "session_name": session_name,
                            "qa_pairs": st.session_state.qa_pairs
                        }
                    )
                    st.download_button(
                        label="⬇️ Click to Download Report",
                        data=response.content,
                        file_name="documind_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")