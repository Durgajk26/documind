"""
PDF Report Exporter
--------------------
Generates a professional PDF report of the Q&A session.
Uses reportlab — a pure Python PDF generation library.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import io
from datetime import datetime


def generate_report(session_name: str, qa_pairs: list[dict]) -> bytes:
    """
    Generate a PDF report from Q&A pairs.

    qa_pairs format:
    [{"question": "...", "answer": "...", "citations": [...]}]

    Returns PDF as bytes (for Streamlit download button).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=24,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=6,
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#666666"),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1A1A1A"),
        spaceBefore=16,
        spaceAfter=8,
    )
    answer_style = ParagraphStyle(
        "Answer",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#333333"),
        spaceAfter=8,
        leading=16,
    )
    citation_style = ParagraphStyle(
        "Citation",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#666666"),
        leftIndent=20,
        spaceAfter=4,
    )

    story = []

    # Header
    story.append(Paragraph("DocuMind", title_style))
    story.append(Paragraph("Document Intelligence Report", subtitle_style))
    story.append(Paragraph(
        f"Session: {session_name}  •  Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A1A1A")))
    story.append(Spacer(1, 20))

    # Q&A pairs
    for i, pair in enumerate(qa_pairs, 1):
        story.append(Paragraph(f"Q{i}: {pair['question']}", question_style))
        story.append(Paragraph(pair["answer"].replace("\n", "<br/>"), answer_style))

        if pair.get("citations"):
            story.append(Paragraph("Sources:", citation_style))
            seen = set()
            for c in pair["citations"]:
                key = f"{c['source']} p.{c['page']}"
                if key not in seen:
                    story.append(Paragraph(f"• {key}", citation_style))
                    seen.add(key)

        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#DDDDDD")))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()