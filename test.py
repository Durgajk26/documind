import pypdf

reader = pypdf.PdfReader("CST4225_CW1_Combined(1).pdf")
for i, page in enumerate(reader.pages[:3]):
    text = page.extract_text()
    print(f"--- PAGE {i+1} ---")
    print(text[:300] if text else "NO TEXT EXTRACTED")
    print()