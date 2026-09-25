from pathlib import Path
import fitz

def load_pdfs(data_dir):
    documents = []
    for path in sorted(Path(data_dir).glob("*.pdf")):
        pdf = fitz.open(path)
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                documents.append({
                    "text": text,
                    "source": path.name,
                    "page": page_no
                })
        pdf.close()
    return documents
