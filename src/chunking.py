def chunk_documents(documents, chunk_size=1000, overlap=150):
    chunks = []
    for doc in documents:
        text = " ".join(doc["text"].split())
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "page": doc["page"]
                })
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks
