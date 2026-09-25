from src.config import *
from src.document_loader import load_pdfs
from src.chunking import chunk_documents
from src.embeddings import Embedder
from src.vector_store import VectorStore

def main():
    print("Loading documents...")
    documents = load_pdfs(DATA_DIR)
    if not documents:
        raise SystemExit("No PDF documents found in data/.")
    print(f"Loaded {len(documents)} pages.")

    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    embedder = Embedder(EMBEDDING_MODEL)
    embeddings = embedder.encode([c["text"] for c in chunks])

    store = VectorStore(VECTOR_DIR, COLLECTION_NAME)
    store.reset()
    store.add(chunks, embeddings)
    print(f"Stored {store.count()} chunks in ChromaDB.")
    print("Indexing complete.")

if __name__ == "__main__":
    main()
