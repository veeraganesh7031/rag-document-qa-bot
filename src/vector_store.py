import chromadb
from pathlib import Path

class VectorStore:
    def __init__(self, persist_dir, collection_name):
        self.client = chromadb.PersistentClient(path=str(Path(persist_dir)))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks, embeddings):
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"source": c["source"], "page": c["page"]}
                for c in chunks
            ]
        )

    def search(self, embedding, top_k=5):
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

    def count(self):
        return self.collection.count()
