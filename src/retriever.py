class Retriever:
    def __init__(self, embedder, vector_store, top_k=5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question):
        query_embedding = self.embedder.encode([question])[0]
        result = self.vector_store.search(query_embedding, self.top_k)
        items = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc, meta, distance in zip(docs, metas, distances):
            items.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "distance": distance
            })
        return items
