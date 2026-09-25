from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        ).tolist()
