from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"📥 Loading local embedding model ({model_name})...")
        # Runs 100% on your local CPU/RAM with zero internet dependency
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_list):
        """Converts raw code snippet lists into lists of numerical floats."""
        if isinstance(text_list, str):
            text_list = [text_list]
        return self.model.encode(text_list).tolist()

    def generate_single_embedding(self, text_string):
        """Converts a user chat question into vector coordinate format."""
        return self.model.encode(text_string).tolist()
