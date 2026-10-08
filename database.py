import os
import chromadb

class VectorDBManager:
    def __init__(self, storage_path="./chroma_db", collection_name="brisurf_api_codebase"):
        self.storage_path = storage_path
        self.collection_name = collection_name
        
        # Initialize local persistent storage on your drive
        self.chroma_client = chromadb.PersistentClient(path=self.storage_path)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

    def get_collection(self):
        return self.collection

    def get_total_chunks(self):
        return self.collection.count()

    def get_all_metadata(self):
        total_chunks = self.get_total_chunks()
        if total_chunks == 0:
            return []
        # Pull 100% of metadata records by lifting the default 100-row limit barrier
        result = self.collection.get(include=["metadatas"], limit=total_chunks)
        return result.get("metadatas", []) if result else []

    def delete_file_records(self, file_path):
        """Clears out old chunk entries matching a file path before a rewrite."""
        self.collection.delete(where={"file_path": file_path})

    def add_segments(self, ids, embeddings, documents, metadatas):
        """Batch inserts vectorized structural code snippets into the index."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            # FIXED: Changed 'metas' to 'metadatas' to match the parameter name
            metadatas=metadatas if isinstance(metadatas, list) else [metadatas] * len(ids)
        )


    def query_semantic_matches(self, query_vector, n_results=5):
        """Searches vector space coordinate matrix layers for relevant code chunks."""
        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
