import logging
import chromadb
from chromadb.utils import embedding_functions
from gateway.settings import load_gateway_settings

log = logging.getLogger("gateway.retriever")
cfg = load_gateway_settings()

class LocalRetriever:
    def __init__(self, collection_name: str = "indian_context"):
        self.client = chromadb.PersistentClient(path=str(cfg.chroma_db_dir))
        
        # Swapped to SentenceTransformer to avoid Chroma's ONNX download timeouts
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.ef
        )
        
    def search(self, query: str) -> str:
        if not query.strip():
            return ""
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=cfg.retrieval_top_k
            )
            
            contexts = []
            if results['documents'] and results['distances']:
                for doc, dist in zip(results['documents'][0], results['distances'][0]):
                    # Lower distance means higher similarity
                    if dist <= cfg.retrieval_distance_threshold:
                        contexts.append(doc)
                        
            return "\n\n".join(contexts)
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return ""
