import faiss
import pickle
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict
from src.config import get_settings

settings = get_settings()

class RAGService:
    """
    Retrieval Augmented Generation Service
    -Fast retreival using FAISS
    -Accurate reranking using CrossEncoder
    
    """
    def __init__(self):
        """Initialize Embedding model and, reranker and load FAISS Index"""
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(f"Loading re-ranker: {settings.RERANKER_MODEL}")
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        
        self.index = None
        
        self.metadata = []
        self._load_index()
        
    def _load_index(self):
        """Loading Faiss index from disk"""
        index_path = Path("data/vector_store/index.faiss")
        metadata_path = Path("data/vector_store/metadata.pkl")
        
        if not index_path.exists() or not metadata_path.exists():
            print("FAISS index or metadata not found. Please run the ingestion process first by Run 'python src/ingest.py'.")
            return
        
        try:
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path,"rb") as f:
                self.metadata = pickle.load(f)
            print("FAISS index and metadata loaded successfully.")
            
        except Exception as e:
            print(f"Error loading FAISS index or metadata: {e}")
            self.index = None
            self.metadata = []
    
    def search(self,query:str)->List[Dict]:
        """
        Two stage retrieval:
        - Retreive top-K using FAISS
        - Rerank using CrossEncoder
        Args:
            query (str): user query
        Returns:
            List[Dict]: list of top-K relevant documents with 'content' and 'metadata'
        
        """
        if not self.index or self.index.ntotal == 0:
            print("FAISS index is not loaded or empty.")
            return []
        #Stage 1: initial retreival based on top k
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)
        k_retreive = min(settings.TOP_K_RETRIEVE,self.index.ntotal)
        distances,indices = self.index.search(query_embedding,k_retreive)
        candidates = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                candidates.append(self.metadata[idx])
        if not candidates:
            return []
        #stage 2 : rerank using cross encoder
        pairs = [[query,doc["text"]] for doc in candidates]
        scores = self.reranker.predict(pairs)
        #sort by rerankerscores
        ranked_candidates = sorted(
            zip(candidates,scores),
            key=lambda x:x[1],
            reverse=True
        )
        
        #Return top-k after reranking 
        final_results = []
        for doc,score in ranked_candidates[:settings.TOP_K_RERANK]:
            result = doc.copy()
            result["relevance_score"]= float(score)
            final_results.append(result)
        return final_results
    
    def format_context_for_llm(self,results:List[Dict])->str:
        """
        Format retreived chunks into context string for llm prompt.
        Args:
        results:List of retreived chunks with metadata
        Returns:
            str: formatted context string with attached source
        """
        if not results:
            return "No relevant information found in the documents."
        formatted_chunks = []
        for i,doc in enumerate(results,1):
            source = doc.get("source","Unknown")
            text = doc.get("text","")
            
            # FIX: Check for page field (flattened structure)
            if "page" in doc:
                source_ref = f"{source}, Page {doc['page']}"
            else:
                source_ref = source
            formatted_chunks.append(f"Document {i}: {source_ref}\n{text}")
        return "\n\n".join(formatted_chunks)
    
def test_search():
    """testing retreival service"""
    rag = RAGService()
    
    
    test_queries = [
        "My domain was suspended, how do I fix it?",
        "How do I get a refund?",
        "What are the nameservers for DNS?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag.search(query)
         
        if results:
            print(f"Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                score = result.get("relevance_score", 0)
                source = result.get("source", "Unknown")
                page = result.get("page")  # FIX: Check if page exists
                
                # FIX: Better handling of page display
                if page:
                    page_str = f", Page {page}"
                else:
                    page_str = ""
                
                print(f"{i}. [{source}{page_str}] (Score: {score:.3f})")
                print(f"   {result['text'][:150]}...\n")
        else:
            print("No results found.\n")

if __name__ == "__main__":
    test_search()