from typing import Dict, Any, List
from sentence_transformers import CrossEncoder
from src.vector_store import HybridVectorStore
from src.llm import LLMInterface
from langchain_core.documents import Document

class ResearchRAGEngine:
    """
    RAG Engine with 2-Stage Retrieval (Bi-Encoder + Cross-Encoder Re-ranker).
    ML Research Note: Cross-Encoders are more computationally expensive but
    provide far more accurate relevance scoring by considering the query and 
    document jointly.
    """
    def __init__(self, vector_store: HybridVectorStore, llm_interface: LLMInterface, rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_store = vector_store
        self.llm = llm_interface.get_llm()
        self.reranker = CrossEncoder(rerank_model)
        
    def _rerank_documents(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        """
        Re-ranks a list of candidate documents based on cross-encoder scores.
        """
        if not documents:
            return []
            
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Sort documents by scores in descending order
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [documents[i] for i in ranked_indices[:top_k]]

    def query(self, question: str, alpha: float = 0.5) -> Dict[str, Any]:
        """
        Executes a 2-stage RAG pipeline.
        1. Retrieval: Hybrid Dense + Sparse Search.
        2. Re-ranking: Cross-Encoder sorting.
        """
        # 1. Retrieval (Hybrid)
        candidate_docs = self.vector_store.hybrid_search(question, k=15, alpha=alpha)
        
        # 2. Re-ranking (Cross-Encoder)
        reranked_docs = self._rerank_documents(question, candidate_docs, top_k=4)
        
        # 3. Augmentation (Context Construction)
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        prompt = f"Use the following context to answer: {context}\n\nQuestion: {question}\nAnswer:"
        
        # 4. Generation
        response = self.llm.invoke(prompt)
        
        return {
            "result": response,
            "source_documents": reranked_docs,
            "retrieval_metrics": {
                "num_candidates": len(candidate_docs),
                "num_reranked": len(reranked_docs)
            }
        }
