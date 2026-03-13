from typing import List, Optional, Tuple
import os
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

class HybridVectorStore:
    """
    Advanced Vector Store supporting Hybrid Search (Dense + Sparse).
    ML Research Note: Combining semantic (Dense) and keyword (Sparse) 
    retrieval often yields higher robustness across diverse query types.
    """
    def __init__(self, collection_name: str = "neuro_research", persist_directory: str = "./data/chroma"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        self.bm25 = None
        self.corpus_docs = []

    def fit_bm25(self, documents: List[Document]):
        """
        Fits the BM25 sparse retriever on the provided corpus.
        """
        self.corpus_docs = documents
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Document]:
        """
        Performs hybrid search by combining Dense and Sparse scores.
        alpha: weight for Dense vs Sparse (1.0 = Dense only, 0.0 = Sparse only)
        """
        # Dense Retrieval
        dense_results = self.db.similarity_search_with_relevance_scores(query, k=k*2)
        
        # Sparse Retrieval (BM25)
        if self.bm25:
            tokenized_query = query.lower().split()
            sparse_scores = self.bm25.get_scores(tokenized_query)
            # Normalize sparse scores
            if np.max(sparse_scores) > 0:
                sparse_scores = sparse_scores / np.max(sparse_scores)
            
            top_sparse_indices = np.argsort(sparse_scores)[-k:][::-1]
            sparse_results = [(self.corpus_docs[i], sparse_scores[i]) for i in top_sparse_indices]
        else:
            sparse_results = []

        # Reciprocal Rank Fusion (RRF) or Simple weighted fusion
        # For simplicity in this implementation, we use weighted fusion of normalized scores
        combined_results = {}
        
        for doc, score in dense_results:
            combined_results[doc.page_content] = {"doc": doc, "score": score * alpha}
            
        for doc, score in sparse_results:
            if doc.page_content in combined_results:
                combined_results[doc.page_content]["score"] += score * (1 - alpha)
            else:
                combined_results[doc.page_content] = {"doc": doc, "score": score * (1 - alpha)}
        
        sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
        return [res["doc"] for res in sorted_results[:k]]

    def add_documents(self, documents: List[Document]):
        self.db.add_documents(documents)
        self.fit_bm25(documents)
