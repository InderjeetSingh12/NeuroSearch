# ðŸ§ª NeuroSearch Research: Advanced RAG Engine

![Research](https://img.shields.io/badge/Role-ML%20Research%20Scientist-blue?style=for-the-badge)
![AI](https://img.shields.io/badge/Architecture-Hybrid%20RAG-darkgreen?style=for-the-badge)
![Models](https://img.shields.io/badge/Models-Bi--Encoder%20%2B%20Cross--Encoder-orange?style=for-the-badge)

**NeuroSearch Research** is a high-performance RAG (Retrieval-Augmented Generation) engine built with a research-first mindset. It implements a 2-stage retrieval pipeline to maximize retrieval precision and semantic relevance.

## ðŸ”¬ Advanced Research Features

### 1. Hybrid Sparse-Dense Retrieval
Combines **ChromaDB Dense Embeddings** (`all-MiniLM-L6-v2`) with **BM25 Sparse Retrieval**. This hybrid approach (Î±-weighted) ensures that the engine captures both semantic meaning and exact keyword matches, significantly reducing retrieval failure.

### 2. 2-Stage Retrieval Pipeline
*   **Stage 1 (Candidates):** Hybrid search retrieves top-15 candidate documents.
*   **Stage 2 (Re-ranking):** Uses a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) to re-score candidates based on query-document joint interaction, keeping only the top-4 for generation.

### 3. Modularity & Scalability
Designed for experimental reproducibility, with decoupled components for ingestion, vector space management, and LLM inference.

## ðŸ“ Mathematical Overview

The combined retrieval score $S$ is calculated as:
$$S = \alpha \cdot S_{dense} + (1 - \alpha) \cdot S_{sparse}$$
where $\alpha$ is the hybrid weighting factor, $S_{dense}$ is the cosine similarity score, and $S_{sparse}$ is the normalized BM25 score.

## ðŸ›  Setup

1.  **Clone & Install:**
    ```bash
    git clone https://github.com/nft94/NeuroSearch.git
    pip install -r requirements.txt
    ```

2.  **Ollama Engine:**
    Ensure Ollama is running with Llama 3 or similar:
    ```bash
    ollama run llama3
    ```

## ðŸ’» CLI Usage

### Ingestion (Hybrid Indexing)
```bash
python main.py ingest --dir ./my_research_docs
```

### Research Chat (with Re-ranking)
```bash
python main.py chat --alpha 0.7
```
*`--alpha 0.7` favors semantic search; set lower for keyword-heavy queries.*

## ðŸ“‚ Project Roadmap
- [x] Hybrid Dense-Sparse Search
- [x] Cross-Encoder Re-ranking
- [ ] Retrieval Evaluation Module (MRR, NDCG)
- [ ] Contextual Compression Transformers
- [ ] Multi-Query Expansion

---
*Created with focus on robust ML pipelines and modular architecture.*
