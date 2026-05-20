# Jed-AI Mission: RAG Pipeline from Scratch

## Assignment

Pick a domain and a corpus of your choice. Build an end-to-end RAG pipeline that:

- Ingests and chunks documents
- Embeds them using **at least two different embedding models**
- Stores them in a vector store
- Implements semantic search wired to an LLM to produce **grounded, cited responses**
- Experiments with **at least two chunking strategies** and compares retrieval quality between them

All responses must include source citations. If the answer isn't in the corpus, the system must say so.

## Deliverables

1. Working pipeline with modular ingestion, retrieval, and generation stages
2. Comparison table: chunking strategy vs. retrieval quality with real numbers
3. One-paragraph write-up on what you'd do differently with a larger corpus
