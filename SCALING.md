# Scaling this demo to handle a larger corpus
To scale this demo to handle real-use and a larger corpus we should:
1. Add an actual API and UI: chainlit is great for prototyping, but this is not usable by the end-user.
2. Scale the ingestion pipeline by making it async and distributed: we can use job queues for orchestration, batch requests for context generation and chunk embedding, etc.
3. Chromadb was awesome for prototyping, but there are other solutions that would provide superior support and scalability like Pinecone, Weaviate, etc
4. Hybrid retrieval and reranking: both strategies should bring major improvements to the overall quality of generations
5. Prefix Caching: this is crucial due to context enrichment. Without caching, every chunk makes a full cost LLM call, which can become expensive quickly.
