from langchain_ollama import OllamaEmbeddings

from rag_pipeline.settings import settings


def make_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
    )
