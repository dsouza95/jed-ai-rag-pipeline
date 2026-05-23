from typing import Any

from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    OllamaEmbeddingFunction,
)


def get_embedding_function(model: str) -> Any:
    """Return a ChromaDB-compatible embedding function for the given model spec.

    Formats:
        "default"             — ChromaDB built-in (all-MiniLM-L6-v2)
        "<model>"             — Ollama model name
    """
    if model == "default":
        return DefaultEmbeddingFunction()

    return OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=model,
    )
