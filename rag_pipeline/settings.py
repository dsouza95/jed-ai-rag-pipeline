from typing import Literal

from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    chain_model: str = "gemma4:e4b"
    chunk_context_model: str = "qwen3:1.7b"
    chunk_overlap: int = 64
    chunk_size: int = 512
    document_parse_tier: Literal[
        "fast", "cost_effective", "agentic", "agentic_plus"
    ] = "agentic"
    embedding_model: str = "embeddinggemma"


settings = RAGSettings()
