from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

ChunkingStrategy = Literal["fixed_size", "hierarchical", "page"]


class RAGSettings(BaseSettings):
    chain_model: str = "gemma4:e4b"
    chunk_context_model: str = "qwen3:1.7b"
    chunk_overlap: int = 64
    chunk_size: int = 512
    chunking_strategy: ChunkingStrategy = "hierarchical"
    document_parse_tier: Literal[
        "fast", "cost_effective", "agentic", "agentic_plus"
    ] = "agentic"
    embedding_model: str = "default"
    rulebooks_path: Path = Path(".rulebooks")
    vector_persist_path: Path = Path(".chroma")


settings = RAGSettings()
