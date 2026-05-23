from rag_pipeline.chunking.fixed_size import FixedSizeChunker
from rag_pipeline.chunking.hierarchical import HierarchicalChunker
from rag_pipeline.settings import ChunkingStrategy, RAGSettings, settings

__all__ = ["ChunkingStrategy", "get_chunker"]


def get_chunker(
    strategy: ChunkingStrategy, cfg: RAGSettings = settings
) -> FixedSizeChunker | HierarchicalChunker:
    match strategy:
        case "fixed_size":
            return FixedSizeChunker(
                chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
            )
        case "hierarchical":
            return HierarchicalChunker(
                chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
            )
