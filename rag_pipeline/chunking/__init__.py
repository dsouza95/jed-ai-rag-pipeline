from rag_pipeline.chunking.fixed_size import FixedSizeChunker
from rag_pipeline.chunking.hierarchical import HierarchicalChunker
from rag_pipeline.settings import ChunkingStrategy

__all__ = ["ChunkingStrategy", "get_chunker"]


def get_chunker(strategy: ChunkingStrategy) -> FixedSizeChunker | HierarchicalChunker:
    match strategy:
        case "fixed_size":
            return FixedSizeChunker()
        case "hierarchical":
            return HierarchicalChunker()
