from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.settings import settings


class FixedSizeChunker:
    """Baseline: pure character-based chunking, no structural awareness."""

    def split(self, markdown: str) -> list[Document]:
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ).create_documents([markdown])
