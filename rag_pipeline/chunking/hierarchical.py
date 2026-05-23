from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_pipeline.settings import settings


class HierarchicalChunker:
    """Header-aware two-stage chunking: split on H1/H2, then by character limit."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, markdown: str) -> list[Document]:
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "section"), ("##", "subsection")],
            strip_headers=False,
        ).split_text(markdown)
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        ).split_documents(sections)
