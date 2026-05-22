from langchain_core.documents import Document

from rag_pipeline.types import RulebookPage


class PageChunker:
    """Coarse baseline: one chunk per page, maximum context per chunk."""

    def split(self, pages: list[RulebookPage]) -> list[Document]:
        return [
            Document(page_content=p.markdown, metadata={"page": p.page_number})
            for p in pages
            if p.markdown.strip()
        ]
