import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_core.documents import Document
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.parsing_get_response import MarkdownPageMarkdownResultPage

from rag_pipeline.chunking import get_chunker
from rag_pipeline.db import make_vector_db_client
from rag_pipeline.embedding import get_embedding_function
from rag_pipeline.indexer.context_chain import chain
from rag_pipeline.settings import RAGSettings, settings
from rag_pipeline.types import RulebookPage

IndexProgressCallback = Callable[[int, int], Awaitable[None]]


def _assign_page_to_chunks(chunks: list[Document], first_page_num: int) -> None:
    page_tag_re = re.compile(r"<page>(\d+)</page>")
    current_page = first_page_num
    for chunk in chunks:
        if match := page_tag_re.search(chunk.page_content):
            current_page = int(match.group(1))
        chunk.page_content = page_tag_re.sub("", chunk.page_content).strip()
        chunk.metadata["page"] = current_page


def _build_full_markdown(pages: list[RulebookPage]) -> str:
    return "\n\n".join(f"<page>{p.page_number}</page>\n{p.markdown}" for p in pages)


def _create_collection(name: str, cfg: RAGSettings = settings) -> chromadb.Collection:
    db_client = make_vector_db_client(cfg.vector_persist_path)
    emb_fn = get_embedding_function(cfg.embedding_model)
    return db_client.create_collection(name=name, embedding_function=emb_fn)


def _maybe_reset_collection(name: str, cfg: RAGSettings = settings):
    db_client = make_vector_db_client(cfg.vector_persist_path)
    try:
        db_client.delete_collection(name=name)
    except NotFoundError:
        pass


async def _enrich_chunks(
    chunks: list[Document],
    game_name: str,
    on_progress: IndexProgressCallback | None = None,
) -> list[Document]:
    enriched: list[Document] = []

    for i, chunk in enumerate(chunks):
        if not chunk.page_content:
            continue

        context = await chain.ainvoke(
            {
                "section": chunk.metadata.get("section", ""),
                "chunk": chunk.page_content,
            }
        )

        chunk.page_content = f"{context}\n\n{chunk.page_content}"
        chunk.metadata.update({"game": game_name, "context": context})
        enriched.append(chunk)

        if on_progress:
            await on_progress(i + 1, len(chunks))

    return enriched


async def parse_rulebook(
    rulebook_file_path: Path,
    cfg: RAGSettings = settings,
) -> list[RulebookPage]:
    llama_client = AsyncLlamaCloud()
    file = await llama_client.files.create(file=rulebook_file_path, purpose="parse")
    rulebook = await llama_client.parsing.parse(
        file_id=file.id,
        tier=cfg.document_parse_tier,
        version="latest",
        expand=["markdown"],
    )

    pages = rulebook.markdown.pages if rulebook.markdown else []
    successful_pages = [
        p for p in pages if isinstance(p, MarkdownPageMarkdownResultPage)
    ]
    if not successful_pages:
        raise ValueError("Failed to parse rulebook or rulebook is empty")

    return [RulebookPage(p.page_number, p.markdown) for p in successful_pages]


async def index_game(
    game_name: str,
    pages: list[RulebookPage],
    cfg: RAGSettings = settings,
    collection_name: str | None = None,
    on_progress: IndexProgressCallback | None = None,
) -> int:
    collection_name = collection_name or game_name
    _maybe_reset_collection(collection_name, cfg)
    collection = _create_collection(collection_name, cfg)

    chunker = get_chunker(cfg.chunking_strategy)
    markdown = _build_full_markdown(pages)
    chunks = chunker.split(markdown)
    _assign_page_to_chunks(chunks, first_page_num=pages[0].page_number)

    if cfg.chunk_context_enrichment:
        chunks = await _enrich_chunks(chunks, game_name, on_progress)
    else:
        chunks = [c for c in chunks if c.page_content]
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def is_game_indexed(game_name: str, cfg: RAGSettings = settings) -> bool:
    try:
        make_vector_db_client(cfg.vector_persist_path).get_collection(name=game_name)
    except NotFoundError:
        return False
    else:
        return True


def list_indexed_games(cfg: RAGSettings = settings) -> list[str]:
    return [
        col.name
        for col in make_vector_db_client(cfg.vector_persist_path).list_collections()
    ]
