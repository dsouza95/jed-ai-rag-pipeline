import shutil
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import chromadb
from chromadb.api.types import Metadata
from chromadb.errors import NotFoundError
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.parsing_get_response import MarkdownPageMarkdownResultPage

from rag_pipeline.db import make_vector_db_client
from rag_pipeline.indexer.context_chain import chain
from rag_pipeline.settings import settings

IndexProgressCallback = Callable[[int, int], Awaitable[None]]


def _build_markdown(pages: list[MarkdownPageMarkdownResultPage]) -> str:
    return "\n\n".join(
        f"<page>{page.page_number}</page>\n{page.markdown}" for page in pages
    )


async def _enrich_chunks(
    chunks: list[Document],
    game_name: str,
    on_progress: IndexProgressCallback | None,
) -> tuple[list[str], list[str], list[Metadata]]:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[Metadata] = []

    for i, chunk in enumerate(chunks):
        context = await chain.ainvoke(
            {
                "section": chunk.metadata.get("section", ""),
                "chunk": chunk.page_content,
            }
        )

        content = chunk.page_content.strip()
        if "<page>" in chunk.page_content:
            content = content.split("</page>", 1)[-1].strip()
        if not content:
            continue

        metadata: Metadata = {
            **chunk.metadata,
            "game": game_name,
            "context": context,
            "content": content,
        }

        ids.append(str(uuid.uuid4()))
        documents.append(f"{context}\n\n{content}")
        metadatas.append(metadata)

        if on_progress:
            await on_progress(i + 1, len(chunks))

    return ids, documents, metadatas


async def _parse_rulebook(
    rulebook_file_path: Path,
) -> list[MarkdownPageMarkdownResultPage]:
    llama_client = AsyncLlamaCloud()
    file = await llama_client.files.create(file=rulebook_file_path, purpose="parse")
    rulebook = await llama_client.parsing.parse(
        file_id=file.id,
        tier=settings.document_parse_tier,
        version="latest",
        expand=["markdown"],
    )

    pages = rulebook.markdown.pages if rulebook.markdown else []
    successful_pages = [
        p for p in pages if isinstance(p, MarkdownPageMarkdownResultPage)
    ]
    if not successful_pages:
        raise ValueError("Failed to parse rulebook or rulebook is empty")

    return successful_pages


def _reset_collection(game_name: str) -> chromadb.Collection:
    db_client = make_vector_db_client()
    try:
        db_client.get_collection(name=game_name)
        db_client.delete_collection(name=game_name)
    except NotFoundError:
        pass
    return db_client.create_collection(name=game_name)


def _split_into_chunks(markdown: str, first_page_num: int) -> list[Document]:
    section_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "section"), ("##", "subsection")],
        strip_headers=False,
    )
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    sections = section_splitter.split_text(markdown)
    chunks = chunk_splitter.split_documents(sections)

    current_page_num = first_page_num
    for chunk in chunks:
        if "<page>" in chunk.page_content:
            current_page_num = int(
                chunk.page_content.split("<page>")[1].split("</page>")[0]
            )
        chunk.metadata["page"] = current_page_num

    return chunks


def _copy_rulebook(rulebook_file_path: Path, game_name: str) -> None:
    settings.rulebooks_path.mkdir(exist_ok=True)
    shutil.copy2(rulebook_file_path, get_rulebook_path(game_name))


async def index_game(
    rulebook_file_path: Path,
    game_name: str,
    on_progress: IndexProgressCallback | None = None,
) -> int:
    collection = _reset_collection(game_name)

    _copy_rulebook(rulebook_file_path, game_name)

    pages = await _parse_rulebook(rulebook_file_path)
    first_page_num = pages[0].page_number

    markdown = _build_markdown(pages)
    chunks = _split_into_chunks(markdown, first_page_num)
    ids, documents, metadatas = await _enrich_chunks(chunks, game_name, on_progress)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return len(chunks)


def get_rulebook_path(game_name: str) -> Path:
    return settings.rulebooks_path / f"{game_name}.pdf"


def is_game_indexed(game_name: str) -> bool:
    try:
        make_vector_db_client().get_collection(name=game_name)
    except NotFoundError:
        return False
    else:
        return True


def list_indexed_games() -> list[str]:
    db_client = make_vector_db_client()
    return [col.name for col in db_client.list_collections()]
