import uuid
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types import ParsingGetResponse
from tqdm.asyncio import tqdm

from rag_pipeline.settings import settings

_CONTEXT_SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
Here is a section from a board game rulebook:
<section>
{section}
</section>

Here is a specific passage from that section:
<chunk>
{chunk}
</chunk>

Write a short 1-2 sentence context that situates this passage within the section. \
Be concise and focus on what makes this chunk useful for answering rules questions.""")

_context_chain = (
    _CONTEXT_SUMMARY_PROMPT
    | ChatOllama(model=settings.chunk_context_model)
    | StrOutputParser()
)


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


def make_vector_db_client():
    return chromadb.PersistentClient(settings.vector_persist_path)


async def parse_rulebook(rulebook_file_path: Path) -> ParsingGetResponse:
    llama_client = AsyncLlamaCloud()
    file = await llama_client.files.create(file=rulebook_file_path, purpose="parse")
    return await llama_client.parsing.parse(
        file_id=file.id,
        tier=settings.document_parse_tier,
        version="latest",
        expand=["markdown_full"],
    )


async def index_game(rulebook_file_path: Path, game_name: str) -> int:
    """
    Indexes a game's rulebook PDF using llama to parse the PDF, splitting by markdown
    sections and smaller chunks that are enriched with section context summary
    """
    db_client = make_vector_db_client()
    try:
        db_client.get_collection(name=game_name)
        db_client.delete_collection(name=game_name)
    except NotFoundError:
        pass

    collection = db_client.create_collection(name=game_name)

    print(f"parsing rulebook for {game_name}...")
    rulebook = await parse_rulebook(rulebook_file_path)
    print(f"Successfully parsed rulebook for {game_name}.")

    markdown = rulebook.markdown_full.strip() if rulebook.markdown_full else ""
    if not markdown:
        raise ValueError("Failed to parse rulebook or rulebook is empty")

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

    print(f"split rulebook into {len(sections)} sections and {len(chunks)} chunks")

    ids, documents, metadatas = [], [], []
    async for chunk in tqdm(chunks, desc="Enriching chunks...", unit="chunk"):
        context = await _context_chain.ainvoke(
            {
                "section": chunk.metadata.get("section", ""),
                "chunk": chunk.page_content,
            }
        )
        ids.append(str(uuid.uuid4()))
        documents.append(f"{context}\n\n{chunk.page_content}")
        metadatas.append({**chunk.metadata, "game": game_name})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return len(chunks)
