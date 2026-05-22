from chromadb.api.types import Metadata

from rag_pipeline.db import make_vector_db_client
from rag_pipeline.embedding import get_embedding_function
from rag_pipeline.settings import RAGSettings, settings


def retrieve(
    game_name: str, query: str, n_results: int = 5, cfg: RAGSettings = settings
) -> list[tuple[str, Metadata]]:
    emb_fn = get_embedding_function(cfg.embedding_model)
    collection = make_vector_db_client(cfg.vector_persist_path).get_collection(
        name=game_name, embedding_function=emb_fn
    )
    results = collection.query(query_texts=query, n_results=n_results)
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return list(zip(documents, metadatas))


def build_context(chunks: list[tuple[str, Metadata]]) -> str:
    excerpt_parts = []
    for index, (doc, meta) in enumerate(chunks):
        page = meta.get("page", "")
        header = f"Excerpt {index + 1}" + (f" (p. {page})" if page else "")
        excerpt_parts.append(f"<excerpt>\n{header}\n{doc}\n</excerpt>")
    return "\n".join(excerpt_parts)
