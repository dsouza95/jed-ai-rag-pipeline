from chromadb.api.types import Metadata

from rag_pipeline.db import make_vector_db_client


def retrieve(
    game_name: str, query: str, n_results: int = 5
) -> list[tuple[str, Metadata]]:
    collection = make_vector_db_client().get_collection(name=game_name)
    results = collection.query(query_texts=query, n_results=n_results)
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return list(zip(documents, metadatas))


def build_context(chunks: list[tuple[str, Metadata]]) -> str:
    excerpt_parts = []
    for index, (doc, meta) in enumerate(chunks):
        page = meta.get("page", "")
        context_text = meta.get("context", "")
        content = meta.get("content", doc)

        header = f"Excerpt {index + 1}" + (f" (p. {page})" if page else "")
        excerpt_parts.append(
            f"<excerpt>\n{header}\n{context_text}\n\n{content}\n</excerpt>"
        )

    return "\n".join(excerpt_parts)
