from pathlib import Path

from rag_pipeline.settings import RAGSettings


class EvalSettings(RAGSettings):
    eval_judge_model: str = "gemma4:e4b"
    vector_persist_path: Path = Path(".chroma_eval")

    def collection_name(self, game: str) -> str:
        safe_emb = self.embedding_model.replace(":", "_").replace("/", "_")
        enriched = "ctx" if self.chunk_context_enrichment else "plain"
        name = (
            f"eval__{game}__{self.chunking_strategy}"
            f"__{enriched}__cs{self.chunk_size}__{safe_emb}"
        )
        return name[:100]

    def label(self) -> str:
        enriched = "+context" if self.chunk_context_enrichment else ""
        return (
            f"{self.chunking_strategy}{enriched}"
            f" cs={self.chunk_size} / {self.embedding_model}"
        )
