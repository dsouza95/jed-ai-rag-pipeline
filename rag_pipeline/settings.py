from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    embedder_model: str = "embeddinggemma"


settings = RAGSettings()
