import chromadb

from rag_pipeline.settings import settings


def make_vector_db_client():
    return chromadb.PersistentClient(settings.vector_persist_path)
