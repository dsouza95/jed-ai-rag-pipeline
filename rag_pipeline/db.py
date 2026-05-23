from pathlib import Path

import chromadb
from chromadb.api import ClientAPI


def make_vector_db_client(path: Path) -> ClientAPI:
    return chromadb.PersistentClient(path)
