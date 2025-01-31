from collections.abc import Callable

from langchain_core.embeddings import Embeddings


class BaseEmbeddings(Embeddings):
    def __init__(self, embedding: Callable[[str], list[float]]) -> None:
        self.embedding = embedding

    def embed_query(self, text: str) -> list[float]:
        return self.embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embedding(text) for text in texts]
