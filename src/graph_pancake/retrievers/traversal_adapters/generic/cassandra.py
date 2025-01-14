from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Sequence,
)

from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import Cassandra
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install langchain-community`")

if TYPE_CHECKING:
    pass

from .base import METADATA_EMBEDDING_KEY, StoreAdapter


class CassandraStoreAdapter(StoreAdapter[Cassandra]):
    def __init__(self, vector_store: Cassandra):
        self.vector_store = vector_store

    def similarity_search_with_embedding_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        msg = "use the async implementation instead."
        raise NotImplementedError(msg)

    async def asimilarity_search_with_embedding_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        results = (
            await self.vector_store.asimilarity_search_with_embedding_id_by_vector(
                **kwargs
            )
        )
        docs: List[Document] = []
        for doc, embedding, id in results:
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding
            doc.id = id
            docs.append(doc)
        return docs

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = self.vector_store.get_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs

    async def aget(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = await self.vector_store.aget_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs
