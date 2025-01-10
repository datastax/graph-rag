# type: ignore

from typing import (
    Any,
    List,
    Sequence,
    Tuple,
    cast,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .mmr_traversal_adapter import MMRTraversalAdapter


class CassandraMMRTraversalAdapter(MMRTraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        from langchain_community.vectorstores import Cassandra

        self._vector_store = cast(Cassandra, vector_store)

    def similarity_search_with_embedding(  # type: ignore
        self, **kwargs: Any
    ) -> Tuple[List[float], List[Document]]:
        self._vector_store.asimilarity_search_with_embedding_id_by_vector
        return self._vector_store.similarity_search_with_embedding(**kwargs)

    async def asimilarity_search_with_embedding(  # type: ignore
        self, **kwargs: Any
    ) -> Tuple[List[float], List[Document]]:
        return await self._vector_store.asimilarity_search_with_embedding(**kwargs)

    def similarity_search_with_embedding_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        return self._vector_store.similarity_search_with_embedding_by_vector(**kwargs)

    async def asimilarity_search_with_embedding_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        return await self._vector_store.asimilarity_search_with_embedding_by_vector(
            **kwargs
        )

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = self._vector_store.get_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs

    async def aget(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = await self._vector_store.aget_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs
