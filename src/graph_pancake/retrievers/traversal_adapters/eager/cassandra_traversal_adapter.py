from typing import (
    Any,
    List,
    cast,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .traversal_adapter import TraversalAdapter


class CassandraTraversalAdapter(TraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        try:
            from langchain_community.vectorstores import Cassandra
        except (ImportError, ModuleNotFoundError):
            raise ImportError("please `pip install langchain-community cassio`")

        self._vector_store = cast(Cassandra, vector_store)
        self._base_vector_store = vector_store

    def similarity_search_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        return self._vector_store.similarity_search_by_vector(**kwargs)

    async def asimilarity_search_by_vector(  # type: ignore
        self, **kwargs: Any
    ) -> List[Document]:
        return await self._vector_store.asimilarity_search_by_vector(**kwargs)
