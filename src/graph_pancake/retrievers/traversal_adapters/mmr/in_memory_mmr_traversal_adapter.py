from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    cast,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from graph_pancake.retrievers.consts import METADATA_EMBEDDING_KEY

from .mmr_traversal_adapter import MMRTraversalAdapter


class InMemoryMMRTraversalAdapter(MMRTraversalAdapter):
    def __init__(
        self, vector_store: VectorStore, support_normalized_metadata: bool = False
    ):
        self._vector_store = cast(InMemoryVectorStore, vector_store)
        self._base_vector_store = vector_store
        self._support_normalized_metadata = support_normalized_metadata

    def _equals_or_contains(
        self, key: str, value: Any, metadata: dict[str, Any]
    ) -> bool:
        """Tests if the key->value exists or is contained in the metadata"""
        if key in metadata and metadata[key] == value:
            return True

        if self._support_normalized_metadata:
            if key in metadata:
                metadata_value = metadata[key]
                if (
                    isinstance(metadata_value, Iterable)
                    and not isinstance(metadata_value, (str, bytes))
                    and value in metadata_value
                ):
                    return True
        return False

    def get(self, ids: Sequence[str], /, **kwargs) -> list[Document]:
        return self._vector_store.get_by_ids(ids)

    def similarity_search_with_embedding_by_vector(  # type: ignore
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        def filter_method(doc: Document) -> bool:
            if filter is None:
                return True

            for key, value in filter.items():
                if not self._equals_or_contains(
                    key=key, value=value, metadata=doc.metadata
                ):
                    return False

            return True

        docs = self._vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter_method,
            **kwargs,
        )

        for doc in docs:
            embedding = self._vector_store.store[doc.id]["vector"]
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding

        return docs
