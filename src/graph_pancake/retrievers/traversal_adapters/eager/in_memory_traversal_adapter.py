from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    cast,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from .traversal_adapter import TraversalAdapter


class InMemoryTraversalAdapter(TraversalAdapter):
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

    def similarity_search_by_vector(  # type: ignore
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

        return [doc.model_copy(deep=True) for doc in docs]
