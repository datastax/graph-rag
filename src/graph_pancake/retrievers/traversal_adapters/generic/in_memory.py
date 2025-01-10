from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .base import METADATA_EMBEDDING_KEY, StoreAdapter

SENTINEL = object()


class InMemoryStoreAdapter(StoreAdapter[InMemoryVectorStore]):
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        *,
        support_normalized_metadata: bool = False,
    ):
        self.vector_store = vector_store
        self.support_normalized_metadata = support_normalized_metadata

    def get(self, ids: Sequence[str], /, **kwargs) -> list[Document]:
        return self.vector_store.get_by_ids(ids)

    async def aget(self, ids: Sequence[str], /, **kwargs) -> list[Document]:
        return await self.vector_store.aget_by_ids(ids)

    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        results = self.vector_store._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=self._filter_method(filter),
        )
        return [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={METADATA_EMBEDDING_KEY: embedding, **doc.metadata},
            )
            for doc, _score, embedding in results
        ]

    def _equals_or_contains(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any],
    ) -> bool:
        """Tests if the key->value exists or is contained in the metadata."""
        actual = metadata.get(key, SENTINEL)
        if actual == value:
            return True

        if (
            self.support_normalized_metadata
            and isinstance(actual, Iterable)
            and not isinstance(actual, (str, bytes))
            and value in actual
        ):
            return True

        return False

    def _filter_method(
        self, filter_dict: Optional[Dict[str, str]] = None
    ) -> Callable[[Document], bool]:
        if filter_dict is None:
            return lambda _doc: True

        def filter(doc: Document) -> bool:
            for key, value in filter_dict.items():
                if not self._equals_or_contains(key, value, doc.metadata):
                    return False
            return True

        return filter
