import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeAlias

from typing_extensions import override

from graph_retriever.adapters.base import Adapter
from graph_retriever.content import Content
from graph_retriever.utils.math import cosine_similarity

SENTINEL = object()

Embedding: TypeAlias = Callable[[str], list[float]]


class InMemoryBase(Adapter, abc.ABC):
    """
    The base class for in-memory adapters that use dict-based metadata filters.

    These are intended (mostly) for demonstration purposes and testing.
    """

    def __init__(self, embedding: Embedding, content: list[Content]) -> None:
        """
        Initialize with the given embedding function.

        Parameters
        ----------
        embedding :
            embedding function to use.
        """
        self.store: dict[str, Content] = {c.id: c for c in content}
        self.embedding = embedding

    @override
    def search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        query_embedding = self.embedding(query)
        docs = self.search(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

    @override
    async def asearch_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        query_embedding = self.embedding(query)
        docs = await self.asearch(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

    @override
    def search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        # get all docs with fixed order in list
        candidates = self._matching_content(filter)

        if not candidates:
            return []

        similarity = cosine_similarity([embedding], [c.embedding for c in candidates])[
            0
        ]

        # get the indices ordered by similarity score
        top_k_idx = similarity.argsort()[::-1][:k]

        return [candidates[idx] for idx in top_k_idx]

    @override
    def get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Content]:
        return [
            c
            for id in ids
            if (c := self.store.get(id, None))
            if self._matches(filter, c)
        ]

    def _matching_content(self, filter: dict[str, Any] | None = None) -> list[Content]:
        """Return a list of content matching the given filters."""
        if filter:
            return [c for c in self.store.values() if self._matches(filter, c)]
        else:
            return list(self.store.values())

    def _matches(self, filter: dict[str, Any] | None, content: Content) -> bool:
        """Return whether `content` matches the given `filter`."""
        if not filter:
            return True

        for key, filter_value in filter.items():
            content_value = content.metadata
            for key_part in key.split("."):
                content_value = content_value.get(key_part, SENTINEL)
                if content_value is SENTINEL:
                    break
            if not self._value_matches(filter_value, content_value):
                return False
        return True

    @abc.abstractmethod
    def _value_matches(self, filter_value: str, content_value: Any) -> bool:
        """Return whether the `content_value` matches the `filter_value`."""
        ...


class InMemory(InMemoryBase):
    """
    In-Memory VectorStore that supports list-based metadata.

    This In-Memory store simulates VectorStores like AstraDB and OpenSearch
    """

    @override
    def _value_matches(self, filter_value: str, content_value: Any) -> bool:
        return (filter_value == content_value) or (
            isinstance(content_value, Iterable)
            and not isinstance(content_value, str | bytes)
            and filter_value in content_value
        )
