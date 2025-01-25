from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, override

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

SENTINEL = object()


class _InMemoryBase(InMemoryVectorStore):
    """
    The base class for In-Memory Vector stores that use dict-based metadata filters.

    This is an alternative to the LangChain InMemoryVectorStore which defines a callable
    callable filter. This version behaves more like most existing vector stores.
    """

    @abstractmethod
    def supports_searching_in_metadata_list_values(self) -> bool:
        """Indicate if the store supports searching inside metadata list values."""

    @override
    def _similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,  # type: ignore
        **kwargs,
    ):
        return super()._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=self._filter_method(filter),
            **kwargs,
        )

    def _equals_or_contains(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any],
    ) -> bool:
        """
        Check if a key-value pair exists or if the value is contained in the metadata.

        Parameters
        ----------
        key : str
            Metadata key to look for.
        value : Any
            Value to check for equality or containment.
        metadata : dict[str, Any]
            Metadata dictionary to inspect.

        Returns
        -------
        bool
            True if and only if `metadata[key] == value` or `metadata[key]` is a
            list containing `value`.
        """
        actual = metadata.get(key, SENTINEL)
        if actual == value:
            return True

        if (
            self.supports_searching_in_metadata_list_values()
            and isinstance(actual, Iterable)
            and not isinstance(actual, str | bytes)
            and value in actual
        ):
            return True

        return False

    def _filter_method(
        self, filter_dict: dict[str, str] | None = None
    ) -> Callable[[Document], bool]:
        """
        Create a filter function based on a metadata dictionary.

        Parameters
        ----------
        filter_dict : dict[str, str], optional
            Dictionary specifying the filter criteria.

        Returns
        -------
        Callable[[Document], bool]
            A function that determines if a document matches the filter criteria.
        """
        if filter_dict is None:
            return lambda _doc: True

        def filter(doc: Document) -> bool:
            for key, value in filter_dict.items():
                if not self._equals_or_contains(key, value, doc.metadata):
                    return False
            return True

        return filter


class InMemoryList(_InMemoryBase):
    """
    An In-Memory VectorStore that supports searching in list-based metadata.

    This In-Memory store simulates VectorStores like AstraDB and OpenSearch
    """

    @override
    def supports_searching_in_metadata_list_values(self) -> bool:
        return True


class InMemoryFlat(_InMemoryBase):
    """
    An In-Memory VectorStore that doesn't support searching in list-based metadata.

    This In-Memory store simulates VectorStores like Chroma and Cassandra
    """

    @override
    def supports_searching_in_metadata_list_values(self) -> bool:
        return False
