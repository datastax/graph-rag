"""Provides an adapter for the InMemoryVectorStore integration."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from typing_extensions import override

from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY
from langchain_graph_retriever.adapters.langchain import LangchainAdapter

SENTINEL = object()


class InMemoryAdapter(LangchainAdapter[InMemoryVectorStore]):
    """
    Adapter for InMemoryVectorStore vector store.

    This adapter integrates the in-memory vector store with the graph
    retriever system, enabling similarity search and document retrieval.

    Parameters
    ----------
    vector_store : InMemoryVectorStore
        The in-memory vector store instance.
    """

    @override
    def _get(self, ids: Sequence[str], /, **kwargs) -> list[Document]:
        docs: list[Document] = []

        for doc_id in ids:
            doc = self.vector_store.store.get(doc_id)
            if doc:
                metadata = doc["metadata"]
                metadata[METADATA_EMBEDDING_KEY] = doc["vector"]
                docs.append(
                    Document(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=metadata,
                    )
                )
        return docs

    @override
    def _similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs,
    ):
        results = self.vector_store._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=self._filter_method(filter_dict=filter),
            **kwargs,
        )
        docs = [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={METADATA_EMBEDDING_KEY: doc_embedding, **doc.metadata},
            )
            for doc, _score, doc_embedding in results
        ]
        return docs

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
            isinstance(actual, Iterable)
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
