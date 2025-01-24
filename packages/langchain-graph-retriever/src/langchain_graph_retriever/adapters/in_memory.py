"""Provides an adapter for the InMemoryVectorStore integration."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, override

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .base import METADATA_EMBEDDING_KEY, Adapter

SENTINEL = object()


class InMemoryAdapter(Adapter[InMemoryVectorStore]):
    """
    Adapter for InMemoryVectorStore.

    This adapter integrates the in-memory vector store with the graph retriever system,
    enabling similarity search and document retrieval.

    Parameters
    ----------
    vector_store : InMemoryVectorStore
        The in-memory vector store instance.
    use_normalized_metadata : bool, default False
        Indicates whether normalized metadata is used.
    denormalized_path_delimiter : str, default "."
        Delimiter for denormalized metadata keys.
    denormalized_static_value : str, default "$"
        Value to use for denormalized metadata entries.
    """

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        *,
        use_normalized_metadata: bool = False,
        denormalized_path_delimiter: str = ".",
        denormalized_static_value: str = "$",
    ):
        super().__init__(
            vector_store,
            use_normalized_metadata=use_normalized_metadata,
            denormalized_path_delimiter=denormalized_path_delimiter,
            denormalized_static_value=denormalized_static_value,
        )

    @override
    def get(self, ids: Sequence[str], /, **kwargs) -> list[Document]:
        docs = self.vector_store.get_by_ids(ids)
        # NOTE: Assumes embedding is deterministic.
        embeddings = self._safe_embedding.embed_documents(
            [doc.page_content for doc in docs]
        )
        return self._add_embeddings(docs, embeddings)

    def _add_embeddings(
        self, docs: Sequence[Document], embeddings: list[list[float]]
    ) -> list[Document]:
        """
        Add embeddings to the metadata of documents.

        Parameters
        ----------
        docs : Sequence[Document]
            List of documents.
        embeddings : list[list[float]])
            Corresponding embeddings.

        Returns
        -------
        list[Document]
            Documents with updated metadata containing embeddings.
        """
        return [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={METADATA_EMBEDDING_KEY: emb, **doc.metadata},
            )
            for doc, emb in zip(docs, embeddings)
        ]

    @override
    async def aget(self, ids: Sequence[str], /, **kwargs) -> list[Document]:
        docs = await self.vector_store.aget_by_ids(ids)
        embeddings = await self._safe_embedding.aembed_documents(
            [doc.page_content for doc in docs]
        )
        return self._add_embeddings(docs, embeddings)

    @override
    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
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
            self.use_normalized_metadata
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
