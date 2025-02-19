"""Defines the base class for vector store adapters."""

import abc
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from graph_retriever import Content
from graph_retriever.adapters import Adapter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores.base import VectorStore
from typing_extensions import (
    override,
)

from langchain_graph_retriever._conversion import doc_to_content
from langchain_graph_retriever.transformers import ShreddingTransformer

StoreT = TypeVar("StoreT", bound=VectorStore)


class LangchainAdapter(Generic[StoreT], Adapter):
    """
    Base adapter for integrating vector stores with the graph retriever system.

    This class provides a foundation for custom adapters, enabling consistent
    interaction with various vector store implementations.

    Parameters
    ----------
    vector_store :
        The vector store instance.
    """

    def __init__(
        self,
        vector_store: StoreT,
    ):
        """Initialize the base adapter."""
        self.vector_store = vector_store

    @property
    def _safe_embedding(self) -> Embeddings:
        if not self.vector_store.embeddings:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self.vector_store.embeddings

    def embed_query(self, query: str):
        """Return the embedding of the query."""
        return self._safe_embedding.embed_query(query)

    async def aembed_query(self, query: str):
        """Return the embedding of the query."""
        return await self._safe_embedding.aembed_query(query)

    def update_filter_hook(
        self, filter: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """
        Update the metadata filter before executing the query.

        Parameters
        ----------
        filter :
            Filter on the metadata to update.

        Returns
        -------
        :
            The updated filter on the metadata to apply.
        """
        return filter

    def format_documents_hook(self, docs: list[Document]) -> list[Content]:
        """
        Format the documents as content after executing the query.

        Parameters
        ----------
        docs :
            The documents returned from the vector store

        Returns
        -------
        :
            The formatted content.
        """
        return [doc_to_content(doc) for doc in docs]

    @override
    def search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        query_embedding = self.embed_query(query)
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
        query_embedding = await self.aembed_query(query)
        docs = await self.asearch(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

    @abc.abstractmethod
    def _search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Return docs (with embeddings) most similar to the query vector.

        Parameters
        ----------
        embedding :
            Embedding to look up documents similar to.
        k :
            Number of Documents to return.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments.

        Returns
        -------
        :
            List of Documents most similar to the query vector.

            Documents should have their embedding added to the
            metadata under the `METADATA_EMBEDDING_KEY` key.
        """

    @override
    def search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        """
        Return contents most similar to the query vector.

        Parameters
        ----------
        embedding :
            Embedding to look up documents similar to.
        k :
            Number of Documents to return.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments.

        Returns
        -------
        :
            List of Contents most similar to the query vector.
        """
        if k == 0:
            return []

        docs = self._search(
            embedding=embedding,
            k=k,
            filter=self.update_filter_hook(filter),
            **kwargs,
        )
        return self.format_documents_hook(docs)

    async def _asearch(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Asynchronously return docs (with embeddings) most similar to the query vector.

        Parameters
        ----------
        embedding :
            Embedding to look up documents similar to.
        k :
            Number of Documents to return.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments.

        Returns
        -------
        :
            List of Documents most similar to the query vector.

            Documents should have their embedding added to the
            metadata under the `METADATA_EMBEDDING_KEY` key.
        """
        return await run_in_executor(
            None,
            self._search,
            embedding,
            k,
            filter,
            **kwargs,
        )

    @override
    async def asearch(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        if k == 0:
            return []

        docs = await self._asearch(
            embedding=embedding,
            k=k,
            filter=self.update_filter_hook(filter),
            **kwargs,
        )
        return self.format_documents_hook(docs)

    def _remove_duplicates(self, ids: Sequence[str]) -> list[str]:
        """
        Remove duplicate ids while preserving order.

        Parameters
        ----------
        ids :
            List of IDs to get.

        Returns
        -------
        :
            List of IDs with duplicates removed
        """
        return list({k: True for k in ids}.keys())

    @override
    def get(
        self,
        ids: Sequence[str],
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        docs = self._get(self._remove_duplicates(ids), filter, **kwargs)
        return self.format_documents_hook(docs)

    @abc.abstractmethod
    def _get(
        self,
        ids: Sequence[str],
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Get documents by ID.

        Fewer documents may be returned than requested if some IDs are not found
        or if there are duplicated IDs. This method should **NOT** raise
        exceptions if no documents are found for some IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field
        of the returned documents.

        Parameters
        ----------
        ids :
            List of IDs to get.
        filter :
            Filter to apply to the recrods.
        kwargs :
            Additional keyword arguments. These are up to the implementation.

        Returns
        -------
        :
            List of documents that were found.
        """

    @override
    async def aget(
        self,
        ids: Sequence[str],
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        docs = await self._aget(self._remove_duplicates(ids), filter, **kwargs)
        return self.format_documents_hook(docs)

    async def _aget(
        self,
        ids: Sequence[str],
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Asynchronously get documents by ID.

        Fewer documents may be returned than requested if some IDs are not found
        or if there are duplicated IDs. This method should **NOT** raise
        exceptions if no documents are found for some IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field
        of the returned documents.

        Parameters
        ----------
        ids :
            List of IDs to get.
        filter :
            Filter to apply to the documents.
        kwargs :
            Additional keyword arguments. These are up to the implementation.

        Returns
        -------
        :
            List of documents that were found.
        """
        return await run_in_executor(
            None,
            self._get,
            ids,
            filter,
            **kwargs,
        )


class ShreddedLangchainAdapter(LangchainAdapter[StoreT]):
    """
    Base adapter for integrating vector stores with the graph retriever system.

    This class provides a foundation for custom adapters, enabling consistent
    interaction with various vector store implementations that do not support
    searching on list-based metadata values.

    Parameters
    ----------
    vector_store :
        The vector store instance.
    shredder: ShreddingTransformer, optional
        An instance of the ShreddingTransformer used for doc insertion.
        If not passed then a default instance of ShreddingTransformer is used.
    nested_metadata_fields: set[str]
        The set of metadata fields that contain nested values.
    """

    def __init__(
        self,
        vector_store: StoreT,
        shredder: ShreddingTransformer | None = None,
        nested_metadata_fields: set[str] = set(),
    ):
        """Initialize the base adapter."""
        super().__init__(vector_store=vector_store)
        self.shredder = ShreddingTransformer() if shredder is None else shredder
        self.nested_metadata_fields = nested_metadata_fields

    @override
    def update_filter_hook(
        self, filter: dict[str, str] | None
    ) -> dict[str, str] | None:
        if filter is None:
            return None

        shredded_filter = {}
        for key, value in filter.items():
            if key in self.nested_metadata_fields:
                shredded_filter[self.shredder.shredded_key(key, value)] = (
                    self.shredder.shredded_value()
                )
            else:
                shredded_filter[key] = value
        return shredded_filter

    @override
    def format_documents_hook(self, docs: list[Document]) -> list[Content]:
        restored = list(self.shredder.restore_documents(documents=docs))
        return super().format_documents_hook(restored)
