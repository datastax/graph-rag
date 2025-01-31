"""Defines the base class for vector store adapters."""

import abc
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from graph_retriever import Adapter, Edge, MetadataEdge
from graph_retriever.content import Content
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore
from typing_extensions import (
    override,
)

from langchain_graph_retriever._conversion import doc_to_content
from langchain_graph_retriever.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)

StoreT = TypeVar("StoreT", bound=VectorStore)


class LangchainAdapter(Generic[StoreT], Adapter):
    """
    Base adapter for integrating vector stores with the graph retriever system.

    This class provides a foundation for custom adapters, enabling consistent
    interaction with various vector store implementations.

    Parameters
    ----------
    vector_store : T
        The vector store instance.
    """

    def __init__(
        self,
        vector_store: StoreT,
    ):
        """
        Initialize the base adapter.

        Parameters
        ----------
        vector_store : T
            The vector store instance.
        """
        self.vector_store = vector_store

    @property
    def _safe_embedding(self) -> Embeddings:
        if not self.vector_store.embeddings:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self.vector_store.embeddings

    @override
    def embed_query(self, query):
        return self._safe_embedding.embed_query(query)

    def update_filter_hook(
        self, filter: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """
        Update the metadata filter before executing the query.

        Parameters
        ----------
        filter : dict[str, str], optional
            Filter on the metadata to update.

        Returns
        -------
        dict[str, Any] | None
            The updated filter on the metadata to apply.
        """
        return filter

    def format_documents_hook(self, docs: list[Document]) -> list[Content]:
        """
        Format the documents as content after executing the query.

        Parameters
        ----------
        docs : list[Document]
            The documents returned from the vector store

        Returns
        -------
        list[Content]
            The formatted content.
        """
        return [doc_to_content(doc) for doc in docs]

    @abc.abstractmethod
    def _similarity_search_with_embedding_by_vector(
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
        embedding : list[float]
            Embedding to look up documents similar to.
        k : int, default 4
            Number of Documents to return.
        filter : dict[str, str], optional
            Filter on the metadata to apply.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[Document]
            List of Documents most similar to the query vector.

            Documents should have their embedding added to the
            metadata under the `METADATA_EMBEDDING_KEY` key.
        """

    @override
    def similarity_search_with_embedding_by_vector(
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
        embedding : list[float]
            Embedding to look up documents similar to.
        k : int, default 4
            Number of Documents to return.
        filter : dict[str, str], optional
            Filter on the metadata to apply.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[Content]
            List of Contents most similar to the query vector.
        """
        docs = self._similarity_search_with_embedding_by_vector(
            embedding=embedding,
            k=k,
            filter=self.update_filter_hook(filter),
            **kwargs,
        )
        return self.format_documents_hook(docs)

    async def _asimilarity_search_with_embedding_by_vector(
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
        embedding : list[float]
            Embedding to look up documents similar to.
        k : int, default 4
            Number of Documents to return.
        filter : dict[str, str], optional
            Filter on the metadata to apply.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[Document]
            List of Documents most similar to the query vector.

            Documents should have their embedding added to the
            metadata under the `METADATA_EMBEDDING_KEY` key.
        """
        return await run_in_executor(
            None,
            self._similarity_search_with_embedding_by_vector,
            embedding,
            k,
            filter,
            **kwargs,
        )

    @override
    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        docs = await self._asimilarity_search_with_embedding_by_vector(
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
        ids : Sequence[str]
            List of IDs to get.

        Returns
        -------
        Sequence[str]
            List of IDs with duplicates removed
        """
        return list({k: True for k in ids}.keys())

    @override
    def get(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Content]:
        docs = self._get(self._remove_duplicates(ids), **kwargs)
        return self.format_documents_hook(docs)

    @abc.abstractmethod
    def _get(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Get documents by id.

        Fewer documents may be returned than requested if some IDs are not found
        or if there are duplicated IDs. This method should **NOT** raise
        exceptions if no documents are found for some IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field
        of the returned documents.

        Parameters
        ----------
        ids : Sequence[str]
            List of IDs to get.
        **kwargs : Any
            Additional keyword arguments. These are up to the implementation.

        Returns
        -------
        list[Document]
            List of documents that were found.
        """

    @override
    async def aget(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Content]:
        docs = await self._aget(self._remove_duplicates(ids), **kwargs)
        return self.format_documents_hook(docs)

    async def _aget(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Asynchronously get documents by id.

        Fewer documents may be returned than requested if some IDs are not found
        or if there are duplicated IDs. This method should **NOT** raise
        exceptions if no documents are found for some IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field
        of the returned documents.

        Parameters
        ----------
        ids : Sequence[str]
            List of IDs to get.
        **kwargs : Any
            Additional keyword arguments. These are up to the implementation.

        Returns
        -------
        list[Document]
            List of documents that were found.
        """
        return await run_in_executor(
            None,
            self._get,
            ids,
            **kwargs,
        )

    def _get_metadata_filter(
        self,
        base_filter: dict[str, Any] | None = None,
        edge: Edge | None = None,
    ) -> dict[str, Any]:
        """
        Return a filter for the `base_filter` and incoming edges from `edge`.

        Parameters
        ----------
        base_filter : dict[str, Any]
            Any base metadata filter that should be used for search.
            Generally corresponds to the user specified filters for the entire
            traversal. Should be combined with the filters necessary to support
            nodes with an *incoming* edge matching `edge`.
        edge : Edge, optional
            An optional edge which should be added to the filter.

        Returns
        -------
        dict[str, Any]
            The metadata dictionary to use for the given filter.
        """
        metadata_filter = {**(base_filter or {})}
        assert isinstance(edge, MetadataEdge)
        if edge is None:
            metadata_filter
        else:
            metadata_filter[edge.incoming_field] = edge.value
        return metadata_filter


class DenormalizedAdapter(LangchainAdapter[StoreT]):
    """
    Base adapter for integrating vector stores with the graph retriever system.

    This class provides a foundation for custom adapters, enabling consistent
    interaction with various vector store implementations that do not support
    searching on list-based metadata values.

    Parameters
    ----------
    vector_store : T
        The vector store instance.
    metadata_denormalizer: MetadataDenormalizer | None
        (Optional) An instance of the MetadataDenormalizer used for doc insertion.
        If not passed then a default instance of MetadataDenormalizer is used.
    nested_metadata_fields: set[str]
        The set of metadata fields that contain nested values.
    """

    def __init__(
        self,
        vector_store: StoreT,
        metadata_denormalizer: MetadataDenormalizer | None = None,
        nested_metadata_fields: set[str] = set(),
    ):
        """
        Initialize the base adapter.

        Parameters
        ----------
        vector_store : T
            The vector store instance.
        metadata_denormalizer: MetadataDenormalizer | None
            (Optional) An instance of the MetadataDenormalizer used for doc insertion.
            If not passed then a default instance of MetadataDenormalizer is used.
        nested_metadata_fields: set[str]
            The set of metadata fields that contain nested values.
        """
        super().__init__(vector_store=vector_store)
        self.metadata_denormalizer = (
            MetadataDenormalizer()
            if metadata_denormalizer is None
            else metadata_denormalizer
        )
        self.nested_metadata_fields = nested_metadata_fields

    @override
    def update_filter_hook(
        self, filter: dict[str, str] | None
    ) -> dict[str, str] | None:
        if filter is None:
            return None

        denormalized_filter = {}
        for key, value in filter.items():
            if key in self.nested_metadata_fields:
                denormalized_filter[
                    self.metadata_denormalizer.denormalized_key(key, value)
                ] = self.metadata_denormalizer.denormalized_value()
            else:
                denormalized_filter[key] = value
        return denormalized_filter

    @override
    def format_documents_hook(self, docs: list[Document]) -> list[Content]:
        normalized = list(self.metadata_denormalizer.revert_documents(documents=docs))
        return super().format_documents_hook(normalized)
