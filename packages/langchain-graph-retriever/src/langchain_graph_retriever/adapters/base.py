"""Defines the base class for vector store adapters."""

import abc
import asyncio
from collections.abc import Iterable, Sequence
from typing import Any, Generic, TypeVar

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore
from typing_extensions import (
    override,
)

from langchain_graph_retriever.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)
from langchain_graph_retriever.types import Edge, IdEdge, MetadataEdge

StoreT = TypeVar("StoreT", bound=VectorStore)

METADATA_EMBEDDING_KEY = "__embedding"


class Adapter(Generic[StoreT], abc.ABC):
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

    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Document]]:
        """
        Return docs (with embeddings) most similar to the query.

        Also returns the embedded query vector.

        Parameters
        ----------
        query : str
            Input text.
        k : int, default 4
            Number of Documents to return.
        filter : dict[str, str], optional
            Filter on the metadata to apply.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        query_embedding : list[float]
            The embedded query vector
        documents : list[Document]
            List of up to `k` documents most similar to the query
            vector.

            Documents should have their embedding added to the
            metadata under the `METADATA_EMBEDDING_KEY` key.
        """
        query_embedding = self._safe_embedding.embed_query(text=query)
        docs = self.similarity_search_with_embedding_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Document]]:
        """
        Asynchronously return docs (with embeddings) most similar to the query.

        Also returns the embedded query vector.

        Parameters
        ----------
        query : str
            Input text.
        k : int, default 4
            Number of Documents to return.
        filter : dict[str, str], optional
            Filter on the metadata to apply.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        query_embedding : list[float]
            The embedded query vector
        documents : list[Document]
            List of up to `k` documents most similar to the query
            vector.

            Documents should have their embedding added to the
            metadata under the `METADATA_EMBEDDING_KEY` key.
        """
        return await run_in_executor(
            None, self.similarity_search_with_embedding, query, k, filter, **kwargs
        )

    def update_filter_hook(
        self, filter: dict[str, str] | None
    ) -> dict[str, str] | None:
        """
        Update the metadata filter before executing the query.

        Parameters
        ----------
        filter : dict[str, str], optional
            Filter on the metadata to update.

        Returns
        -------
        dict[str, str] | None
            The updated filter on the metadata to apply.
        """
        return filter

    def format_documents_hook(self, docs: list[Document]) -> list[Document]:
        """
        Format the documents filter after executing the query.

        Parameters
        ----------
        docs : list[Document]
            The documents returned from the vector store

        Returns
        -------
        list[Document]
            The formatted documents
        """
        return docs

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

    def similarity_search_with_embedding_by_vector(
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

    async def asimilarity_search_with_embedding_by_vector(
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
        seen = set()
        return [id_ for id_ in ids if id_ not in seen and not seen.add(id_)]  # type: ignore

    def get(
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

    async def aget(
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

    def get_adjacent(
        self,
        outgoing_edges: set[Edge],
        query_embedding: list[float],
        adjacent_k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """
        Return the docs with incoming edges from any of the given edges.

        Parameters
        ----------
        outgoing_edges : set[Edge]
            The edges to look for.
        query_embedding : list[float]
            The query embedding used for selecting the most relevant nodes.
        adjacent_k : int
            The numebr of relevant nodes to select for the edges.
        strategy : Strategy
            The traversal strategy being used.
        filter : dict[str, Any], optional
            Optional metadata to filter the results.
        **kwargs : Any
            Keyword arguments to pass to the similarity search.

        Returns
        -------
        Iterable[Document]
            Iterable of adjacent nodes.

        Raises
        ------
        ValueError
            If unsupported edge types are encountered.
        """
        results: list[Document] = []

        ids = []
        for outgoing_edge in outgoing_edges:
            if isinstance(outgoing_edge, MetadataEdge):
                docs = self.similarity_search_with_embedding_by_vector(
                    embedding=query_embedding,
                    k=adjacent_k,
                    filter=self._get_metadata_filter(
                        base_filter=filter, edge=outgoing_edge
                    ),
                    **kwargs,
                )
                results.extend(docs)
            elif isinstance(outgoing_edge, IdEdge):
                ids.append(outgoing_edge.id)
            else:
                raise ValueError(f"Unsupported edge: {outgoing_edge}")

        if ids:
            results.extend(self.get(ids))
        return results

    async def aget_adjacent(
        self,
        outgoing_edges: set[Edge],
        query_embedding: list[float],
        adjacent_k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """
        Asynchronously return the docs with incoming edges from any of the given edges.

        Parameters
        ----------
        outgoing_edges : set[Edge]
            The edges to look for.
        query_embedding : list[float]
            The query embedding used for selecting the most relevant nodes.
        adjacent_k : int
            The numebr of relevant nodes to select for the edges.
        filter : dict[str, Any], optional
            Optional metadata to filter the results.
        **kwargs : Any
            Keyword arguments to pass to the similarity search.

        Returns
        -------
        Iterable[Document]
            Iterable of adjacent nodes.

        Raises
        ------
        ValueError
            If unsupported edge types are encountered.
        """
        tasks = []
        ids = []
        for outgoing_edge in outgoing_edges:
            if isinstance(outgoing_edge, MetadataEdge):
                tasks.append(
                    self.asimilarity_search_with_embedding_by_vector(
                        embedding=query_embedding,
                        k=adjacent_k,
                        filter=self._get_metadata_filter(
                            base_filter=filter, edge=outgoing_edge
                        ),
                        **kwargs,
                    )
                )
            elif isinstance(outgoing_edge, IdEdge):
                ids.append(outgoing_edge.id)
            else:
                raise ValueError(f"Unsupported edge: {outgoing_edge}")

        if ids:
            tasks.append(self.aget(ids))

        results: list[Document] = []
        for completed_task in asyncio.as_completed(tasks):
            docs = await completed_task
            results.extend(docs)
        return results

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


class DenormalizedAdapter(Adapter[StoreT]):
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
    def format_documents_hook(self, docs: list[Document]) -> list[Document]:
        return list(self.metadata_denormalizer.revert_documents(documents=docs))
