"""Defines the base class for vector store adapters."""

import abc
import asyncio
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Generic,
    TypeVar,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore

from langchain_graph_retriever.edge_helper import Edge
from langchain_graph_retriever.strategies import Strategy

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
    use_normalized_metadata : bool
        Indicates whether normalized metadata is used.
    denormalized_path_delimiter : str
        Delimiter for denormalized metadata keys.
    denormalized_static_value : str
        Value to use for denormalized metadata entries.
    """

    def __init__(
        self,
        vector_store: StoreT,
        *,
        use_normalized_metadata: bool,
        denormalized_path_delimiter: str = ".",
        denormalized_static_value: str = "$",
    ):
        """
        Initialize the base adapter.

        Parameters
        ----------
        vector_store : T
            The vector store instance.
        use_normalized_metadata : bool
            Indicates whether normalized metadata is used.
        denormalized_path_delimiter : str
            Delimiter for denormalized metadata keys.
        denormalized_static_value : str
            Value to use for denormalized metadata entries.
        """
        self.vector_store = vector_store
        self.use_normalized_metadata = use_normalized_metadata
        self.denormalized_path_delimiter = denormalized_path_delimiter
        self.denormalized_static_value = denormalized_static_value

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

    @abc.abstractmethod
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
        return await run_in_executor(
            None,
            self.similarity_search_with_embedding_by_vector,
            embedding,
            k,
            filter,
            **kwargs,
        )

    @abc.abstractmethod
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
        return await run_in_executor(
            None,
            self.get,
            ids,
            **kwargs,
        )

    def get_adjacent(
        self,
        outgoing_edges: set[Edge],
        strategy: Strategy,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """
        Return the docs with incoming edges from any of the given edges.

        Parameters
        ----------
        outgoing_edges : set[Edge]
            The edges to look for.
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
        """
        results: list[Document] = []
        for outgoing_edge in outgoing_edges:
            docs = self.similarity_search_with_embedding_by_vector(
                embedding=strategy._query_embedding,
                k=strategy.adjacent_k,
                filter=self._get_metadata_filter(
                    base_filter=filter, edge=outgoing_edge
                ),
                **kwargs,
            )
            results.extend(docs)
            if not self.use_normalized_metadata:
                # If we denormalized the metadata, we actually do two queries.
                # One, for normalized values (above) and one for denormalized.
                # This ensures that cases where the key had a single value are
                # caught as well. This could *maybe* be handled differently if
                # we know keys that were always denormalized.
                docs = self.similarity_search_with_embedding_by_vector(
                    embedding=strategy._query_embedding,
                    k=strategy.adjacent_k,
                    filter=self._get_metadata_filter(
                        base_filter=filter, edge=outgoing_edge, denormalize_edge=True
                    ),
                    **kwargs,
                )
                results.extend(docs)
        return results

    async def aget_adjacent(
        self,
        outgoing_edges: set[Edge],
        strategy: Strategy,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """
        Asynchronously return the docs with incoming edges from any of the given edges.

        Parameters
        ----------
        outgoing_edges : set[Edge]
            The edges to look for.
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
        """
        tasks = [
            self.asimilarity_search_with_embedding_by_vector(
                embedding=strategy._query_embedding,
                k=strategy.adjacent_k,
                filter=self._get_metadata_filter(
                    base_filter=filter, edge=outgoing_edge
                ),
                **kwargs,
            )
            for outgoing_edge in outgoing_edges
        ]
        if not self.use_normalized_metadata:
            # If we denormalized the metadata, we actually do two queries.
            # One, for normalized values (above) and one for denormalized.
            # This ensures that cases where the key had a single value are
            # caught as well. This could *maybe* be handled differently if
            # we know keys that were always denormalized.
            tasks.extend(
                [
                    self.asimilarity_search_with_embedding_by_vector(
                        embedding=strategy._query_embedding,
                        k=strategy.adjacent_k,
                        filter=self._get_metadata_filter(
                            base_filter=filter,
                            edge=outgoing_edge,
                            denormalize_edge=True,
                        ),
                        **kwargs,
                    )
                    for outgoing_edge in outgoing_edges
                ]
            )

        results: list[Document] = []
        for completed_task in asyncio.as_completed(tasks):
            docs = await completed_task
            results.extend(docs)
        return results

    def _get_metadata_filter(
        self,
        base_filter: dict[str, Any] | None = None,
        edge: Edge | None = None,
        denormalize_edge: bool = False,
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
        denormalize_edge : bool, default False
            Whether edges should be denormalized.

        Returns
        -------
        dict[str, Any]
            The metadata dictionary to use for the given filter.
        """
        metadata_filter = {**(base_filter or {})}
        if edge is None:
            metadata_filter
        elif denormalize_edge:
            metadata_filter[
                f"{edge.key}{self.denormalized_path_delimiter}{edge.value}"
            ] = self.denormalized_static_value
        else:
            metadata_filter[edge.key] = edge.value
        return metadata_filter
