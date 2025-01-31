"""Defines the base class for vector store adapters."""

import abc
import asyncio
from collections.abc import Iterable, Sequence
from typing import Any

from graph_retriever.content import Content
from graph_retriever.types import Edge, IdEdge, MetadataEdge
from graph_retriever.utils.run_in_executor import run_in_executor


class Adapter(abc.ABC):
    """
    Base adapter for integrating vector stores with the graph retriever system.

    This class provides a foundation for custom adapters, enabling consistent
    interaction with various vector store implementations.
    """

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Return the embedding of the query."""
        ...

    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        """
        Return content most similar to the query.

        Also returns the embedded query vector.

        Parameters
        ----------
        query : str
            Input text.
        k : int, default 4
            Number of `Content` items to return.
        filter : dict[str, str], optional
            Filter on the metadata to apply.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        query_embedding : list[float]
            The embedded query vector
        contents : list[Content]
            List of up to `k` `Content` most similar to the query vector.
        """
        query_embedding = self.embed_query(query)
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
    ) -> tuple[list[float], list[Content]]:
        """
        Asynchronously return content most similar to the query.

        Also returns the embedded query vector.

        Parameters
        ----------
        query : str
            Input text.
        k : int, default 4
            Number of `Content` items to return.
        filter : dict[str, str], optional
            Filter on the metadata to apply.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        query_embedding : list[float]
            The embedded query vector
        contents : list[Content]
            List of up to `k` `Content` most similar to the query
            vector.
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
    ) -> list[Content]:
        """
        Return content most similar to the query vector.

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
            List of `Content` most similar to the query vector.
        """
        ...

    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        """
        Asynchronously return content most similar to the query vector.

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
            List of contetn most similar to the query vector.
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
    ) -> list[Content]:
        """
        Get content by id.

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
        list[Content]
            List of content that was found.
        """
        ...

    async def aget(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Content]:
        """
        Asynchronously get content by id.

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
        list[Content]
            List of content that were found.
        """
        return await run_in_executor(
            None,
            self.get,
            ids,
            **kwargs,
        )

    def get_adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        """
        Return the content with at least one matching incoming edge.

        Parameters
        ----------
        edges : set[Edge]
            The edges to look for.
        query_embedding : list[float]
            The query embedding used for selecting the most relevant nodes.
        k : int
            The number of relevant content items to select.
        strategy : Strategy
            The traversal strategy being used.
        filter : dict[str, Any], optional
            Optional metadata to filter the results.
        **kwargs : Any
            Keyword arguments to pass to the similarity search.

        Returns
        -------
        Iterable[Content]
            Iterable of adjacent content.

        Raises
        ------
        ValueError
            If unsupported edge types are encountered.
        """
        results: list[Content] = []

        ids = []
        for edge in edges:
            if isinstance(edge, MetadataEdge):
                docs = self.similarity_search_with_embedding_by_vector(
                    embedding=query_embedding,
                    k=k,
                    filter=self._get_metadata_filter(base_filter=filter, edge=edge),
                    **kwargs,
                )
                results.extend(docs)
            elif isinstance(edge, IdEdge):
                ids.append(edge.id)
            else:
                raise ValueError(f"Unsupported edge: {edge}")

        if ids:
            results.extend(self.get(ids))
        return results

    async def aget_adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        """
        Asynchronously return the content with at least one matching edge.

        Parameters
        ----------
        edges : set[Edge]
            The edges to look for.
        query_embedding : list[float]
            The query embedding used for selecting the most relevant nodes.
        k : int
            The number of relevant content items to select for the edges.
        filter : dict[str, Any], optional
            Optional metadata to filter the results.
        **kwargs : Any
            Keyword arguments to pass to the similarity search.

        Returns
        -------
        Iterable[Content]
            Iterable of adjacent contents.

        Raises
        ------
        ValueError
            If unsupported edge types are encountered.
        """
        tasks = []
        ids = []
        for edge in edges:
            if isinstance(edge, MetadataEdge):
                tasks.append(
                    self.asimilarity_search_with_embedding_by_vector(
                        embedding=query_embedding,
                        k=k,
                        filter=self._get_metadata_filter(base_filter=filter, edge=edge),
                        **kwargs,
                    )
                )
            elif isinstance(edge, IdEdge):
                ids.append(edge.id)
            else:
                raise ValueError(f"Unsupported edge: {edge}")

        if ids:
            tasks.append(self.aget(ids))

        results: list[Content] = []
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
