"""Defines the base class for vector store adapters."""

import abc
import asyncio
from collections.abc import Iterable, Sequence
from typing import Any

from immutabledict import immutabledict

from graph_retriever.content import Content
from graph_retriever.edges import Edge, IdEdge, MetadataEdge
from graph_retriever.utils.run_in_executor import run_in_executor
from graph_retriever.utils.top_k import top_k


class Adapter(abc.ABC):
    """
    Base adapter for integrating vector stores with the graph retriever system.

    This class provides a foundation for custom adapters, enabling consistent
    interaction with various vector store implementations.
    """

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        """
        Return content items most similar to the query.

        Also returns the embedded query vector.

        Parameters
        ----------
        query :
            Input text.
        k :
            Number of content items to return.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments.

        Returns
        -------
        query_embedding :
            The query embedding used for selecting the most relevant content.
        contents :
            List of up to `k` content items most similar to the query vector.
        """
        ...

    async def asearch_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        """
        Asynchronously return content items most similar to the query.

        Also returns the embedded query vector.

        Parameters
        ----------
        query :
            Input text.
        k :
            Number of content items to return.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments.

        Returns
        -------
        query_embedding :
            The query embedding used for selecting the most relevant content.
        contents :
            List of up to `k` content items most similar to the query
            vector.
        """
        return await run_in_executor(
            None, self.search_with_embedding, query, k, filter, **kwargs
        )

    @abc.abstractmethod
    def search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        """
        Return content items most similar to the query vector.

        Parameters
        ----------
        embedding :
            The query embedding used for selecting the most relevant content.
        k :
            Number of content items to return.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments.

        Returns
        -------
        :
            List of content items most similar to the query vector.
        """
        ...

    async def asearch(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        """
        Asynchronously return content items most similar to the query vector.

        Parameters
        ----------
        embedding :
            The query embedding used for selecting the most relevant content.
        k :
            Number of content items to return.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments.

        Returns
        -------
        :
            List of content items most similar to the query vector.
        """
        return await run_in_executor(
            None,
            self.search,
            embedding,
            k,
            filter,
            **kwargs,
        )

    @abc.abstractmethod
    def get(
        self,
        ids: Sequence[str],
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        """
        Get content items by ID.

        Fewer content items may be returned than requested if some IDs are
        not found or if there are duplicated IDs. This method should **NOT**
        raise exceptions if no content items are found for some IDs.

        Users should not assume that the order of the returned content items
        matches  the order of the input IDs. Instead, users should rely on
        the ID field of the returned content items.

        Parameters
        ----------
        ids :
            List of IDs to get.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments. These are up to the implementation.

        Returns
        -------
        :
            List of content items that were found.
        """
        ...

    async def aget(
        self,
        ids: Sequence[str],
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        """
        Asynchronously get content items by ID.

        Fewer content items may be returned than requested if some IDs are
        not found or if there are duplicated IDs. This method should **NOT**
        raise exceptions if no content items are found for some IDs.

        Users should not assume that the order of the returned content items
        matches  the order of the input IDs. Instead, users should rely on
        the ID field of the returned content items.

        Parameters
        ----------
        ids :
            List of IDs to get.
        filter :
            Filter on the metadata to apply.
        kwargs :
            Additional keyword arguments. These are up to the implementation.

        Returns
        -------
        :
            List of content items that were found.
        """
        return await run_in_executor(
            None,
            self.get,
            ids,
            filter,
            **kwargs,
        )

    def adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        """
        Return the content items with at least one matching incoming edge.

        Parameters
        ----------
        edges :
            The edges to look for.
        query_embedding :
            The query embedding used for selecting the most relevant content.
        k :
            The number of relevant content items to select.
        filter :
            Optional metadata to filter the results.
        kwargs :
            Keyword arguments to pass to the similarity search.

        Returns
        -------
        :
            Iterable of adjacent content items.

        Raises
        ------
        ValueError
            If unsupported edge types are encountered.
        """
        results: list[Content] = []

        ids = []
        for edge in edges:
            if isinstance(edge, MetadataEdge):
                docs = self.search(
                    embedding=query_embedding,
                    k=k,
                    filter=self._metadata_filter(base_filter=filter, edge=edge),
                    **kwargs,
                )
                results.extend(docs)
            elif isinstance(edge, IdEdge):
                ids.append(edge.id)
            else:
                raise ValueError(f"Unsupported edge: {edge}")

        if ids:
            results.extend(self.get(ids, filter=filter))

        return top_k(
            results,
            embedding=query_embedding,
            k=k,
        )

    async def aadjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        """
        Asynchronously return the content items with at least one matching edge.

        Parameters
        ----------
        edges :
            The edges to look for.
        query_embedding :
            The query embedding used for selecting the most relevant content.
        k :
            The number of relevant content items to select for the edges.
        filter :
            Optional metadata to filter the results.
        kwargs :
            Keyword arguments to pass to the similarity search.

        Returns
        -------
        :
            Iterable of adjacent content items.

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
                    self.asearch(
                        embedding=query_embedding,
                        k=k,
                        filter=self._metadata_filter(base_filter=filter, edge=edge),
                        **kwargs,
                    )
                )
            elif isinstance(edge, IdEdge):
                ids.append(edge.id)
            else:
                raise ValueError(f"Unsupported edge: {edge}")

        if ids:
            tasks.append(self.aget(ids, filter))

        results: list[Content] = [
            c
            for completed_task in asyncio.as_completed(tasks)
            for c in await completed_task
        ]

        return top_k(
            results,
            embedding=query_embedding,
            k=k,
        )

    def _metadata_filter(
        self,
        edge: Edge,
        base_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return a filter for the `base_filter` and incoming edges from `edge`.

        Parameters
        ----------
        base_filter :
            Any base metadata filter that should be used for search.
            Generally corresponds to the user specified filters for the entire
            traversal. Should be combined with the filters necessary to support
            nodes with an *incoming* edge matching `edge`.
        edge :
            An optional edge which should be added to the filter.

        Returns
        -------
        :
            The metadata dictionary to use for the given filter.
        """
        assert isinstance(edge, MetadataEdge)
        value = edge.value
        if isinstance(value, immutabledict):
            value = dict(value)
        return {edge.incoming_field: value, **(base_filter or {})}
