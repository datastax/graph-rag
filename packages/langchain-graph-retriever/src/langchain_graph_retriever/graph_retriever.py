"""Provides a graph-based retriever combining vector search and graph traversal."""

from collections.abc import Sequence
from functools import cached_property
from typing import (
    Any,
    Self,
)

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, computed_field, model_validator

from ._traversal import Traversal
from .adapters.base import Adapter
from .adapters.inference import infer_adapter
from .edge_helper import EdgeHelper
from .strategies import Eager, Strategy


# this class uses pydantic, so store must be provided at init time.
class GraphRetriever(BaseRetriever):
    """Retriever combining vector search and graph traversal.

    The `GraphRetriever` class performs retrieval by first using vector search to find
    relevant documents, and then applying graph traversal to explore connected
    documents. It supports multiple traversal strategies and integrates seamlessly
    with LangChain's retriever framework.

    Parameters
    ----------
    store : Adapter | VectorStore
        The vector store or adapter used for document retrieval.
    edges : list[str | tuple[str, str]]
        Definitions of edges used for graph traversal.
    strategy : Strategy
        The traversal strategy to use. Defaults to an `Eager`
        (breadth-first) strategy.

    Attributes
    ----------
    store : Adapter | VectorStore
        The vector store or adapter used for document retrieval.
    edges : list[str | tuple[str, str]]
        Definitions of edges used for graph traversal.
    strategy : Strategy
        The traversal strategy to use.
    """

    store: Adapter | VectorStore
    edges: list[str | tuple[str, str]]
    strategy: Strategy = Eager()

    # Capture the extra fields in `self.model_extra` rather than ignoring.
    model_config = ConfigDict(extra="allow")

    # Move the `k` extra argument (if any) to the strategy.
    @model_validator(mode="after")
    def apply_extra(self) -> Self:
        """Apply extra configuration to the traversal strategy.

        This method captures additional fields provided in `model_extra` and applies
        them to the current traversal strategy. Any extra fields are cleared after
        they are applied.

        Returns
        -------
            Self: The updated `GraphRetriever` instance.
        """
        if self.model_extra:
            self.strategy = self.strategy.model_validate(
                {**self.strategy.model_dump(), **self.model_extra}
            )
            self.model_extra.clear()
        return self

    def _edge_helper(
        self, edges: list[str | tuple[str, str]] | None = None
    ) -> EdgeHelper:
        """Create an `EdgeHelper` instance for managing edges during traversal.

        Args:
            edges (list[str | tuple[str, str]] | None): Optional edge definitions.
                If not provided, uses the default edge definitions from the retriever.

        Returns
        -------
            EdgeHelper: An instance of `EdgeHelper` configured with the specified edges.
        """
        return EdgeHelper(
            edges=self.edges if edges is None else edges,
            denormalized_path_delimiter=self.adapter.denormalized_path_delimiter,
            denormalized_static_value=self.adapter.denormalized_static_value,
            use_normalized_metadata=self.adapter.use_normalized_metadata,
        )

    @computed_field  # type: ignore
    @cached_property
    def adapter(self) -> Adapter:
        """The adapter to use during traversals."""
        return infer_adapter(self.store)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        edges: list[str | tuple[str, str]] | None = None,
        initial_roots: Sequence[str] = (),
        filter: dict[str, Any] | None = None,
        store_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve doc nodes using graph traversal and similarity search.

        This method first retrieves documents based on similarity to the query, and
        then applies a traversal strategy to explore connected nodes in the graph.

        Args:
            query (str): The query string to search for.
            edges (list[str | tuple[str, str]] | None): Optional edge definitions for
                this retrieval.
            initial_roots (Sequence[str]): Document IDs to use as initial roots. The top
                `adjacent_k` nodes connected to each root are included in the initial
                candidates.
            filter (dict[str, Any] | None): Optional metadata filter to apply.
            store_kwargs (dict[str, Any]): Additional keyword arguments for the store.
            **kwargs (Any): Additional arguments for configuring the traversal strategy.

        Returns
        -------
            list[Document]: The retrieved documents.
        """
        traversal = Traversal(
            query=query,
            edges=self._edge_helper(edges),
            strategy=Strategy.build(base_strategy=self.strategy, **kwargs),
            store=self.adapter,
            metadata_filter=filter,
            initial_root_ids=initial_roots,
            store_kwargs=store_kwargs,
        )

        return traversal.traverse()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        edges: list[str | tuple[str, str]] | None = None,
        initial_roots: Sequence[str] = (),
        filter: dict[str, Any] | None = None,
        store_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Document]:
        """
        Asynchronously retrieve doc nodes using graph traversal and similarity search.

        This method first retrieves documents based on similarity to the query, and
        then applies a traversal strategy to explore connected nodes in the graph.

        Args:
            query (str): The query string to search for.
            edges (list[str | tuple[str, str]] | None): Optional edge definitions for
                this retrieval.
            initial_roots (Sequence[str]): Document IDs to use as initial roots. The top
                `adjacent_k` nodes connected to each root are included in the initial
                candidates.
            filter (dict[str, Any] | None): Optional metadata filter to apply.
            store_kwargs (dict[str, Any]): Additional keyword arguments for the store.
            **kwargs (Any): Additional arguments for configuring the traversal strategy.

        Returns
        -------
            list[Document]: The retrieved documents.
        """
        traversal = Traversal(
            query=query,
            edges=self._edge_helper(edges),
            strategy=Strategy.build(base_strategy=self.strategy, **kwargs),
            store=self.adapter,
            metadata_filter=filter,
            initial_root_ids=initial_roots,
            store_kwargs=store_kwargs,
        )

        return await traversal.atraverse()
