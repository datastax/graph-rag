"""Provides a graph-based retriever combining vector search and graph traversal."""

from collections.abc import Sequence
from functools import cached_property
from typing import (
    Any,
)

from graph_retriever import Adapter, EdgeFunction, EdgeSpec, atraverse, traverse
from graph_retriever.strategies import Eager, Strategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, computed_field, model_validator
from typing_extensions import Self

from langchain_graph_retriever._conversion import node_to_doc
from langchain_graph_retriever.adapters.inference import infer_adapter


# this class uses pydantic, so store must be provided at init time.
class GraphRetriever(BaseRetriever):
    """
    Retriever combining vector search and graph traversal.

    The `GraphRetriever` class performs retrieval by first using vector search to find
    relevant documents, and then applying graph traversal to explore connected
    documents. It supports multiple traversal strategies and integrates seamlessly
    with LangChain's retriever framework.

    Parameters
    ----------
    store : Adapter | VectorStore
        The vector store or adapter used for document retrieval.
    edges : list[EdgeSpec] | EdgeFunction, default []
        Function to use for extracting edges from nodes. May be passed a list of
        arguments to construct a `MetadataEdgeFunction` from, or an
        `EdgeFunction`.
    strategy : Strategy, default Eager()
        The traversal strategy to use.
        Defaults to an `Eager` (breadth-first) strategy which explores
        the top `adjacent_k` for each edge.

    Attributes
    ----------
    store : Adapter | VectorStore
        The vector store or adapter used for document retrieval.
    edges : list[str | tuple[str, str | Id]] | EdgeFunction
        Definitions of edges used for graph traversal.
    strategy : Strategy
        The traversal strategy to use.
    """

    store: Adapter | VectorStore
    edges: list[EdgeSpec] | EdgeFunction = []
    strategy: Strategy = Eager()

    # Capture the extra fields in `self.model_extra` rather than ignoring.
    model_config = ConfigDict(extra="allow")

    # Move the `k` extra argument (if any) to the strategy.
    @model_validator(mode="after")
    def apply_extra(self) -> Self:
        """
        Apply extra configuration to the traversal strpategy.

        This method captures additional fields provided in `model_extra` and applies
        them to the current traversal strategy. Any extra fields are cleared after
        they are applied.

        Returns
        -------
        Self
            The updated `GraphRetriever` instance.
        """
        if self.model_extra:
            self.strategy = self.strategy.model_validate(
                {**self.strategy.model_dump(), **self.model_extra}
            )
            self.model_extra.clear()
        return self

    @computed_field  # type: ignore
    @cached_property
    def adapter(self) -> Adapter:
        """The adapter to use during traversals."""
        return infer_adapter(self.store)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        edges: list[EdgeSpec] | EdgeFunction | None = None,
        initial_roots: Sequence[str] = (),
        filter: dict[str, Any] | None = None,
        store_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Document]:
        """
        Retrieve doc nodes using graph traversal and similarity search.

        This method first retrieves documents based on similarity to the query, and
        then applies a traversal strategy to explore connected nodes in the graph.

        Parameters
        ----------
        query : str
            The query string to search for.
        edges : list[EdgeSpec] | EdgeFunction, optional
            Optional edge definitions for this retrieval.
        initial_roots : Sequence[str]
            Document IDs to use as initial roots. The top `adjacent_k` nodes
            connected to each root are included in the initial candidates.
        filter : dict[str, Any], optional
            Optional metadata filter to apply.
        store_kwargs : dict[str, Any], optional
            Additional keyword arguments for the store.
        **kwargs : Any
            Additional arguments for configuring the traversal strategy.

        Returns
        -------
        list[Document]
            The retrieved documents.

        Raises
        ------
        ValueError
            If edges weren't provided in this call or the constructor.
        """
        edges = edges or self.edges
        if edges is None:
            raise ValueError("'edges' must be provided in this call or the constructor")

        nodes = traverse(
            query=query,
            edges=edges,
            strategy=Strategy.build(base_strategy=self.strategy, **kwargs),
            store=self.adapter,
            metadata_filter=filter,
            initial_root_ids=initial_roots,
            store_kwargs=store_kwargs,
        )
        return [node_to_doc(n) for n in nodes]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        edges: list[EdgeSpec] | EdgeFunction | None = None,
        initial_roots: Sequence[str] = (),
        filter: dict[str, Any] | None = None,
        store_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Document]:
        """
        Asynchronously retrieve doc nodes using graph traversal and similarity search.

        This method first retrieves documents based on similarity to the query, and
        then applies a traversal strategy to explore connected nodes in the graph.

        Parameters
        ----------
        query : str
            The query string to search for.
        edges : list[EdgeSpec] | EdgeFunction, optional
            Override edge definitions for this invocation.
        initial_roots : Sequence[str]
            Document IDs to use as initial roots. The top `adjacent_k` nodes
            connected to each root are included in the initial candidates.
        filter : dict[str, Any], optional
            Optional metadata filter to apply.
        store_kwargs : dict[str, Any], optional
            Additional keyword arguments for the store.
        **kwargs : Any
            Additional arguments for configuring the traversal strategy.

        Returns
        -------
        list[Document]
            The retrieved documents.

        Raises
        ------
        ValueError
            If edges weren't provided in this call or the constructor.
        """
        edges = edges or self.edges
        if edges is None:
            raise ValueError("'edges' must be provided in this call or the constructor")
        nodes = await atraverse(
            query=query,
            edges=edges,
            strategy=Strategy.build(base_strategy=self.strategy, **kwargs),
            store=self.adapter,
            metadata_filter=filter,
            initial_root_ids=initial_roots,
            store_kwargs=store_kwargs,
        )
        return [node_to_doc(n) for n in nodes]
