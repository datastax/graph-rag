"""Provides a graph-based retriever combining vector search and graph traversal."""

from collections.abc import Sequence
from functools import cached_property
from typing import (
    Any,
)

from graph_retriever import atraverse, traverse
from graph_retriever.adapters import Adapter
from graph_retriever.edges import EdgeFunction, EdgeSpec
from graph_retriever.strategies import Eager, Strategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.base import VectorStore
from pydantic import ConfigDict, computed_field, model_validator
from typing_extensions import Self

from langchain_graph_retriever._conversion import node_to_doc
from langchain_graph_retriever.adapters.inference import infer_adapter


# this class uses pydantic, so store must be provided at init time.
class GraphRetriever(BaseRetriever):
    """
    Retriever combining vector search and graph traversal.

    The [GraphRetriever][langchain_graph_retriever.GraphRetriever] class
    retrieves documents by first performing a vector search to identify relevant
    documents, followed by graph traversal to explore their connections. It
    supports multiple traversal strategies and integrates seamlessly with
    LangChain's retriever framework.

    Parameters
    ----------
    store : Adapter | VectorStore
        The adapter or vector store used for document retrieval.
    edges : list[EdgeSpec] | EdgeFunction, default []
        A list of [EdgeSpec][graph_retriever.edges.EdgeSpec] for use in creating a
        [MetadataEdgeFunction][graph_retriever.edges.MetadataEdgeFunction],
        or an [EdgeFunction][graph_retriever.edges.EdgeFunction].
    strategy : Strategy, default Eager
        The traversal strategy to use.
        Defaults to an [Eager][graph_retriever.strategies.Eager] (breadth-first)
        strategy which explores the top `adjacent_k` for each edge.

    Attributes
    ----------
    store : Adapter | VectorStore
        The adapter or vector store used for document retrieval.
    edges : list[EdgeSpec] | EdgeFunction
        A list of [EdgeSpec][graph_retriever.edges.EdgeSpec] for use in creating a
        [MetadataEdgeFunction][graph_retriever.edges.MetadataEdgeFunction],
        or an [EdgeFunction][graph_retriever.edges.EdgeFunction].
    strategy : Strategy
        The traversal strategy to use.
    adapter: Adapter
        The adapter to use during traversals
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
        Apply extra configuration to the traversal strategy.

        This method captures additional fields provided in the `model_extra` argument
        and applies them to the current traversal strategy. Any extra fields are
        cleared after they are applied.

        Returns
        -------
        : GraphRetriever
            The updated GraphRetriever instance.
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

        Notes
        -----
        You can execute this method by calling `.invoke()` on the retriever.

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
        kwargs : dict, optional
            Additional arguments for configuring the traversal strategy.

        Returns
        -------
        :
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

        Notes
        -----
        You can execute this method by calling `.ainvoke()` on the retriever.

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
        kwargs : dict, optional
            Additional arguments for configuring the traversal strategy.

        Returns
        -------
        :
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
