"""Provide LangChain retriever combining vector search and graph traversal."""

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
    """Retriever combining vector-search and graph traversal."""

    store: Adapter | VectorStore
    edges: list[str | tuple[str, str]]
    strategy: Strategy = Eager()

    @computed_field
    def k(self) -> int:
        """Return the (maximum) number of documents which will be fetched."""
        return self.strategy.k

    # Capture the extra fields in `self.model_extra` rather than ignoring.
    model_config = ConfigDict(extra="allow")

    # Move the `k` extra argument (if any) to the strategy.
    @model_validator(mode="after")
    def apply_extra(self) -> Self:
        """Apply the value of `k` to the strategy."""
        if self.model_extra:
            self.strategy = self.strategy.model_validate(
                {**self.strategy.model_dump(), **self.model_extra}
            )
            self.model_extra.clear()
        return self

    def _edge_helper(
        self, edges: list[str | tuple[str, str]] | None = None
    ) -> EdgeHelper:
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
        """Retrieve document nodes from this graph vector store using MMR-traversal.

        This strategy first retrieves the top `start_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`. At each step,
        it considers the (remaining) documents from `start_k` as well as any
        documents connected by edges to a selected document retrieved based on
        similarity (a "root").

        Args:
            query: The query string to search for.
            strategy: Specify or override the strategy to use for this retrieval.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `start_k = 0`.
            filter: Optional metadata to filter the results.
            store_kwargs: Optional kwargs passed to queries to the store.
            **kwargs: Additional keyword arguments passed to traversal state.

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
        """Asynchronously retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `start_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.
        At each step, it considers the (remaining) documents from `start_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `start_k = 0`.
            filter: Optional metadata to filter the results.
            store_kwargs: Optional kwargs passed to queries to the store.
            **kwargs: Additional keyword arguments passed to traversal state.

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
