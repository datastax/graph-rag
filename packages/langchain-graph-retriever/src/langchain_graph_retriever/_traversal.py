from typing import Any, Sequence
from langchain_core.documents import Document

from .adapters.base import Adapter
from .strategies import Strategy
from .edge_helper import EdgeHelper

class Traversal():
    """Performs a single traversal.

    This should *not* be reused between traversals.
    """

    def __init__(
        self,
        query: str,
        *,
        edges: EdgeHelper,
        strategy: Strategy,
        store: Adapter,
        metadata_filter: dict[str, Any] | None = None,
        initial_root_ids: Sequence[str] = ()
    ) -> None:
        self._used = False

        self.query = query
        self.edges = edges
        self.strategy = strategy
        self.store = store
        self.metadata_filter = metadata_filter
        self.initial_root_ids = initial_root_ids

    def _check_first_use(self):
        assert not self._used, "Traversals cannot be re-used."
        self._used = True

    def traverse(self) -> list[Document]:
        self._check_first_use()

        # Retrieve initial candidates.
        initial_docs = self._fetch_initial_candidates(
            self.query, state=state, filter=filter, **store_kwargs
        )
        state.add_docs(initial_docs, depth=0)

        if initial_roots:
            neighborhood_adjacent_docs = self._fetch_neighborhood_candidates(
                initial_roots,
                state=state,
                filter=filter,
                **store_kwargs,
            )
            state.add_docs(neighborhood_adjacent_docs, depth=0)

        while True:
            # Select the next batch of nodes, and (new) outgoing edges.
            next_outgoing_edges = state.select_next_edges()
            if next_outgoing_edges is None:
                break
            elif next_outgoing_edges:
                # Find the (new) document with incoming edges from those edges.
                adjacent_docs = self.adapter.get_adjacent(
                    outgoing_edges=next_outgoing_edges,
                    strategy=state.strategy,
                    filter=filter,
                    **store_kwargs,
                )

                state.add_docs(adjacent_docs)

        return state.finish()

    async def atraverse(self) -> list[Document]:
        self._check_first_use()

        initial_docs = self._fetch_initial_candidates(
            self.query, state=state, filter=filter, **store_kwargs
        )
        state.add_docs(initial_docs, depth=0)

        if initial_roots:
            neighborhood_adjacent_docs = self._fetch_neighborhood_candidates(
                initial_roots,
                state=state,
                filter=filter,
                **store_kwargs,
            )
            state.add_docs(neighborhood_adjacent_docs, depth=0)

        while True:
            # Select the next batch of nodes, and (new) outgoing edges.
            next_outgoing_edges = state.select_next_edges()
            if next_outgoing_edges is None:
                break
            elif next_outgoing_edges:
                # Find the (new) document with incoming edges from those edges.
                adjacent_docs = self.adapter.get_adjacent(
                    outgoing_edges=next_outgoing_edges,
                    strategy=state.strategy,
                    filter=filter,
                    **store_kwargs,
                )

                state.add_docs(adjacent_docs)

        return st

    def _fetch_initial_candidates(self) -> list[Document]:
        """Get the embedded query and the set of initial candidates.

        Args:
            query: String to compute embedding and fetch initial matches for.
            state: The travel state we're retrieving candidates fore.
            filter: Optional metadata filter to apply.
            **kwargs: Additional keyword arguments.

        """
        query_embedding, docs = self.store.similarity_search_with_embedding(
            query=query,
            k=state.strategy.start_k,
            filter=filter,
            **kwargs,
        )
        state.strategy.query_embedding = query_embedding
        return docs

    async def _afetch_initial_candidates(
        self,
        query: str,
        *,
        state: _TraversalState,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        query_embedding, docs = await self.adapter.asimilarity_search_with_embedding(
            query=query,
            k=state.strategy.start_k,
            filter=filter,
            **kwargs,
        )
        state.strategy.query_embedding = query_embedding
        return docs

    def _fetch_neighborhood_candidates(
        self,
        neighborhood: Sequence[str],
        *,
        state: _TraversalState,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        neighborhood_docs = self.adapter.get(neighborhood)
        neighborhood_nodes = state.add_docs(neighborhood_docs)

        # Record the neighborhood nodes (specifically the outgoing edges from the
        # neighborhood) as visited.
        outgoing_edges = state.visit_nodes(neighborhood_nodes.values())

        # Fetch the candidates.
        return self.adapter.get_adjacent(
            outgoing_edges=outgoing_edges,
            strategy=state.strategy,
            filter=filter,
            **kwargs,
        )

    async def _afetch_neighborhood_candidates(
        self,
        neighborhood: Sequence[str],
        *,
        state: _TraversalState,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ):
        neighborhood_docs = await self.adapter.aget(neighborhood)
        neighborhood_nodes = state.add_docs(neighborhood_docs)

        # Record the neighborhood nodes (specifically the outgoing edges from the
        # neighborhood) as visited.
        outgoing_edges = state.visit_nodes(neighborhood_nodes.values())

        # Fetch the candidates.
        return await self.adapter.aget_adjacent(
            outgoing_edges=outgoing_edges,
            strategy=state.strategy,
            filter=filter,
            **kwargs,
        )