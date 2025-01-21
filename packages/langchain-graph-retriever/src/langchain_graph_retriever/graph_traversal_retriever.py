from typing import (
    Any,
    Iterable,
    List,
    Sequence,
    Tuple,
    Union,
)

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import computed_field

from .edge import Edge
from .edge_helper import EdgeHelper
from .node import Node
from .adapters.base import METADATA_EMBEDDING_KEY, Adapter
from .strategy.base import Strategy

INFINITY = float("inf")


class _TraversalState:
    """Manages the book-keeping necessary to keep track of traversal state."""

    def __init__(
        self,
        *,
        edge_helper: EdgeHelper,
        base_strategy: Strategy | None,
        strategy: Strategy | dict[str, Any] | None,
    ) -> None:
        self.edge_helper = edge_helper

        # Deep copy in case the strategy has mutable state
        if isinstance(strategy, Strategy):
            self.strategy = strategy.model_copy(deep=True)
        elif isinstance(strategy, dict):
            assert (
                base_strategy is not None
            ), "Must set strategy in init to support field-overrides."
            self.strategy = base_strategy.model_copy(update=strategy, deep=True)
        elif strategy is None:
            assert base_strategy is not None, "Must set strategy in init or invocation."
            self.strategy = base_strategy.model_copy(deep=True)
        else:
            raise ValueError(f"Unsupported strategy {strategy}")

        self.visited_edges: set[Edge] = set()
        self.edge_depths: dict[Edge, int] = {}
        self.doc_cache: dict[str, Document] = {}
        self.node_cache: dict[str, Node] = {}

        self.selected_nodes: dict[str, Node] = {}

    def _doc_to_new_node(
        self, doc: Document, *, depth: int | None = None
    ) -> Node | None:
        if doc.id is None:
            raise ValueError("All documents should have ids")
        if doc.id in self.node_cache:
            return None

        doc = self.doc_cache.setdefault(doc.id, doc)
        assert doc.id is not None
        incoming_edges, outgoing_edges = self.edge_helper.get_incoming_outgoing(
            doc.metadata
        )
        if depth is None:
            depth = min(
                [
                    d
                    for e in incoming_edges
                    if (d := self.edge_depths.get(e, None)) is not None
                ],
                default=0,
            )
        node = Node(
            id=doc.id,
            depth=depth,
            embedding=doc.metadata[METADATA_EMBEDDING_KEY],
            metadata=doc.metadata,
            incoming_edges=incoming_edges,
            outgoing_edges=outgoing_edges,
        )
        self.node_cache[doc.id] = node

        return node

    def add_docs(
        self, docs: Iterable[Document], *, depth: int | None = None
    ) -> dict[str, Node]:
        # Record the depth of new nodes.
        nodes = {
            node.id: node
            for doc in docs
            if (node := self._doc_to_new_node(doc, depth=depth)) is not None
            if (
                self.strategy.max_depth is None or node.depth <= self.strategy.max_depth
            )
        }
        self.strategy.add_nodes(nodes)
        return nodes

    def visit_nodes(self, nodes: Iterable[Node]) -> set[Edge]:
        """Record the nodes as visited, returning the new outgoing edges.

        After this call, the outgoing edges will be added to the visited
        set, and not revisited during the traversal.
        """
        new_outgoing_edges: dict[Edge, int] = {}
        for node in nodes:
            node_new_outgoing_edges = node.outgoing_edges - self.visited_edges
            for edge in node_new_outgoing_edges:
                depth = new_outgoing_edges.setdefault(edge, node.depth + 1)
                if node.depth + 1 < depth:
                    new_outgoing_edges[edge] = node.depth + 1

        self.edge_depths.update(new_outgoing_edges)

        new_outgoing_edge_set = set(new_outgoing_edges.keys())
        self.visited_edges.update(new_outgoing_edge_set)
        return new_outgoing_edge_set

    def select_next_edges(self) -> set[Edge] | None:
        """Select the next round of nodes.

        Returns the set of new edges that need to be explored.
        """
        remaining = self.strategy.k - len(self.selected_nodes)

        if remaining <= 0:
            return None

        next_nodes = self.strategy.select_nodes(limit=remaining)
        if not next_nodes:
            return None

        next_nodes = [n for n in next_nodes if n.id not in self.selected_nodes]
        if len(next_nodes) == 0:
            return None

        self.selected_nodes.update({n.id: n for n in next_nodes})
        new_outgoing_edges = self.visit_nodes(next_nodes)
        return new_outgoing_edges

    def finish(self) -> list[Document]:
        final_nodes = self.strategy.finalize_nodes(self.selected_nodes.values())
        docs = []
        for node in final_nodes:
            doc = self.doc_cache.get(node.id, None)
            if doc is None:
                raise RuntimeError(
                    f"unexpected, cache should contain doc id: {node.id}"
                )
            # Compute new metadata from extra metadata and metadata.
            # This allows us to avoid modifying the orginal metadata.
            metadata = {
                "depth": node.depth,
                **node.extra_metadata,
                **doc.metadata,
            }
            # Remove the metadata embedding key. TODO: Find a better way to do this.
            metadata.pop(METADATA_EMBEDDING_KEY, None)
            docs.append(
                Document(
                    id=node.id,
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )
        return docs


# this class uses pydantic, so store and edges
# must be provided at init time.
class GraphTraversalRetriever(BaseRetriever):
    store: Adapter
    edges: List[Union[str, Tuple[str, str]]]
    strategy: Strategy | None = None
    extra_args: dict[str, Any] = {}

    @computed_field  # type: ignore
    @property
    def edge_helper(self) -> EdgeHelper:
        return EdgeHelper(
            edges=self.edges,
            denormalized_path_delimiter=self.store.denormalized_path_delimiter,
            denormalized_static_value=self.store.denormalized_static_value,
            use_normalized_metadata=self.store.use_normalized_metadata,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        strategy: Strategy | dict[str, Any] | None = None,
        initial_roots: Sequence[str] = (),
        filter: dict[str, Any] | None = None,
        store_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve document nodes from this graph vector store using MMR-traversal.
        This strategy first retrieves the top `start_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.
        At each step, it considers the (remaining) documents from `start_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").
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
        state = _TraversalState(
            base_strategy=self.strategy,
            strategy=strategy,
            edge_helper=self.edge_helper,
        )

        # Retrieve initial candidates.
        initial_docs = self._fetch_initial_candidates(
            query, state=state, filter=filter, **store_kwargs
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
                adjacent_docs = self.store.get_adjacent(
                    outgoing_edges=next_outgoing_edges,
                    strategy=state.strategy,
                    filter=filter,
                    **store_kwargs,
                )

                state.add_docs(adjacent_docs)

        return state.finish()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        strategy: Strategy | dict[str, Any] | None = None,
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
            strategy: Specify or override the strategy to use for this retrieval.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `start_k = 0`.
            filter: Optional metadata to filter the results.
            store_kwargs: Optional kwargs passed to queries to the store.
            **kwargs: Additional keyword arguments passed to traversal state.
        """
        state = _TraversalState(
            edge_helper=self.edge_helper,
            base_strategy=self.strategy,
            strategy=strategy,
        )

        # Retrieve initial candidates and initialize state.
        initial_docs = await self._afetch_initial_candidates(
            query, state=state, filter=filter, **store_kwargs
        )
        state.add_docs(initial_docs, depth=0)

        if initial_roots:
            neighborhood_adjacent_docs = await self._afetch_neighborhood_candidates(
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
                adjacent_docs = await self.store.aget_adjacent(
                    outgoing_edges=next_outgoing_edges,
                    strategy=state.strategy,
                    filter=filter,
                    **store_kwargs,
                )

                state.add_docs(adjacent_docs)

        return state.finish()

    def _fetch_initial_candidates(
        self,
        query: str,
        *,
        state: _TraversalState,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Gets the embedded query and the set of initial candidates.

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
        query_embedding, docs = await self.store.asimilarity_search_with_embedding(
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
        neighborhood_docs = self.store.get(neighborhood)
        neighborhood_nodes = state.add_docs(neighborhood_docs)

        # Record the neighborhood nodes (specifically the outgoing edges from the
        # neighborhood) as visited.
        outgoing_edges = state.visit_nodes(neighborhood_nodes.values())

        # Fetch the candidates.
        return self.store.get_adjacent(
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
        neighborhood_docs = await self.store.aget(neighborhood)
        neighborhood_nodes = state.add_docs(neighborhood_docs)

        # Record the neighborhood nodes (specifically the outgoing edges from the
        # neighborhood) as visited.
        outgoing_edges = state.visit_nodes(neighborhood_nodes.values())

        # Fetch the candidates.
        return await self.store.aget_adjacent(
            outgoing_edges=outgoing_edges,
            strategy=state.strategy,
            filter=filter,
            **kwargs,
        )
