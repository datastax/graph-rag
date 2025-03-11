"""Implements the traversal logic for graph-based document retrieval."""

import copy
from collections.abc import Iterable, Sequence
from typing import Any

from graph_retriever.adapters import Adapter
from graph_retriever.content import Content
from graph_retriever.edges import Edge, EdgeFunction, EdgeSpec, MetadataEdgeFunction
from graph_retriever.strategies import NodeTracker, Strategy
from graph_retriever.types import Node
from graph_retriever.utils.math import cosine_similarity


def traverse(
    query: str,
    *,
    edges: list[EdgeSpec] | EdgeFunction,
    strategy: Strategy,
    store: Adapter,
    metadata_filter: dict[str, Any] | None = None,
    initial_root_ids: Sequence[str] = (),
    store_kwargs: dict[str, Any] = {},
) -> list[Node]:
    """
    Perform a graph traversal to retrieve nodes for a specific query.

    Parameters
    ----------
    query :
        The query string for the traversal.
    edges :
        A list of [EdgeSpec][graph_retriever.edges.EdgeSpec] for use in creating a
        [MetadataEdgeFunction][graph_retriever.edges.MetadataEdgeFunction],
        or an [EdgeFunction][graph_retriever.edges.EdgeFunction].
    strategy :
        The traversal strategy that defines how nodes are discovered, selected,
        and finalized.
    store :
        The vector store adapter used for similarity searches and document
        retrieval.
    metadata_filter :
        Optional filter for metadata during traversal.
    initial_root_ids :
        IDs of the initial root nodes for the traversal.
    store_kwargs :
        Additional arguments passed to the store adapter.

    Returns
    -------
    :
        Nodes returned by the traversal.
    """
    traversal = _Traversal(
        query=query,
        edges=edges,
        strategy=copy.deepcopy(strategy),
        store=store,
        metadata_filter=metadata_filter,
        initial_root_ids=initial_root_ids,
        store_kwargs=store_kwargs,
    )
    return traversal.traverse()


async def atraverse(
    query: str,
    *,
    edges: list[EdgeSpec] | EdgeFunction,
    strategy: Strategy,
    store: Adapter,
    metadata_filter: dict[str, Any] | None = None,
    initial_root_ids: Sequence[str] = (),
    store_kwargs: dict[str, Any] = {},
) -> list[Node]:
    """
    Asynchronously perform a graph traversal to retrieve nodes for a specific query.

    Parameters
    ----------
    query :
        The query string for the traversal.
    edges :
        A list of [EdgeSpec][graph_retriever.edges.EdgeSpec] for use in creating a
        [MetadataEdgeFunction][graph_retriever.edges.MetadataEdgeFunction],
        or an [EdgeFunction][graph_retriever.edges.EdgeFunction].
    strategy :
        The traversal strategy that defines how nodes are discovered, selected,
        and finalized.
    store :
        The vector store adapter used for similarity searches and document
        retrieval.
    metadata_filter :
        Optional filter for metadata during traversal.
    initial_root_ids :
        IDs of the initial root nodes for the traversal.
    store_kwargs :
        Additional arguments passed to the store adapter.

    Returns
    -------
    :
        Nodes returned by the traversal.
    """
    traversal = _Traversal(
        query=query,
        edges=edges,
        strategy=copy.deepcopy(strategy),
        store=store,
        metadata_filter=metadata_filter,
        initial_root_ids=initial_root_ids,
        store_kwargs=store_kwargs,
    )
    return await traversal.atraverse()


class _Traversal:
    """
    Handles a single traversal operation for a graph-based retrieval system.

    The `_Traversal` class manages the process of discovering, visiting, and selecting
    nodes within a graph, based on a query and a traversal strategy. It supports
    synchronous and asynchronous traversal, enabling retrieval of documents in a
    controlled, iterative manner.

    This class should not be reused between traversals.
    """

    def __init__(
        self,
        query: str,
        *,
        edges: list[EdgeSpec] | EdgeFunction,
        strategy: Strategy,
        store: Adapter,
        metadata_filter: dict[str, Any] | None = None,
        initial_root_ids: Sequence[str] = (),
        store_kwargs: dict[str, Any] = {},
    ) -> None:
        self.query = query

        self.edge_function: EdgeFunction
        if isinstance(edges, list):
            self.edge_function = MetadataEdgeFunction(edges)
        elif callable(edges):
            self.edge_function = edges
        else:
            raise ValueError(f"Invalid edges: {edges}")

        self.strategy = strategy
        self.store = store
        self.metadata_filter = metadata_filter
        self.initial_root_ids = initial_root_ids
        self.store_kwargs = store_kwargs

        self._used = False
        self._visited_edges: set[Edge] = set()
        self._edge_depths: dict[Edge, int] = {}
        self._node_tracker: NodeTracker = NodeTracker(
            select_k=strategy.select_k, max_depth=strategy.max_depth
        )

    def _check_first_use(self):
        assert not self._used, "Traversals cannot be re-used."
        self._used = True

    def traverse(self) -> list[Node]:
        """
        Execute the traversal synchronously.

        This method retrieves initial candidates, discovers and visits nodes,
        and explores edges iteratively until the traversal is complete.

        Returns
        -------
        :
            The final set of nodes resulting from the traversal.
        """
        self._check_first_use()

        # Retrieve initial candidates.
        initial_content = self._fetch_initial_candidates()
        if self.initial_root_ids:
            initial_content.extend(self.store.get(self.initial_root_ids))
        nodes = self._contents_to_nodes(initial_content, depth=0)

        while True:
            self.strategy.iteration(nodes=nodes, tracker=self._node_tracker)

            if self._node_tracker._should_stop_traversal():
                break

            next_outgoing_edges = self.select_next_edges(self._node_tracker.to_traverse)
            new_content = self._fetch_adjacent(next_outgoing_edges)
            nodes = self._contents_to_nodes(new_content)

            self._node_tracker.to_traverse.clear()

        return list(self.strategy.finalize_nodes(self._node_tracker.selected))

    async def atraverse(self) -> list[Node]:
        """
        Execute the traversal asynchronously.

        This method retrieves initial candidates, discovers and visits nodes,
        and explores edges iteratively until the traversal is complete.

        Returns
        -------
        :
            The final set of nodes resulting from the traversal.
        """
        self._check_first_use()

        # Retrieve initial candidates.
        initial_content = await self._afetch_initial_candidates()
        if self.initial_root_ids:
            initial_content.extend(await self.store.aget(self.initial_root_ids))
        nodes = self._contents_to_nodes(initial_content, depth=0)

        while True:
            self.strategy.iteration(nodes=nodes, tracker=self._node_tracker)

            if self._node_tracker._should_stop_traversal():
                break

            next_outgoing_edges = self.select_next_edges(self._node_tracker.to_traverse)
            new_content = await self._afetch_adjacent(next_outgoing_edges)
            nodes = self._contents_to_nodes(new_content)

            self._node_tracker.to_traverse.clear()

        return list(self.strategy.finalize_nodes(self._node_tracker.selected))

    def _fetch_initial_candidates(self) -> list[Content]:
        """
        Retrieve initial candidates based on the query and strategy.

        Returns
        -------
        :
            The initial content retrieved via similarity search.
        """
        query_embedding, docs = self.store.search_with_embedding(
            query=self.query,
            k=self.strategy.start_k,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )
        self.strategy._query_embedding = query_embedding
        return docs

    async def _afetch_initial_candidates(self) -> list[Content]:
        query_embedding, docs = await self.store.asearch_with_embedding(
            query=self.query,
            k=self.strategy.start_k,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )
        self.strategy._query_embedding = query_embedding
        return docs

    def _fetch_adjacent(self, edges: set[Edge]) -> Iterable[Content]:
        """
        Retrieve documents adjacent to the specified outgoing edges.

        This method uses the vector store adapter to fetch documents connected to
        the provided edges.

        Parameters
        ----------
        edges :
            The edges whose adjacent documents need to be fetched.

        Returns
        -------
        :
            The set of content adjacent to the specified edges.
        """
        return self.store.adjacent(
            edges=edges,
            query_embedding=self.strategy._query_embedding,
            k=self.strategy.adjacent_k,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )

    async def _afetch_adjacent(self, edges: set[Edge]) -> Iterable[Content]:
        """
        Asynchronously retrieve documents adjacent to the specified outgoing edges.

        This method uses the vector store adapter to fetch documents connected to
        the provided edges.

        Parameters
        ----------
        edges :
            The edges whose adjacent documents need to be fetched.

        Returns
        -------
        :
            The set of content adjacent to the specified edges.
        """
        return await self.store.aadjacent(
            edges=edges,
            query_embedding=self.strategy._query_embedding,
            k=self.strategy.adjacent_k,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )

    def _contents_to_nodes(
        self, contents: Iterable[Content], *, depth: int | None = None
    ) -> Iterable[Node]:
        """
        Convert a content object into a node for traversal.

        This method creates a new `Node` instance, associates it with the document's
        metadata, and calculates its depth based on the incoming edges.

        Parameters
        ----------
        content :
            The content to convert into a node.
        depth :
            The depth of the node. If None, the depth is calculated based on the
            incoming edges.

        Returns
        -------
        :
            The newly created nodes.
        """
        # Determine which contents to include.
        content_dict = {c.id: c for c in contents if self._node_tracker._not_visited(c)}

        # Compute scores (as needed).
        if any(c.score is None for c in content_dict.values()):
            scores = cosine_similarity(
                [self.strategy._query_embedding],
                [c.embedding for c in content_dict.values() if c.score is None],
            )[0]
        else:
            scores = []

        # Create the nodes
        scores_it = iter(scores)
        nodes = []
        for content in content_dict.values():
            # Determine incoming/outgoing edges.
            edges = self.edge_function(content)

            # Compute the depth
            if depth is None:
                depth = min(
                    [
                        d
                        for e in edges.incoming
                        if (d := self._edge_depths.get(e, None)) is not None
                    ],
                    default=0,
                )

            score = content.score or next(scores_it)
            nodes.append(
                Node(
                    id=content.id,
                    content=content.content,
                    depth=depth,
                    embedding=content.embedding,
                    similarity_score=score,
                    metadata=content.metadata,
                    incoming_edges=edges.incoming,
                    outgoing_edges=edges.outgoing,
                )
            )
        return nodes

    def select_next_edges(self, nodes: dict[str, Node]) -> set[Edge]:
        """
        Find the unvisited outgoing edges from the set of new nodes to traverse.

        This method updates the traversal state by recording the outgoing edges of the
        provided nodes. Outgoing edges that have not been visited before are identified
        and added to the set of edges to explore in subsequent traversal steps.

        Parameters
        ----------
        nodes :
            The new nodes to traverse

        Returns
        -------
        :
            The set of new outgoing edges that need to be explored.

        Notes
        -----
        - The `new_outgoing_edges` dictionary tracks the depth of each outgoing
        edge.
        - If a node's outgoing edge leads to a lower depth, the edge's depth is
        updated to reflect the shortest path.
        - The `_visited_edges` set is updated to include all outgoing edges
        from the provided nodes.
        """
        new_outgoing_edges: dict[Edge, int] = {}
        for node in nodes.values():
            node_new_outgoing_edges = node.outgoing_edges - self._visited_edges
            for edge in node_new_outgoing_edges:
                depth = new_outgoing_edges.setdefault(edge, node.depth + 1)
                if node.depth + 1 < depth:
                    new_outgoing_edges[edge] = node.depth + 1

        self._edge_depths.update(new_outgoing_edges)

        new_outgoing_edge_set = set(new_outgoing_edges.keys())
        self._visited_edges.update(new_outgoing_edge_set)
        return new_outgoing_edge_set
