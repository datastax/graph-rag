"""Implements the traversal logic for graph-based document retrieval."""

from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.documents import Document

from .adapters.base import METADATA_EMBEDDING_KEY, Adapter
from .edge_helper import Edge, EdgeHelper
from .node import Node
from .strategies import Strategy


class Traversal:
    """Handles a single traversal operation for a graph-based retrieval system.

    The `Traversal` class manages the process of discovering, visiting, and selecting
    nodes within a graph, based on a query and a traversal strategy. It supports
    synchronous and asynchronous traversal, enabling retrieval of documents in a
    controlled, iterative manner.

    This class should not be reused between traversals.

    Attributes
    ----------
        query (str): The query string for the traversal.
        edges (EdgeHelper): A helper object for managing graph edges.
        strategy (Strategy): The traversal strategy that defines how nodes are
            discovered, selected, and finalized.
        store (Adapter): The vector store adapter used for similarity searches and
            document retrieval.
        metadata_filter (dict[str, Any] | None): Optional filter for metadata during
            traversal.
        initial_root_ids (Sequence[str]): IDs of the initial root nodes for the
            traversal.
        store_kwargs (dict[str, Any]): Additional arguments passed to the store adapter.
    """

    def __init__(
        self,
        query: str,
        *,
        edges: EdgeHelper,
        strategy: Strategy,
        store: Adapter,
        metadata_filter: dict[str, Any] | None = None,
        initial_root_ids: Sequence[str] = (),
        store_kwargs: dict[str, Any] = {},
    ) -> None:
        self.query = query
        self.edges = edges
        self.strategy = strategy
        self.store = store
        self.metadata_filter = metadata_filter
        self.initial_root_ids = initial_root_ids
        self.store_kwargs = store_kwargs

        self._used = False
        self._visited_edges: set[Edge] = set()
        self._edge_depths: dict[Edge, int] = {}
        self._doc_cache: dict[str, Document] = {}
        self._node_cache: dict[str, Node] = {}
        self._selected_nodes: dict[str, Node] = {}

    def _check_first_use(self):
        assert not self._used, "Traversals cannot be re-used."
        self._used = True

    def traverse(self) -> list[Document]:
        """Execute the traversal synchronously.

        This method retrieves initial candidates, discovers and visits nodes,
        and explores edges iteratively until the traversal is complete.

        Returns
        -------
            list[Document]: The final set of documents resulting from the traversal.
        """
        self._check_first_use()

        # Retrieve initial candidates.
        initial_docs = self._fetch_initial_candidates()
        self.add_docs(initial_docs, depth=0)

        if self.initial_root_ids:
            neighborhood_adjacent_docs = self._fetch_neighborhood_candidates()
            self.add_docs(neighborhood_adjacent_docs, depth=0)

        while True:
            # Select the next batch of nodes, and (new) outgoing edges.
            next_outgoing_edges = self.select_next_edges()
            if next_outgoing_edges is None:
                break
            elif next_outgoing_edges:
                # Find the (new) document with incoming edges from those edges.
                adjacent_docs = self._fetch_adjacent(next_outgoing_edges)
                self.add_docs(adjacent_docs)

        return self.finish()

    async def atraverse(self) -> list[Document]:
        """Execute the traversal asynchronously.

        This method retrieves initial candidates, discovers and visits nodes,
        and explores edges iteratively until the traversal is complete.

        Returns
        -------
            list[Document]: The final set of documents resulting from the traversal.
        """
        self._check_first_use()

        # Retrieve initial candidates.
        initial_docs = await self._afetch_initial_candidates()
        self.add_docs(initial_docs, depth=0)

        if self.initial_root_ids:
            neighborhood_adjacent_docs = await self._afetch_neighborhood_candidates()
            self.add_docs(neighborhood_adjacent_docs, depth=0)

        while True:
            # Select the next batch of nodes, and (new) outgoing edges.
            next_outgoing_edges = self.select_next_edges()
            if next_outgoing_edges is None:
                break
            elif next_outgoing_edges:
                # Find the (new) document with incoming edges from those edges.
                adjacent_docs = await self._afetch_adjacent(next_outgoing_edges)
                self.add_docs(adjacent_docs)

        return self.finish()

    def _fetch_initial_candidates(self) -> list[Document]:
        """Retrieve initial candidates based on the query and strategy.

        Returns
        -------
            list[Document]: The initial set of documents retrieved via similarity
            search.
        """
        query_embedding, docs = self.store.similarity_search_with_embedding(
            query=self.query,
            k=self.strategy.start_k,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )
        self.strategy.query_embedding = query_embedding
        return docs

    async def _afetch_initial_candidates(self) -> list[Document]:
        query_embedding, docs = await self.store.asimilarity_search_with_embedding(
            query=self.query,
            k=self.strategy.start_k,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )
        self.strategy.query_embedding = query_embedding
        return docs

    def _fetch_neighborhood_candidates(self) -> Iterable[Document]:
        """Retrieve neighborhood candidates for traversal.

        This method fetches initial root documents, converts them to nodes, and records
        their outgoing edges as visited. It then fetches additional candidates adjacent
        to these nodes.

        Returns
        -------
            Iterable[Document]: The set of documents adjacent to the initial root nodes.
        """
        neighborhood_docs = self.store.get(self.initial_root_ids)
        neighborhood_nodes = self.add_docs(neighborhood_docs)

        # Record the neighborhood nodes (specifically the outgoing edges from the
        # neighborhood) as visited.
        outgoing_edges = self.visit_nodes(neighborhood_nodes.values())

        # Fetch the candidates.
        return self._fetch_adjacent(outgoing_edges)

    async def _afetch_neighborhood_candidates(
        self,
    ) -> Iterable[Document]:
        """Asynchronously retrieve neighborhood candidates for traversal.

        This method fetches initial root documents, converts them to nodes, and records
        their outgoing edges as visited. It then fetches additional candidates adjacent
        to these nodes.

        Returns
        -------
            Iterable[Document]: The set of documents adjacent to the initial root nodes.
        """
        neighborhood_docs = await self.store.aget(self.initial_root_ids)
        neighborhood_nodes = self.add_docs(neighborhood_docs)

        # Record the neighborhood nodes (specifically the outgoing edges from the
        # neighborhood) as visited.
        outgoing_edges = self.visit_nodes(neighborhood_nodes.values())

        # Fetch the candidates.
        return await self._afetch_adjacent(outgoing_edges)

    def _fetch_adjacent(self, outgoing_edges: set[Edge]) -> Iterable[Document]:
        """Retrieve documents adjacent to the specified outgoing edges.

        This method uses the vector store adapter to fetch documents connected to
        the provided edges.

        Args:
            outgoing_edges (set[Edge]): The edges whose adjacent documents need to
                be fetched.

        Returns
        -------
            Iterable[Document]: The set of documents adjacent to the specified edges.
        """
        return self.store.get_adjacent(
            outgoing_edges=outgoing_edges,
            strategy=self.strategy,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )

    async def _afetch_adjacent(self, outgoing_edges: set[Edge]) -> Iterable[Document]:
        """Asynchronously retrieve documents adjacent to the specified outgoing edges.

        This method uses the vector store adapter to fetch documents connected to
        the provided edges.

        Args:
            outgoing_edges (set[Edge]): The edges whose adjacent documents need to
                be fetched.

        Returns
        -------
            Iterable[Document]: The set of documents adjacent to the specified edges.
        """
        return await self.store.aget_adjacent(
            outgoing_edges=outgoing_edges,
            strategy=self.strategy,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )

    def _doc_to_new_node(
        self, doc: Document, *, depth: int | None = None
    ) -> Node | None:
        """Convert a document into a new node for the traversal.

        This method checks whether the document has already been processed. If not,
        it creates a new `Node` instance, associates it with the document's metadata,
        and calculates its depth based on the incoming edges.

        Args:
            doc (Document): The document to convert into a node.
            depth (int | None, optional): The depth of the node. If None, the depth
                is calculated based on the incoming edges.

        Returns
        -------
            Node | None: The newly created node, or None if the document has already
            been processed.

        Raises
        ------
            ValueError: If the document does not have an ID.
        """
        if doc.id is None:
            raise ValueError("All documents should have ids")
        if doc.id in self._node_cache:
            return None

        doc = self._doc_cache.setdefault(doc.id, doc)
        assert doc.id is not None
        incoming_edges, outgoing_edges = self.edges.get_incoming_outgoing(doc.metadata)
        if depth is None:
            depth = min(
                [
                    d
                    for e in incoming_edges
                    if (d := self._edge_depths.get(e, None)) is not None
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
        self._node_cache[doc.id] = node

        return node

    def add_docs(
        self, docs: Iterable[Document], *, depth: int | None = None
    ) -> dict[str, Node]:
        """Add a batch of documents to the traversal and convert them into nodes.

        This method records the depth of new nodes, filters them based on the
        strategy's maximum depth, and updates the strategy with the discovered nodes.

        Args:
            docs (Iterable[Document]): The documents to add.
            depth (int | None): The depth to assign to the nodes. If None, the depth
                is inferred based on the incoming edges.

        Returns
        -------
            dict[str, Node]: A dictionary of node IDs to the newly created nodes.
        """
        # Record the depth of new nodes.
        nodes = {
            node.id: node
            for doc in docs
            if (node := self._doc_to_new_node(doc, depth=depth)) is not None
            if (
                self.strategy.max_depth is None or node.depth <= self.strategy.max_depth
            )
        }
        self.strategy.discover_nodes(nodes)
        return nodes

    def visit_nodes(self, nodes: Iterable[Node]) -> set[Edge]:
        """Mark nodes as visited and return their new outgoing edges.

        This method updates the traversal state by marking the provided nodes as visited
        and recording their outgoing edges. Outgoing edges that have not been visited
        before are identified and added to the set of edges to explore in subsequent
        traversal steps.

        Args:
            nodes (Iterable[Node]): The nodes to mark as visited.

        Returns
        -------
            set[Edge]: The set of new outgoing edges that need to be explored.

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
        for node in nodes:
            node_new_outgoing_edges = node.outgoing_edges - self._visited_edges
            for edge in node_new_outgoing_edges:
                depth = new_outgoing_edges.setdefault(edge, node.depth + 1)
                if node.depth + 1 < depth:
                    new_outgoing_edges[edge] = node.depth + 1

        self._edge_depths.update(new_outgoing_edges)

        new_outgoing_edge_set = set(new_outgoing_edges.keys())
        self._visited_edges.update(new_outgoing_edge_set)
        return new_outgoing_edge_set

    def select_next_edges(self) -> set[Edge] | None:
        """Select the next set of edges to explore.

        This method uses the traversal strategy to select the next batch of nodes
        and identifies new outgoing edges for exploration.

        Returns
        -------
            set[Edge] | None: The set of new edges to explore, or None if the traversal
            is complete.
        """
        remaining = self.strategy.k - len(self._selected_nodes)

        if remaining <= 0:
            return None

        next_nodes = self.strategy.select_nodes(limit=remaining)
        if not next_nodes:
            return None

        next_nodes = [n for n in next_nodes if n.id not in self._selected_nodes]
        if len(next_nodes) == 0:
            return None

        self._selected_nodes.update({n.id: n for n in next_nodes})
        new_outgoing_edges = self.visit_nodes(next_nodes)
        return new_outgoing_edges

    def finish(self) -> list[Document]:
        """Finalize the traversal and return the final set of documents.

        This method finalizes the selected nodes using the traversal strategy,
        processes their metadata, and assembles the final list of documents.

        Returns
        -------
            list[Document]: The final set of documents resulting from the traversal.
        """
        final_nodes = self.strategy.finalize_nodes(self._selected_nodes.values())
        docs = []
        for node in final_nodes:
            doc = self._doc_cache.get(node.id, None)
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
