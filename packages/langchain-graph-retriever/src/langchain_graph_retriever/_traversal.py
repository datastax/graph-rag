from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

from .adapters.base import METADATA_EMBEDDING_KEY, Adapter
from .edge_helper import Edge, EdgeHelper
from .node import Node
from .strategies import Strategy


class Traversal:
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
        neighborhood_docs = await self.store.aget(self.initial_root_ids)
        neighborhood_nodes = self.add_docs(neighborhood_docs)

        # Record the neighborhood nodes (specifically the outgoing edges from the
        # neighborhood) as visited.
        outgoing_edges = self.visit_nodes(neighborhood_nodes.values())

        # Fetch the candidates.
        return await self._afetch_adjacent(outgoing_edges)

    def _fetch_adjacent(self, outgoing_edges: set[Edge]) -> Iterable[Document]:
        return self.store.get_adjacent(
            outgoing_edges=outgoing_edges,
            strategy=self.strategy,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )

    async def _afetch_adjacent(self, outgoing_edges: set[Edge]) -> Iterable[Document]:
        return await self.store.aget_adjacent(
            outgoing_edges=outgoing_edges,
            strategy=self.strategy,
            filter=self.metadata_filter,
            **self.store_kwargs,
        )

    def _doc_to_new_node(
        self, doc: Document, *, depth: int | None = None
    ) -> Node | None:
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
        """Record the nodes as visited, returning the new outgoing edges.

        After this call, the outgoing edges will be added to the visited
        set, and not revisited during the traversal.
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
        """Select the next round of nodes.

        Returns
        -------
        The set of new edges that need to be explored.

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
