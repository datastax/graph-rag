"""Provide eager (breadth-first) traversal strategy."""

from collections.abc import Iterable
from typing import override

from ..node import Node
from .base import Strategy


class Eager(Strategy):
    """
    Eager traversal strategy (breadth-first).

    This strategy selects all discovered nodes at each traversal step. It ensures
    breadth-first traversal by processing nodes layer by layer, which is useful for
    scenarios where all nodes at the current depth should be explored before proceeding
    to the next depth.

    Attributes
    ----------
        k (int): Number of nodes to retrieve during traversal. Default is 5.
        start_k (int): Number of initial documents to fetch via similarity, added
            to any specified starting nodes. Default is 4.
        adjacent_k (int): Number of adjacent documents to fetch for each outgoing edge.
            Default is 10.
        max_depth (int | None): Maximum traversal depth. If None, there is no limit.
        query_embedding (list[float]): Embedding vector for the query.
    """

    _nodes: list[Node] = []

    @override
    def discover_nodes(self, nodes: dict[str, Node]) -> None:
        self._nodes.extend(nodes.values())

    @override
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        nodes = self._nodes[:limit]
        self._nodes = []
        return nodes
