"""Provide score-based traversal strategy."""

import heapq
from collections.abc import Callable, Iterable
from typing import override

from ..node import Node
from .base import Strategy


class Scored(Strategy):
    """Score-based traversal strategy.

    This strategy uses a scoring function (`scorer`) to rank nodes and selects
    the top-scored nodes in each iteration. Nodes are processed based on their
    scores, ensuring that higher-priority nodes are visited earlier.

    Attributes
    ----------
        k (int): Number of nodes to retrieve during traversal. Default is 5.
        start_k (int): Number of initial documents to fetch via similarity, added
            to any specified starting nodes. Default is 4.
        adjacent_k (int): Number of adjacent documents to fetch for each outgoing edge.
            Default is 10.
        max_depth (int | None): Maximum traversal depth. If None, there is no limit.
        query_embedding (list[float]): Embedding vector for the query.
        scorer (Callable[[Node], float]): A function to compute the score for each node.
            This is invoked once when the node is first discovered. Note that the depth
            of the node may be an upper-bound on the actual shortest path.
        select_k (int): The number of top-scored nodes to select in each iteration.
            Default is 10.
    """

    scorer: Callable[[Node], float]
    """Scoring function to apply to each node.

    This will be invoked once when the node is first discovered, meaning
    the depth may be an upper-bound on the actual shortest path for the node.
    """

    select_k: int = 10
    """Number of top-scored nodes to select in each iteration. Default 10."""

    _nodes: list[tuple[float, Node]] = []

    @override
    def discover_nodes(self, nodes: dict[str, Node]) -> None:
        for node in nodes.values():
            heapq.heappush(self._nodes, (self.scorer(node), node))

    @override
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        selected: list[Node] = []
        for _ in range(0, min(limit, self.select_k)):
            if len(self._nodes) == 0:
                break
            selected.append(heapq.heappop(self._nodes)[1])
        return selected
