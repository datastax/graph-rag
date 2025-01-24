"""Provide score-based traversal strategy."""

import heapq
from collections.abc import Callable, Iterable
from typing import override

from langchain_graph_retriever.strategies.base import Strategy
from langchain_graph_retriever.types import Node


class Scored(Strategy):
    """
    Score-based traversal strategy.

    This strategy uses a scoring function (`scorer`) to rank nodes and selects
    the top-scored nodes in each iteration. Nodes are processed based on their
    scores, ensuring that higher-priority nodes are visited earlier.

    Parameters
    ----------
    k : int, default 5
        Maximum number of nodes to retrieve during traversal.
    start_k : int, default 4
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k : int, default 10
        Number of documents to fetch for each outgoing edge.
    max_depth : int, default None
        Maximum traversal depth. If `None`, there is no limit.
    scorer : Callable[[Node], float]
        A function to compute the score for each node. This is invoked once when
        the node is first discovered. Note that the depth of the node may be an
        upper-bound on the actual shortest path.
    select_k : int, default 10
        The number of top-scored nodes to select in each iteration.

    Attributes
    ----------
    k : int
        Maximum number of nodes to retrieve during traversal.
    start_k : int
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k : int
        Number of documents to fetch for each outgoing edge.
    max_depth : int
        Maximum traversal depth. If `None`, there is no limit.
    scorer : Callable[[Node], float]
        A function to compute the score for each node. This is invoked once when
        the node is first discovered. Note that the depth of the node may be an
        upper-bound on the actual shortest path.
    select_k : int
        The number of top-scored nodes to select in each iteration.
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
