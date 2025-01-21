"""Provide score-based traversal strategy."""

import heapq
from typing import Callable, Iterable, override

from ..node import Node
from .base import Strategy


class Scored(Strategy):
    """Use `scorer` to select the top nodes in each iteration."""

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
