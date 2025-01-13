import heapq
from typing import Callable, Iterable

from ..node import Node
from .base import TraversalStrategy


class Scored(TraversalStrategy):
    """Node selector choosing the top `select_k` nodes according to `scorer`
    in each iteration.
    """

    scorer: Callable[[Node], float]
    """Scoring function to apply to each node.

    This will be invoked once when the node is first discovered, meaning
    the depth may be an upper-bound on the actual shortest path for the node.
    """

    select_k: int = 10
    """Number of top-scored nodes to select in each iteration. Default 10."""

    _nodes: list[tuple[float, Node]] = []

    def add_nodes(self, nodes: dict[str, Node]) -> None:
        for node in nodes.values():
            heapq.heappush(self._nodes, (self._scorer(node), node))

    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        selected: list[Node] = []
        for _ in range(0, min(limit, self._select_k)):
            if len(self._nodes) == 0:
                break
            selected.append(heapq.heappop(self._nodes)[1])
        return selected
