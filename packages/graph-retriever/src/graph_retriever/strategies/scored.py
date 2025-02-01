import heapq
from collections.abc import Callable, Iterable

from typing_extensions import override

from graph_retriever.strategies.base import Strategy
from graph_retriever.types import Node


class _ScoredNode:
    def __init__(self, score: float, node: Node) -> None:
        self.score = score
        self.node = node

    def __lt__(self, other) -> bool:
        return other.score < self.score


class Scored(Strategy):
    """Strategy selecing nodes using a scoring function."""

    scorer: Callable[[Node], float]
    per_iteration_limit: int | None = None

    _nodes: list[_ScoredNode] = []

    @override
    def discover_nodes(self, nodes: dict[str, Node]) -> None:
        for node in nodes.values():
            heapq.heappush(self._nodes, _ScoredNode(self.scorer(node), node))

    @override
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        if self.per_iteration_limit and self.per_iteration_limit < limit:
            limit = self.per_iteration_limit

        selected = []
        for _x in range(limit):
            if not self._nodes:
                break

            selected.append(heapq.heappop(self._nodes).node)
        return selected
