import dataclasses
import heapq
from collections.abc import Callable

from typing_extensions import override

from graph_retriever.strategies.base import NodeTracker, Strategy
from graph_retriever.types import Node


class _ScoredNode:
    def __init__(self, score: float, node: Node) -> None:
        self.score = score
        self.node = node

    def __lt__(self, other: "_ScoredNode") -> bool:
        return other.score < self.score


@dataclasses.dataclass
class Scored(Strategy):
    """Strategy selecting nodes using a scoring function."""

    scorer: Callable[[Node], float]
    _nodes: list[_ScoredNode] = dataclasses.field(default_factory=list)

    per_iteration_limit: int = 2

    @override
    def iteration(self, nodes: dict[str, Node], tracker: NodeTracker) -> None:
        for node in nodes.values():
            heapq.heappush(self._nodes, _ScoredNode(self.scorer(node), node))

        selected = {}
        for _x in range(self.per_iteration_limit):
            if not self._nodes:
                break

            node = heapq.heappop(self._nodes).node
            selected[node.id] = node
        tracker.select_and_traverse(selected)
