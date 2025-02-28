import dataclasses
import heapq
from collections.abc import Callable, Iterable

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

    per_iteration_limit: int | None = None

    @override
    def iteration(self, nodes: Iterable[Node], tracker: NodeTracker) -> None:
        for node in nodes:
            print(f"adding node: {node.id} to heap")
            heapq.heappush(self._nodes, _ScoredNode(self.scorer(node), node))

        limit = tracker.num_remaining
        if self.per_iteration_limit and self.per_iteration_limit < limit:
            limit = self.per_iteration_limit

        for _x in range(limit):
            if not self._nodes:
                print("no nodes remaining, ending iteration")
                break

            node = heapq.heappop(self._nodes).node
            print(f"popped node: {node.id} off heap, adding to select and traverse")
            tracker.select_and_traverse([node])
