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
    """
    Scored traversal strategy.

    This strategy uses a scoring function to select nodes using a local maximum
    approach. In each iteration, it chooses the top scoring nodes available and
    then traverses the connected nodes.

    Parameters
    ----------
    scorer:
        A callable function that returns the score of a node.
    select_k :
        Maximum number of nodes to retrieve during traversal.
    start_k :
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k :
        Number of documents to fetch for each outgoing edge.
    max_depth :
        Maximum traversal depth. If `None`, there is no limit.
    per_iteration_limit:
        Maximum number of nodes to select and traverse during a single
        iteration.
    k:
        Deprecated: Use `select_k` instead.
        Maximum number of nodes to select and return during traversal.
    """

    scorer: Callable[[Node], float]
    _nodes: list[_ScoredNode] = dataclasses.field(default_factory=list)

    per_iteration_limit: int | None = None

    @override
    def iteration(self, nodes: Iterable[Node], tracker: NodeTracker) -> None:
        for node in nodes:
            heapq.heappush(self._nodes, _ScoredNode(self.scorer(node), node))

        limit = tracker.num_remaining
        if self.per_iteration_limit:
            limit = min(limit, self.per_iteration_limit)

        while limit > 0 and self._nodes:
            highest = heapq.heappop(self._nodes)
            node = highest.node
            node.extra_metadata["_score"] = highest.score
            limit -= tracker.select_and_traverse([node])
