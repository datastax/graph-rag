import heapq
from typing import Any, Callable, Iterable

from ..node import Node
from .node_selector import NodeSelector


class EagerScoringNodeSelector(NodeSelector):
    """Node selection based on an eager scoring function."""

    def __init__(self, scorer: Callable[[Node], float], *, select_k: int = 1, **kwargs: dict[str, Any]) -> None:
        """Node selector choosing the top `select_k` nodes according to `scorer`
        in each iteration.

        Args:
            - scorer: The scoring function to apply. Will only be applied when the
              node is added, which means it is not re-executed if the `depth` changes.
            - select_k: The number of nodes to select at each iteration.
        """
        self._scorer = scorer
        self._nodes: list[tuple[float, Node]] = []
        self._select_k = select_k

    def add_nodes(self, nodes: dict[str, Node]) -> None:
        for node in nodes.values():
            try:
                heapq.heappush(self._nodes, (self._scorer(node), node))
            except TypeError as e:
                if "'<' not supported between instances" in str(e):
                    msg = "The scorer function must give a unique score for each node."
                    raise RuntimeError(msg) from None
                else:
                    raise

    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        selected: list[Node] = []
        for _ in range(0, min(limit, self._select_k)):
            if len(self._nodes) == 0:
                break
            selected.append(heapq.heappop(self._nodes)[1])
        return selected
