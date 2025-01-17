from typing import Iterable

from ..node import Node
from .base import TraversalStrategy


class Eager(TraversalStrategy):
    """Node selection that selects all nodes at each step."""

    _nodes: list[Node] = []

    def add_nodes(self, nodes: dict[str, Node]) -> None:
        self._nodes.extend(nodes.values())

    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        nodes = self._nodes[:limit]
        self._nodes = []
        return nodes
