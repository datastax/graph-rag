"""Provide eager (breadth-first) traversal strategy."""

from typing import Iterable, override

from ..node import Node
from .base import Strategy


class Eager(Strategy):
    """Node selection that selects all nodes at each step."""

    _nodes: list[Node] = []

    @override
    def discover_nodes(self, nodes: dict[str, Node]) -> None:
        self._nodes.extend(nodes.values())

    @override
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        nodes = self._nodes[:limit]
        self._nodes = []
        return nodes
