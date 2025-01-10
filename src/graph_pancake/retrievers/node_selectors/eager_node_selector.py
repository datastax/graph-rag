from typing import Any, Iterable

from ..node import Node
from .node_selector import NodeSelector


class EagerNodeSelector(NodeSelector):
    """Node selection that selects all nodes at each step."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        self._nodes: list[Node] = []

    def add_nodes(self, nodes: dict[str, Node]) -> None:
        self._nodes.extend(nodes.values())

    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        nodes = self._nodes[:limit]
        self._nodes = []
        return nodes
