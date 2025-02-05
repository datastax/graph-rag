"""Provide eager (breadth-first) traversal strategy."""

import dataclasses
from collections.abc import Iterable

from typing_extensions import override

from graph_retriever.strategies.base import Strategy
from graph_retriever.types import Node


@dataclasses.dataclass
class Eager(Strategy):
    """
    Eager traversal strategy (breadth-first).

    This strategy selects all discovered nodes at each traversal step. It ensures
    breadth-first traversal by processing nodes layer by layer, which is useful for
    scenarios where all nodes at the current depth should be explored before proceeding
    to the next depth.

    Parameters
    ----------
    k : int, default 5
        Maximum number of nodes to retrieve during traversal.
    start_k : int, default 4
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k : int, default 10
        Number of documents to fetch for each outgoing edge.
    max_depth : int, optional
        Maximum traversal depth. If `None`, there is no limit.
    """

    _nodes: list[Node] = dataclasses.field(default_factory=list)

    @override
    def discover_nodes(self, nodes: dict[str, Node]) -> None:
        self._nodes.extend(nodes.values())

    @override
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        nodes = self._nodes[:limit]
        self._nodes = []
        return nodes
