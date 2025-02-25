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
    k :
        Maximum number of nodes to retrieve during traversal.
    start_k :
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k :
        Number of documents to fetch for each outgoing edge.
    max_depth :
        Maximum traversal depth. If `None`, there is no limit.
    """

    # iteration(discovered: dict[str, Node]) -> None : Called after each iteration.
    # finish() -> Nodes: Called to finalize the selected nodes.

    # traverse(nodes) -> None: Called to set the nodes to traverse in the next iteration.
    # select(nodes) -> None: Called to set the nodes to select at the end.

    @override
    def iteration(self, nodes: dict[str, Node]) -> None:
        self.traverse(nodes)
        self.select(nodes)