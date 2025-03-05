"""Provide eager (breadth-first) traversal strategy."""

import dataclasses
from collections.abc import Iterable

from typing_extensions import override

from graph_retriever.strategies.base import NodeTracker, Strategy
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
    select_k :
        Maximum number of nodes to retrieve during traversal.
    start_k :
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k :
        Number of documents to fetch for each outgoing edge.
    max_depth :
        Maximum traversal depth. If `None`, there is no limit.
    k:
        Deprecated: Use `select_k` instead.
        Maximum number of nodes to select and return during traversal.
    """

    @override
    def iteration(self, nodes: Iterable[Node], tracker: NodeTracker) -> None:
        tracker.select_and_traverse(nodes)
