"""Define the base traversal strategy."""

import abc
from typing import Iterable

from pydantic import BaseModel

from ..node import Node


class Strategy(BaseModel, abc.ABC):
    """Interface for configuring node selection during the traversal."""

    k: int = 5
    """Number of nodes to retrieve during the traversal. Default 5."""

    start_k: int = 4
    """Number of initial documents to fetch via similarity.

    Will be added to the specified starting nodes, if any.
    """

    adjacent_k: int = 10
    """Number of adjacent Documents to fetch for each outgoing edge. Default 10.
    """

    max_depth: int | None = None
    """Maximum depth to retrieve. Default no limit."""

    query_embedding: list[float] = []
    """Query embedding."""

    @abc.abstractmethod
    def discover_nodes(self, nodes: dict[str, Node]) -> None:
        """Add discovered nodes to the strategy.

        Args:
            nodes: The nodes being discovered. Keyed by node ID.

        """
        ...

    @abc.abstractmethod
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        """Select discovered nodes to visit in the next iteration.

        Traversal ends if this returns an empty list, even if `k` nodes haven't
        been selected in total yet.

        Any nodes reachable via new edges will be discovered before the next
        call to `select_nodes`.

        Args:
            limit: The maximum number of nodes to select.

        Returns
        -------
        The nodes selected for the next iteration.
        Traversal ends if this returns empty list.

        """
        ...

    def finalize_nodes(self, nodes: Iterable[Node]) -> Iterable[Node]:
        """Finalize the selected nodes."""
        return nodes
