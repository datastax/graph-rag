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
    def add_nodes(self, nodes: dict[str, Node]) -> None:
        """Add nodes to the set of available nodes."""
        ...

    @abc.abstractmethod
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        """Return the nodes to select at the next iteration.

        Iteration ends when this returns an empty list.
        """
        ...

    def finalize_nodes(self, nodes: Iterable[Node]) -> Iterable[Node]:
        """Finalize the selected nodes."""
        return nodes
