"""Define the base traversal strategy."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Iterable
from typing import Any

from graph_retriever.types import Node


class NodeTracker:
    """Helper class for tracking traversal progress."""

    def __init__(self) -> None:
        self.visited_ids: set[str] = set()
        self.to_traverse: dict[str, Node] = {}
        self.selected: dict[str, Node] = {}

    def select(self, nodes: dict[str, Node]) -> None:
        """Select nodes to be included in the result set."""
        self.selected.update(nodes)

    def traverse(self, nodes: dict[str, Node]) -> None:
        """Select nodes to be included in the next traversal."""
        self.to_traverse.update(nodes)

    def select_and_traverse(self, nodes: dict[str, Node]) -> None:
        """Select nodes to be included in the result set and the next traversal."""
        self.select(nodes=nodes)
        self.traverse(nodes=nodes)


@dataclasses.dataclass(kw_only=True)
class Strategy(abc.ABC):
    """
    Interface for configuring node selection and traversal strategies.

    This base class defines how nodes are selected, traversed, and finalized during
    a graph traversal. Implementations can customize behaviors like limiting the depth
    of traversal, scoring nodes, or selecting the next set of nodes for exploration.

    Parameters
    ----------
    k :
        Maximum number of nodes to select and return during traversal.
    start_k :
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k :
        Number of documents to fetch for each outgoing edge.
    traverse_k :
        Maximum number of nodes to traverse outgoing edges from before returning.
    max_depth :
        Maximum traversal depth. If `None`, there is no limit.
    """

    k: int = 5
    start_k: int = 4
    adjacent_k: int = 10
    traverse_k: int = 4  # max_traverse?
    max_depth: int | None = None

    _query_embedding: list[float] = dataclasses.field(default_factory=list)

    @abc.abstractmethod
    def iteration(self, *, nodes: dict[str, Node], tracker: NodeTracker) -> None:
        """
        Process the newly discovered nodes on each iteration.

        This method should call `traverse` and/or `select` as appropriate
        to update the nodes that need to be traversed in this iteration or
        selected at the end of the retrieval, respectively.

        Parameters
        ----------
        nodes :
            Discovered nodes keyed by their IDs.
        """
        ...

    def finalize_nodes(self, selected: Iterable[Node]) -> Iterable[Node]:
        """
        Finalize the selected nodes.

        This method is called before returning the final set of nodes.

        Returns
        -------
        :
            Finalized nodes.
        """
        # Take the first `self.k` selected items.
        # Strategies may override finalize to perform reranking if needed.
        return list(selected)[: self.k]

    @staticmethod
    def build(
        base_strategy: Strategy,
        **kwargs: Any,
    ) -> Strategy:
        """
        Build a strategy for a retrieval operation.

        Combines a base strategy with any provided keyword arguments to
        create a customized traversal strategy.

        Parameters
        ----------
        base_strategy :
            The base strategy to start with.
        kwargs :
            Additional configuration options for the strategy.

        Returns
        -------
        :
            A configured strategy instance.

        Raises
        ------
        ValueError
            If 'strategy' is set incorrectly or extra arguments are invalid.
        """
        # Check if there is a new strategy to use. Otherwise, use the base.
        strategy: Strategy
        if "strategy" in kwargs:
            if next(iter(kwargs.keys())) != "strategy":
                raise ValueError("Error: 'strategy' must be set before other args.")
            strategy = kwargs.pop("strategy")
            if not isinstance(strategy, Strategy):
                raise ValueError(
                    f"Unsupported 'strategy' type {type(strategy).__name__}."
                    " Must be a sub-class of Strategy"
                )
        elif base_strategy is not None:
            strategy = base_strategy
        else:
            raise ValueError("'strategy' must be set in `__init__` or invocation")

        # Apply the kwargs to update the strategy.
        assert strategy is not None
        strategy = dataclasses.replace(strategy, **kwargs)

        return strategy
