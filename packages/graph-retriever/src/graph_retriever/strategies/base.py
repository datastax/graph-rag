"""Define the base traversal strategy."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Iterable
from typing import Any

from graph_retriever.content import Content
from graph_retriever.types import Node


class NodeTracker:
    """Helper class for tracking traversal progress."""

    def __init__(self, select_k: int, max_depth: int | None) -> None:
        self._select_k: int = select_k
        self._max_depth: int | None = max_depth
        self._visited_node_ids: set[str] = set()
        self.to_traverse: set[Node] = set()
        self.selected: list[Node] = []

    @property
    def num_remaining(self):
        """The remaining number of nodes to be selected."""
        return max(self._select_k - len(self.selected), 0)

    def select(self, nodes: Iterable[Node]) -> None:
        """Select nodes to be included in the result set."""
        self.selected.extend(nodes)

    def traverse(self, nodes: Iterable[Node]) -> int:
        """Select nodes to be included in the next traversal."""
        for node in nodes:
            if node.id in self._visited_node_ids:
                print(f"{id} is visited, skipping traversal")
                continue
            if self._max_depth is not None and node.depth >= self._max_depth:
                print(
                    f"{id} depth {node.depth} is beyond {self._max_depth}, skipping traversal"
                )
                continue
            self.to_traverse.add(node)
            self._visited_node_ids.add(node.id)
        return len(self.to_traverse)

    def select_and_traverse(self, nodes: Iterable[Node]) -> int:
        """Select nodes to be included in the result set and the next traversal."""
        self.select(nodes)
        return self.traverse(nodes)

    def _not_visited(self, item: Content | Node):
        """Returns true if the content or node has not been visited."""
        return item.id not in self._visited_node_ids

    def _should_stop_traversal(self):
        """Returns true if traversal should be stopped"""
        return self.num_remaining == 0 or len(self.to_traverse) == 0


@dataclasses.dataclass(kw_only=True)
class Strategy(abc.ABC):
    """
    Interface for configuring node selection and traversal strategies.

    This base class defines how nodes are selected, traversed, and finalized during
    a graph traversal. Implementations can customize behaviors like limiting the depth
    of traversal, scoring nodes, or selecting the next set of nodes for exploration.

    Parameters
    ----------
    select_k :
        Maximum number of nodes to select and return during traversal.
    start_k :
        Number of nodes to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k :
        Number of nodes to fetch for each outgoing edge.
    max_traverse :
        Maximum number of nodes to traverse outgoing edges from before returning.
        If `None`, there is no limit.
    max_depth :
        Maximum traversal depth. If `None`, there is no limit.
    """

    select_k: int = 5
    start_k: int = 4
    adjacent_k: int = 10
    max_traverse: int | None = None
    max_depth: int | None = None

    _query_embedding: list[float] = dataclasses.field(default_factory=list)

    @abc.abstractmethod
    def iteration(self, *, nodes: Iterable[Node], tracker: NodeTracker) -> None:
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
        return list(selected)[: self.select_k]

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
