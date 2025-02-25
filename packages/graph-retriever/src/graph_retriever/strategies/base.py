"""Define the base traversal strategy."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Iterable
from typing import Any

from graph_retriever.types import Node

class NodeTracker():
    visited: dict[str, Node] = {}
    to_traverse: dict[str, Node] = {}
    selected: dict[str, Node] = {}

    def select(self, nodes: dict[str, Node]) -> None:
        ...

    def traverse(self, nodes: dict[str, Node]) -> None:
        ...

    def select_and_traverse(self, nodes: dict[str, Node]) -> None:
        ...

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
    traverse_k: int = 4 # max_traverse?
    max_depth: int | None = None

    _query_embedding: list[float] = dataclasses.field(default_factory=list)

    @abc.abstractmethod
    def iteration(self, *, discovered: dict[str, Node],
                  tracker: NodeTracker) -> None:
        """
        Called on each iteration with the newly discovered nodes.

        This method should call `traverse` and/or `select` as appropriate
        to update the nodes that need to be traversed in this iteration or
        selected at the end of the retrieval, respectively.

        Parameters
        ----------
        nodes :
            Discovered nodes keyed by their IDs.
        """
        ...

    # def traverse(self, nodes: dict[str, Node]) -> None:
    #     """
    #     Called to queue nodes for traversal.
    #     """
    #     for node in nodes.values():
    #         # TODO: Have the strategy track the visited nodes?
    #         self._to_traverse.setdefault(node.id, node)

    # def select(self, nodes: Iterable[Node]):
    #     """
    #     Called by a strategy to indicate the given nodes are selected.

    #     Should be called as soon as nodes are definitely going to be selected.

    #     Parameters
    #     ----------
    #     nodes :
    #         The nodes to select.
    #     """
    #     for node in nodes:
    #         self.selected.setdefault(node.id, node)

    # def next_traversal(self, *, limit: int) -> Iterable[Node]:
    #     """
    #     Select discovered nodes to visit in the next iteration.

    #     This method determines which nodes will be traversed next. If it returns
    #     an empty list, traversal ends even if fewer than `k` nodes have been selected.

    #     Parameters
    #     ----------
    #     limit :
    #         Maximum number of nodes to select.

    #     Returns
    #     -------
    #     :
    #         Selected nodes for the next iteration. Traversal ends if this is empty.
    #     """
    #     ...

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
        return list(selected.values())[:self.k]

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
