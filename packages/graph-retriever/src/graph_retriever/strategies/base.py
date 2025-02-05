"""Define the base traversal strategy."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Iterable
from typing import Any

from graph_retriever.types import Node


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
        Maximum number of nodes to retrieve during traversal.
    start_k :
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k :
        Number of documents to fetch for each outgoing edge.
    max_depth :
        Maximum traversal depth. If `None`, there is no limit.
    """

    k: int = 5
    start_k: int = 4
    adjacent_k: int = 10
    max_depth: int | None = None

    _query_embedding: list[float] = dataclasses.field(default_factory=list)

    @abc.abstractmethod
    def discover_nodes(self, nodes: dict[str, Node]) -> None:
        """
        Add discovered nodes to the strategy.

        This method updates the strategy's state with nodes discovered during
        the traversal process.

        Parameters
        ----------
        nodes :
            Discovered nodes keyed by their IDs.
        """
        ...

    @abc.abstractmethod
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        """
        Select discovered nodes to visit in the next iteration.

        This method determines which nodes will be traversed next. If it returns
        an empty list, traversal ends even if fewer than `k` nodes have been selected.

        Parameters
        ----------
        limit :
            Maximum number of nodes to select.

        Returns
        -------
        :
            Selected nodes for the next iteration. Traversal ends if this is empty.
        """
        ...

    def finalize_nodes(self, nodes: Iterable[Node]) -> Iterable[Node]:
        """
        Finalize the selected nodes.

        This method is called before returning the final set of nodes.

        Parameters
        ----------
        nodes :
            Nodes selected for finalization.

        Returns
        -------
        :
            Finalized nodes.
        """
        return nodes

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
