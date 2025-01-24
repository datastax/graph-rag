"""Define the base traversal strategy."""

from __future__ import annotations

import abc
import warnings
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from ..node import Node


class Strategy(BaseModel, abc.ABC):
    """
    Interface for configuring node selection and traversal strategies.

    This base class defines how nodes are selected, traversed, and finalized during
    a graph traversal. Implementations can customize behaviors like limiting the depth
    of traversal, scoring nodes, or selecting the next set of nodes for exploration.

    Attributes
    ----------
        k (int): Number of nodes to retrieve during traversal. Default is 5.
        start_k (int): Number of initial documents to fetch via similarity, added
            to any specified starting nodes. Default is 4.
        adjacent_k (int): Number of adjacent documents to fetch for each outgoing edge.
            Default is 10.
        max_depth (int | None): Maximum traversal depth. If None, there is no limit.
        query_embedding (list[float]): Embedding vector for the query.
    """

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

        This method updates the strategy's state with nodes discovered during
        the traversal process.

        Args:
            nodes (dict[str, Node]): Discovered nodes keyed by their IDs.
        """
        ...

    @abc.abstractmethod
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        """Select discovered nodes to visit in the next iteration.

        This method determines which nodes will be traversed next. If it returns
        an empty list, traversal ends even if fewer than `k` nodes have been selected.

        Args:
            limit (int): Maximum number of nodes to select.

        Returns
        -------
            Iterable[Node]: Selected nodes for the next iteration. Traversal
            ends if this is empty.
        """
        ...

    def finalize_nodes(self, nodes: Iterable[Node]) -> Iterable[Node]:
        """Finalize the selected nodes.

        This method is called before returning the final set of nodes.

        Args:
            nodes (Iterable[Node]): Nodes selected for finalization.

        Returns
        -------
            Iterable[Node]: Finalized nodes.
        """
        return nodes

    @staticmethod
    def build(
        base_strategy: Strategy,
        **kwargs: Any,
    ) -> Strategy:
        """Build a strategy for a retrieval operation.

        Combines a base strategy with any provided keyword arguments to
        create a customized traversal strategy.

        Args:
            base_strategy (Strategy): The base strategy to start with.
            **kwargs: Additional configuration options for the strategy.

        Returns
        -------
            Strategy: A configured strategy instance.

        Raises
        ------
            ValueError: If 'strategy' is set incorrectly or extra arguments are invalid.
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

        # Warn if any of the kwargs don't exist in the strategy.
        # Note: We could rely on Pydantic with forbidden extra arguments to
        # handle this, however the experience isn't as nice (Validation error
        # rather than warning, no indication of which field, etc.).
        assert strategy is not None
        invalid_keys = _invalid_keys(strategy, kwargs)
        if invalid_keys is not None:
            warnings.warn(f"Unsupported key(s) {invalid_keys} set.")

        # Apply the kwargs to update the strategy.
        # This uses `model_validate` rather than `model_copy`` to re-apply validation.
        strategy = strategy.model_validate(
            {**strategy.model_dump(), **kwargs},
        )

        return strategy


def _invalid_keys(model: BaseModel, dict: dict[str, Any]) -> str | None:
    """Identify invalid keys in the given dictionary for a Pydantic model.

    Args:
        model (BaseModel): The Pydantic model to validate against.
        dict (dict[str, Any]): The dictionary to check.

    Returns
    -------
        str | None: A comma-separated string of invalid keys, or None if all keys
        are valid.
    """
    invalid_keys = set(dict.keys()) - set(model.model_fields.keys())
    if invalid_keys:
        return ", ".join([f"'{k}'" for k in invalid_keys])
    else:
        return None
