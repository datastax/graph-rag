"""Define the base traversal strategy."""

import abc
import warnings
from typing import Any, Iterable, Optional

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

    @staticmethod
    def build(
        base_strategy: Optional["Strategy"] = None,
        base_k: int | None = None,
        **kwargs: Any,
    ) -> "Strategy":
        """Build a strategy for an retrieval.

        Build a strategy for an retrieval from the base strategy, any strategy passed in
        the invocation, and any related key word arguments.
        """
        # Check if there is a new strategy to use. Otherwise, use the base.
        strategy: Strategy | None = None
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
            if base_k:
                strategy = strategy.model_copy(update={"k": base_k})
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
    invalid_keys = set(dict.keys()) - set(model.model_fields.keys())
    if invalid_keys:
        return ", ".join([f"'{k}'" for k in invalid_keys])
    else:
        return None
