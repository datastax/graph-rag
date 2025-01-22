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
        strategy: Strategy | None = None
        if base_strategy is not None:
            # Deep copy in case the strategy has mutable state
            strategy = base_strategy.model_copy(deep=True)
            if base_k is not None:
                strategy.k = base_k
            for key, value in kwargs.items():
                if key == "strategy":
                    if isinstance(value, Strategy):
                        strategy = value.model_copy(deep=True)
                    elif isinstance(value, dict):
                        for k in value.keys():
                            if k not in strategy.model_fields.keys():
                                warnings.warn(
                                    f"Unsupported key '{k}' in 'strategy' set, ignored."
                                )
                        strategy = strategy.model_copy(update=value)
                    else:
                        raise ValueError(f"Unsupported strategy {value}")
                elif key in strategy.model_fields.keys():
                    strategy = strategy.model_copy(update={key: value})
                else:
                    warnings.warn(f"Unsupported key '{key}' set, ignored.")

        else:  # no base strategy
            for key, value in kwargs.items():
                if key == "strategy":
                    if isinstance(value, Strategy):
                        strategy = value.model_copy(deep=True)
                    else:
                        raise ValueError(f"Unsupported 'strategy': {value}")
                else:
                    if strategy is None:
                        raise ValueError(
                            "Error: 'strategy' must be set before other args."
                        )
                    elif key in strategy.model_fields.keys():
                        strategy = strategy.model_copy(update={key: value})
                    else:
                        warnings.warn(f"Unsupported key '{key}' set, ignored.")
        if strategy is None:
            raise ValueError("'strategy' must be set.")
        return strategy
