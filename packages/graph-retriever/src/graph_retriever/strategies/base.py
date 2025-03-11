"""Define the base traversal strategy."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Iterable
from typing import Any

from graph_retriever.content import Content
from graph_retriever.types import Node

DEFAULT_SELECT_K = 5


class NodeTracker:
    """
    Helper class initiating node selection and traversal.

    Call .select(nodes) to add nodes to the result set.
    Call .traverse(nodes) to add nodes to the next traversal.
    Call .select_and_traverse(nodes) to add nodes to the result set and the next
        traversal.
    """

    def __init__(self, select_k: int, max_depth: int | None) -> None:
        self._select_k: int = select_k
        self._max_depth: int | None = max_depth
        self._visited_node_ids: set[str] = set()
        # use a dict to preserve order
        self.to_traverse: dict[str, Node] = dict()
        self.selected: list[Node] = []

    @property
    def num_remaining(self):
        """The remaining number of nodes to be selected."""
        return max(self._select_k - len(self.selected), 0)

    def select(self, nodes: Iterable[Node]) -> None:
        """Select nodes to be included in the result set."""
        for node in nodes:
            node.extra_metadata["_depth"] = node.depth
            node.extra_metadata["_similarity_score"] = node.similarity_score
        self.selected.extend(nodes)

    def traverse(self, nodes: Iterable[Node]) -> int:
        """
        Select nodes to be included in the next traversal.

        Returns
        -------
        Number of nodes added for traversal.

        Notes
        -----
        - Nodes are only added if they have not been visited before.
        - Nodes are only added if they do not exceed the maximum depth.
        - If no new nodes are chosen for traversal, or selected for output, then
            the traversal will stop.
        - Traversal will also stop if the number of selected nodes reaches the select_k
            limit.
        """
        new_nodes = {
            n.id: n
            for n in nodes
            if self._not_visited(n)
            if self._max_depth is None or n.depth < self._max_depth
        }
        self.to_traverse.update(new_nodes)
        self._visited_node_ids.update(new_nodes.keys())
        return len(new_nodes)

    def select_and_traverse(self, nodes: Iterable[Node]) -> int:
        """
        Select nodes to be included in the result set and the next traversal.

        Returns
        -------
        Number of nodes added for traversal.

        Notes
        -----
        - Nodes are only added for traversal if they have not been visited before.
        - Nodes are only added for traversal if they do not exceed the maximum depth.
        - If no new nodes are chosen for traversal, or selected for output, then
            the traversal will stop.
        - Traversal will also stop if the number of selected nodes reaches the select_k
            limit.
        """
        self.select(nodes)
        return self.traverse(nodes)

    def _not_visited(self, item: Content | Node):
        """Return true if the content or node has not been visited."""
        return item.id not in self._visited_node_ids

    def _should_stop_traversal(self):
        """Return true if traversal should be stopped."""
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
    k:
        Deprecated: Use `select_k` instead.
        Maximum number of nodes to select and return during traversal.
    """

    select_k: int = dataclasses.field(default=DEFAULT_SELECT_K)
    start_k: int = 4
    adjacent_k: int = 10
    max_traverse: int | None = None
    max_depth: int | None = None
    k: int = dataclasses.field(default=DEFAULT_SELECT_K, repr=False)

    _query_embedding: list[float] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """Allow passing the deprecated 'k' value instead of 'select_k'."""
        if self.select_k == DEFAULT_SELECT_K and self.k != DEFAULT_SELECT_K:
            self.select_k = self.k
        else:
            self.k = self.select_k

    @abc.abstractmethod
    def iteration(self, *, nodes: Iterable[Node], tracker: NodeTracker) -> None:
        """
        Process the newly discovered nodes on each iteration.

        This method should call `tracker.traverse()` and/or `tracker.select()`
        as appropriate to update the nodes that need to be traversed in this iteration
        or selected at the end of the retrieval, respectively.

        Parameters
        ----------
        nodes :
            The newly discovered nodes. These are nodes which have not been
            visited before which have an incoming edge which has not been
            visited before from a node which is newly traversed in the previous
            iteration.
        tracker :
            The tracker object to manage the traversal and selection of nodes.

        Notes
        -----
        - This method is called once for each iteration of the traversal.
        - In order to stop iterating either choose to not traverse any additional nodes
        or don't select any additional nodes for output.
        """
        ...

    def finalize_nodes(self, selected: Iterable[Node]) -> Iterable[Node]:
        """
        Finalize the selected nodes.

        This method is called before returning the final set of nodes. It allows
        the strategy to perform any final processing or re-ranking of the selected
        nodes.

        Parameters
        ----------
        selected :
            The selected nodes to be finalized

        Returns
        -------
        :
            Finalized nodes.

        Notes
        -----
        - The default implementation returns the first `self.select_k` selected nodes
        without any additional processing.
        """
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
        if "k" in kwargs:
            kwargs["select_k"] = kwargs.pop("k")
        strategy = dataclasses.replace(strategy, **kwargs)

        return strategy
