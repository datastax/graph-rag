import abc
from typing import Iterable, ParamSpec, Protocol, TypeVar, runtime_checkable

from ..node import Node


P = ParamSpec("P")
T = TypeVar("T", bound="NodeSelector", covariant=True)


@runtime_checkable
class NodeSelectorProtocol(Protocol[P, T]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


NodeSelectorFactory = NodeSelectorProtocol[P, T]


class NodeSelector(abc.ABC):
    """Interface for configuring node selection during the traversal."""

    @abc.abstractmethod
    def add_nodes(self, nodes: dict[str, Node]) -> None:
        """Add nodes to the set of available nodes."""
        ...

    @abc.abstractmethod
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        """Return the nodes to select at the next iteration."""
        ...

    def finalize_nodes(self, nodes: Iterable[Node]) -> Iterable[Node]:
        """Finalize the selected nodes."""
        return nodes
