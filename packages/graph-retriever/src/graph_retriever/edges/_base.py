import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from immutabledict import immutabledict

from graph_retriever import Content


class Edge(abc.ABC):
    """
    An edge identifies properties necessary for finding matching nodes.

    Sub-classes should be hashable.
    """

    pass


@dataclass(frozen=True)
class MetadataEdge(Edge):
    """
    Link to nodes with specific metadata.

    A `MetadataEdge` connects to nodes with either:

    - `node.metadata[field] == value`
    - `node.metadata[field] CONTAINS value` (if the metadata is a collection).

    Parameters
    ----------
    incoming_field :
        The name of the metadata field storing incoming edges.
    value :
        The value associated with the key for this edge
    """

    def __init__(self, incoming_field: str, value: Any) -> None:
        # `self.field = value` and `setattr(self, "field", value)` -- don't work
        # because of frozen. we need to call `__setattr__` directly (as the
        # default `__init__` would do) to initialize the fields of the frozen
        # dataclass.
        object.__setattr__(self, "incoming_field", incoming_field)

        if isinstance(value, dict):
            value = immutabledict(value)
        object.__setattr__(self, "value", value)

    incoming_field: str
    value: Any


@dataclass(frozen=True)
class IdEdge(Edge):
    """
    An `IdEdge` connects to nodes with `node.id == id`.

    Parameters
    ----------
    id :
        The ID of the node to link to.
    """

    id: str


@dataclass
class Edges:
    """
    Information about the incoming and outgoing edges.

    Parameters
    ----------
    incoming :
        Incoming edges that link to this node.
    outgoing :
        Edges that this node link to. These edges should be defined in terms of
        the *incoming* `Edge` they match. For instance, a link from "mentions"
        to "id" would link to `IdEdge(...)`.
    """

    incoming: set[Edge]
    outgoing: set[Edge]


EdgeFunction: TypeAlias = Callable[[Content], Edges]
"""A function for extracting edges from nodes.

Implementations should be deterministic.
"""
