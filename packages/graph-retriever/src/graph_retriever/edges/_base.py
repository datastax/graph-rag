import abc
from collections.abc import Callable
from dataclasses import dataclass
import dataclasses
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
    fields :
        The fields constrained by this edge.
    """

    fields: immutabledict[str, Any]

    def __init__(self, fields: dict[str, Any] | immutabledict[str, Any]) -> None:
        # self.fields and setattr(self, ...) -- don't work because of frozen.
        # we need to call `__setattr__` directly (as the default `__init__` would do)
        # to initialize the fields of the frozen dataclass.
        object.__setattr__(self, "fields", immutabledict(fields))

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
