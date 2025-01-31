"""Defines the `Node` class used during graph traversal."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .content import Content


@dataclass
class Node:
    """
    Represents a node in the traversal graph.

    The `Node` class contains information about a document during graph traversal,
    including its depth, embedding, edges, and metadata.

    Attributes
    ----------
    id : str
        The unique identifier of the document represented by this node.
    content : str
        The content.
    depth : int
        The depth (number of edges) through which this node was discovered. This
        depth may not reflect the true depth in the full graph if only a subset
        of edges is retrieved.
    embedding : list[float])
        The embedding vector of the document, used for similarity calculations.
    metadata : dict[str, Any]
        Metadata from the original document. This is a reference to the original
        document metadata and should not be modified directly. Any updates to
        metadata should be made to `extra_metadata`.
    extra_metadata : dict[str, Any]
        Additional metadata to override or augment the original document
        metadata during traversal.
    """

    id: str
    content: str
    depth: int
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    incoming_edges: set[Edge] = field(default_factory=set)
    outgoing_edges: set[Edge] = field(default_factory=set)

    extra_metadata: dict[str, Any] = field(default_factory=dict)


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

    A `MetadataEdge` defines nodes with `node.metadata[field] == value` or
    `node.metadata[field] CONTAINS value` (if the metadata is a collection).

    Attributes
    ----------
    incoming_field : str
        The name of the metadata field storing incoming edges.
    value : Any
        The value associated with the key for this edge
    """

    incoming_field: str
    value: Any


@dataclass(frozen=True)
class IdEdge(Edge):
    """
    Nodes with `node.id == id`.

    Attributes
    ----------
    id : str
        The ID of the node to link to.
    """

    id: str


@dataclass
class Edges:
    """
    Information about the incoming and outgoing edges.

    Attributes
    ----------
    incoming : set[Edge]
        Incoming edges that link to this node.
    outgoing : set[Edge]
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
