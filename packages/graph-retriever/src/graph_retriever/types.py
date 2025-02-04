"""Defines the `Node` class used during graph traversal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graph_retriever.edges import Edge


@dataclass
class Node:
    """
    Represents a node in the traversal graph.

    The [Node][graph_retriever.Node] class contains information about a document
    during graph traversal, including its depth, embedding, edges, and metadata.

    Parameters
    ----------
    id : str
        The unique identifier of the document represented by this node.
    content : str
        The content.
    depth : int
        The depth (number of edges) through which this node was discovered. This
        depth may not reflect the true depth in the full graph if only a subset
        of edges is retrieved.
    embedding : list[float]
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
