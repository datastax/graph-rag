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
    id :
        The unique identifier of the document represented by this node.
    content :
        The content.
    depth :
        The depth (number of edges) through which this node was discovered. This
        depth may not reflect the true depth in the full graph if only a subset
        of edges is retrieved.
    embedding :
        The embedding vector of the document, used for similarity calculations.
    metadata :
        Metadata from the original document. This is a reference to the original
        document metadata and should not be modified directly. Any updates to
        metadata should be made to `extra_metadata`.
    extra_metadata :
        Additional metadata to override or augment the original document
        metadata during traversal.
    """

    id: str
    content: str
    depth: int
    similarity_score: float
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    incoming_edges: set[Edge] = field(default_factory=set)
    outgoing_edges: set[Edge] = field(default_factory=set)

    extra_metadata: dict[str, Any] = field(default_factory=dict)
