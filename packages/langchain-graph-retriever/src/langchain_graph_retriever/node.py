"""Defines the `Node` class used during graph traversal."""

from dataclasses import dataclass, field
from typing import Any

from .edge_helper import Edge


@dataclass
class Node:
    """Represents a node in the traversal graph.

    The `Node` class contains information about a document during graph traversal,
    including its depth, embedding, edges, and metadata.

    Attributes
    ----------
        id (str): The unique identifier of the document represented by this node.
        depth (int): The depth (number of edges) through which this node was discovered.
            This depth may not reflect the true depth in the full graph if only a subset
            of edges is retrieved.
        embedding (list[float]): The embedding vector of the document, used for
            similarity calculations.
        incoming_edges (set[Edge]): The set of edges pointing to this node.
        outgoing_edges (set[Edge]): The set of edges originating from this node.
        metadata (dict[str, Any]): Metadata from the original document. This is a
            reference to the original document metadata and should not be modified
            directly.
        extra_metadata (dict[str, Any]): Additional metadata to override or augment
            the original document metadata during traversal.
    """

    id: str
    """The unique identifier of this node, corresponding to the document ID."""

    depth: int
    """The depth (number of edges) through which this node was discovered.

    This may be larger than the true depth in the complete graph if only a subset
    of edges is retrieved (e.g., through similarity search). The depth corresponds
    to the retrieved set of edges, which may exclude edges necessary for a shorter
    path.
    """

    embedding: list[float]
    """The embedding vector of the document, used for similarity-based traversal."""

    incoming_edges: set[Edge]
    """A set of edges pointing to this node from other nodes."""

    outgoing_edges: set[Edge]
    """A set of edges originating from this node to other nodes."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata from the original document.

    This is a reference to the original document metadata, and should not be modified.
    Instead, modify `extra_metadata` which will be used as overrides when the final
    document is produced.
    """

    extra_metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata to augment or override the original document's metadata
    when producing final traversal results.
    """
