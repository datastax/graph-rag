"""Define the node class."""

from dataclasses import dataclass, field
from typing import Any

from .edge_helper import Edge


@dataclass
class Node:
    """Information about a node during the traversal."""

    id: str
    """The document ID of this node."""

    depth: int
    """The depth (number of edges) through which this node was discovered.
    This may be larger than the *actual* depth of the node in the complete
    graph. If only a subset of edges are retrieved (such as when using
    similarity search to select only the most relevant edges), then this will
    correspond to the depth in the retrieved set of edges, which may be more
    than the true depth, if edges necessary for the shorter path are not used.
    """

    embedding: list[float]

    incoming_edges: set[Edge]
    outgoing_edges: set[Edge]

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata from the original document.

    This is a reference to the original document metadata, and should not be modified.
    Instead, modify `extra_metadata` which will be used as overrides when the final
    document is produced.
    """

    extra_metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata to add to the original document for the results."""
