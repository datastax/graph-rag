"""
Specification and implementation of edges functions.

These are responsible for extracting edges from nodes and expressing them in way
that the adapters can implement.
"""

from ._base import Edge, EdgeFunction, Edges, IdEdge, MetadataEdge
from .metadata import MetadataEdgeFunction

__all__ = [
    "Edge",
    "MetadataEdge",
    "IdEdge",
    "Edges",
    "EdgeFunction",
    "MetadataEdgeFunction",
]
