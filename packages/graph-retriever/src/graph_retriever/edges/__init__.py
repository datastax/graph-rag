"""
Specification and implementation of edges functions.

These are responsible for extracting edges from nodes and expressing them in way
that the adapters can implement.
"""

from ._base import Edge, EdgeFunction, Edges, IdEdge, MetadataEdge
from .metadata import EdgeSpec, Id, MetadataEdgeFunction

__all__ = [
    "Edge",
    "MetadataEdge",
    "IdEdge",
    "Edges",
    "EdgeFunction",
    "EdgeSpec",
    "Id",
    "MetadataEdgeFunction",
]
