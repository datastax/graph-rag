"""Utilities for handling and extracting edges in metadata."""

import warnings
from collections.abc import Iterable
from typing import Any

from langchain_graph_retriever.types import Edge, Edges, MetadataEdge, Node

BASIC_TYPES = (str, bool, int, float, complex, bytes)

# Sentinel object used with `dict.get(..., SENTINEL)` calls to distinguish
# between present but `None` (returns `None`) and absent (returns `SENTINEL`)
# elements.
SENTINEL = object()


class MetadataEdgeFunction:
    """
    Helper for extracting and encoding edges in metadata.

    This class provides tools to extract incoming and outgoing edges from document
    metadata and normalize metadata where needed. Both incoming and outgoing edges
    use the same target name, enabling equality matching for keys.

    Parameters
    ----------
        edges : list[tuple[str, str]]
            Definitions of edges for traversal, represented
            as pairs of incoming and outgoing keys.
            - If a string, the same key is used for both incoming and outgoing.
            - If a tuple, the first element is the outgoing key, and the second is
                the incoming key.

    Attributes
    ----------
        edges : list[tuple[str, str]]
            Definitions of edges for traversal, represented
            as pairs of incoming and outgoing keys.
            - If a string, the same key is used for both incoming and outgoing.
            - If a tuple, the first element is the outgoing key, and the second is
                the incoming key.

    Raises
    ------
        ValueError: If an invalid edge definition is provided.
    """

    def __init__(
        self,
        edges: list[str | tuple[str, str]],
    ) -> None:
        self.edges = []
        for edge in edges:
            if isinstance(edge, str):
                self.edges.append((edge, edge))
            elif (
                isinstance(edge, tuple)
                and len(edge) == 2
                and all(isinstance(item, str) for item in edge)
            ):
                self.edges.append((edge[0], edge[1]))
            else:
                raise ValueError(
                    "Invalid type for edge. must be 'str' or 'tuple[str,str]'"
                )

    def _edges_from_dict(
        self,
        metadata: dict[str, Any],
        *,
        incoming: bool = False,
    ) -> set[Edge]:
        """
        Extract edges from the metadata based on declared edge definitions.

        Parameters
        ----------
        metadata :dict[str, Any]
            The metadata dictionary to process.
        incoming : bool, default False
            If True, extracts edges for incoming relationships.

        Returns
        -------
        set[Edge]
            A set of edges extracted from the metadata.

        Notes
        -----
        - Handles both simple (key-value) and iterable metadata fields.
        - Issues warnings for unsupported or unexpected values.
        """
        edges: set[Edge] = set()
        for source_key, target_key in self.edges:
            if incoming:
                source_key = target_key

            value = metadata.get(source_key, SENTINEL)
            if isinstance(value, BASIC_TYPES):
                edges.add(MetadataEdge(incoming_field=target_key, value=value))
            elif isinstance(value, Iterable):
                for item in value:
                    if isinstance(item, BASIC_TYPES):
                        edges.add(MetadataEdge(incoming_field=target_key, value=item))
                    else:
                        warnings.warn(
                            f"Unsupported item value {item} in '{source_key}'"
                        )
            elif value is not SENTINEL:
                warnings.warn(f"Unsupported value {value} in '{source_key}'")
        return edges

    def __call__(self, node: Node) -> Edges:
        """
        Extract incoming and outgoing edges from metadata.

        This method retrieves edges based on the declared edge definitions, taking
        into account whether nested metadata is used.

        Parameters
        ----------
        metadata : dict[str, Any]
            The metadata dictionary to extract edges from.

        Returns
        -------
        Edges
            specyfing the incoming and outgoing edges of the node
        """
        outgoing_edges = self._edges_from_dict(node.metadata)
        incoming_edges = self._edges_from_dict(node.metadata, incoming=True)

        return Edges(incoming=incoming_edges, outgoing=outgoing_edges)
