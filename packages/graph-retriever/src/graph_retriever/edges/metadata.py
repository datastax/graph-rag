"""Utilities for handling and extracting edges in metadata."""

import warnings
from collections.abc import Iterable
from typing import Any, TypeAlias

from graph_retriever.content import Content

from ._base import Edge, Edges, IdEdge, MetadataEdge

BASIC_TYPES = (str, bool, int, float, complex, bytes)

# Sentinel object used with `dict.get(..., SENTINEL)` calls to distinguish
# between present but `None` (returns `None`) and absent (returns `SENTINEL`)
# elements.
SENTINEL = object()

ID_MAGIC_STRING = "$id"


class Id:
    """
    Place-holder type indicating that the ID should be used.

    Deprecated: Use "$id" instead.
    """

    pass


EdgeSpec: TypeAlias = tuple[str | Id, str | Id]
"""
The definition of an edge for traversal, represented as a pair of fields
representing the source and target of the edge. Each may be:

- A string, `key`, indicating `doc.metadata[key]` as the value.
- The magic string `"$id"`, indicating `doc.id` as the value.

Examples
--------
```
url_to_href_edge          = ("url", "href")
keywords_to_keywords_edge = ("keywords", "keywords")
mentions_to_id_edge       = ("mentions", "$id")
id_to_mentions_edge       = ("$id", "mentions)
```
"""


def _nested_get(metadata: dict[str, Any], key: str) -> Any:
    value = metadata
    for key_part in key.split("."):
        value = value.get(key_part, SENTINEL)
        if value is SENTINEL:
            break
    return value


class MetadataEdgeFunction:
    """
    Helper for extracting and encoding edges in metadata.

    This class provides tools to extract incoming and outgoing edges from
    document metadata. Both incoming and outgoing edges use the same target
    name, enabling equality matching for keys.

    Parameters
    ----------
    edges :
        Definitions of edges for traversal, represented as a pair of fields
        representing the source and target of the edges.

    Raises
    ------
    ValueError
        If an invalid edge definition is provided.
    """

    def __init__(
        self,
        edges: list[EdgeSpec],
    ) -> None:
        self.edges = edges
        for source, target in edges:
            if not isinstance(source, str | Id):
                raise ValueError(f"Expected 'str | Id' but got: {source}")
            if not isinstance(target, str | Id):
                raise ValueError(f"Expected 'str | Id' but got: {target}")

    def _edges_from_dict(
        self,
        id: str,
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
        incoming :
            If True, extracts edges for incoming relationships.

        Returns
        -------
        :
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

            if target_key == ID_MAGIC_STRING or isinstance(target_key, Id):

                def mk_edge(v) -> Edge:
                    return IdEdge(id=str(v))
            else:

                def mk_edge(v) -> Edge:
                    return MetadataEdge(incoming_field=target_key, value=v)

            if source_key == ID_MAGIC_STRING or isinstance(source_key, Id):
                edges.add(mk_edge(id))
            else:
                value = _nested_get(metadata, source_key)
                if isinstance(value, BASIC_TYPES):
                    edges.add(mk_edge(value))
                elif isinstance(value, Iterable):
                    for item in value:
                        if isinstance(item, BASIC_TYPES):
                            edges.add(mk_edge(item))
                        else:
                            warnings.warn(
                                f"Unsupported item value {item} in '{source_key}'"
                            )
                elif value is not SENTINEL:
                    warnings.warn(f"Unsupported value {value} in '{source_key}'")
        return edges

    def __call__(self, content: Content) -> Edges:
        """
        Extract incoming and outgoing edges for a piece of content.

        This method retrieves edges based on the declared edge definitions, taking
        into account whether nested metadata is used.

        Parameters
        ----------
        content :
            The content to extract edges from.

        Returns
        -------
        :
            the incoming and outgoing edges of the node
        """
        outgoing_edges = self._edges_from_dict(content.id, content.metadata)
        incoming_edges = self._edges_from_dict(
            content.id, content.metadata, incoming=True
        )

        return Edges(incoming=incoming_edges, outgoing=outgoing_edges)
