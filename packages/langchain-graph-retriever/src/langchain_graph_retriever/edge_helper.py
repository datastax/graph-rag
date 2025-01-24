"""Utilities for handling and extracting edges in metadata."""

import warnings
from collections.abc import Iterable
from typing import Any, NamedTuple

BASIC_TYPES = (str, bool, int, float, complex, bytes)


# Sentinel object used with `dict.get(..., SENTINEL)` calls to distinguish
# between present but `None` (returns `None`) and absent (returns `SENTINEL`)
# elements.
SENTINEL = object()


class Edge(NamedTuple):
    """Represents an edge to all nodes with the given key/value pair.

    Attributes
    ----------
        key : str
            The metadata key associated with the edge.
        value : Any
            The value associated with the key for this edge.
    """

    key: str
    value: Any


class EdgeHelper:
    """Helper for extracting and encoding edges in metadata.

    This class provides tools to extract incoming and outgoing edges from document
    metadata and normalize metadata where needed. Both incoming and outgoing edges
    use the same target name, enabling equality matching for keys.

    Parameters
    ----------
        use_normalized_metadata : bool
            Indicates whether normalized metadata is used.
        denormalized_path_delimiter : str
            Delimiter for splitting keys in denormalized metadata.
        denormalized_static_value : Any
            Value used to mark static entries in denormalized metadata.
        edges : list[tuple[str, str]]
            Definitions of edges for traversal, represented
            as pairs of incoming and outgoing keys.
            - If a string, the same key is used for both incoming and outgoing.
            - If a tuple, the first element is the outgoing key, and the second is
                the incoming key.

    Attributes
    ----------
        use_normalized_metadata : bool
            Indicates whether normalized metadata is used.
        denormalized_path_delimiter : str
            Delimiter for splitting keys in denormalized metadata.
        denormalized_static_value : Any
            Value used to mark static entries in denormalized metadata.
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
        *,
        use_normalized_metadata: bool = False,
        denormalized_path_delimiter: str = ".",
        denormalized_static_value: Any = "$",
    ) -> None:
        self.use_normalized_metadata = use_normalized_metadata
        self.denormalized_path_delimiter = denormalized_path_delimiter
        self.denormalized_static_value = denormalized_static_value

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
        warn_normalized: bool = False,
        incoming: bool = False,
    ) -> set[Edge]:
        """Extract edges from the metadata based on declared edge definitions.

        Args:
            metadata (dict[str, Any]): The metadata dictionary to process.
            warn_normalized (bool, optional): If True, warnings are issued for
                normalized metadata. Defaults to False.
            incoming (bool, optional): If True, extracts edges for incoming
                relationships. Defaults to False.

        Returns
        -------
            set[Edge]: A set of edges extracted from the metadata.

        Notes
        -----
            - Handles both simple (key-value) and iterable metadata fields.
            - Issues warnings for unsupported or unexpected values.
        """
        edges = set()
        for source_key, target_key in self.edges:
            if incoming:
                source_key = target_key

            value = metadata.get(source_key, SENTINEL)
            if isinstance(value, BASIC_TYPES):
                edges.add(Edge(target_key, value))
            elif isinstance(value, Iterable):
                # Note: `str` and `bytes` are in `BASIC_TYPES` so no need to
                # guard against.
                if warn_normalized:
                    warnings.warn(f"Normalized value {value} in '{source_key}'")
                else:
                    for item in value:
                        if isinstance(item, BASIC_TYPES):
                            edges.add(Edge(target_key, item))
                        else:
                            warnings.warn(
                                f"Unsupported item value {item} in '{source_key}'"
                            )
            elif value is not SENTINEL:
                warnings.warn(f"Unsupported value {value} in '{source_key}'")
        return edges

    def _normalize_metadata(
        self, denormalized_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize metadata by extracting key-value pairs based on the delimiter.

        Args:
            denormalized_metadata (dict[str, Any]): The denormalized metadata
                dictionary.

        Returns
        -------
            dict[str, Any]: A dictionary containing normalized key-value pairs.

        Notes
        -----
            - Only processes keys with the `denormalized_static_value`.
            - Skips items that cannot be compared or are invalid.
        """
        normalized: dict[str, Any] = {}
        for key, value in denormalized_metadata.items():
            try:
                if value != self.denormalized_static_value:
                    continue
            except (TypeError, ValueError):
                # Skip items that can't be compared
                continue

            split = key.split(self.denormalized_path_delimiter, 2)
            if len(split) == 2 and len(split[1]) > 0:
                normalized.setdefault(split[0], set()).add(split[1])
        return normalized

    def get_incoming_outgoing(
        self, metadata: dict[str, Any]
    ) -> tuple[set[Edge], set[Edge]]:
        """Extract incoming and outgoing edges from metadata.

        This method retrieves edges based on the declared edge definitions, taking
        into account whether normalized metadata is used. It combines normalized
        and denormalized edge data as needed.

        Args:
            metadata (dict[str, Any]): The metadata dictionary to extract edges from.

        Returns
        -------
            tuple[set[Edge], set[Edge]]: A tuple containing:
                - Incoming edges as a set of `Edge` objects.
                - Outgoing edges as a set of `Edge` objects.
        """
        warn_normalized = not self.use_normalized_metadata
        outgoing_edges = self._edges_from_dict(
            metadata, warn_normalized=warn_normalized
        )
        incoming_edges = self._edges_from_dict(
            metadata, incoming=True, warn_normalized=warn_normalized
        )

        if not self.use_normalized_metadata:
            normalized = self._normalize_metadata(metadata)

            outgoing_edges.update(self._edges_from_dict(normalized))
            incoming_edges.update(self._edges_from_dict(normalized, incoming=True))

        return (incoming_edges, outgoing_edges)
