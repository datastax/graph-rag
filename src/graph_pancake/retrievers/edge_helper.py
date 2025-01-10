import warnings
from typing import Any, Iterable

from .edge import Edge

BASIC_TYPES = (str, bool, int, float, complex, bytes)


# Sentinel object used with `dict.get(..., SENTINEL)` calls to distinguish
# between present but `None` (returns `None`) and absent (returns `SENTINEL`)
# elements.
SENTINEL = object()


class EdgeHelper:
    """Helper for extracting and encoding edges in metadata.

    Both incoming and outgoing edges are reported using the target name.
    This ensures that using them as keys allows equality matching.
    """

    def __init__(
        self,
        edges: list[str | tuple[str, str]],
        *,
        use_denormalized_metadata: bool = False,
        denormalized_path_delimiter: str = ".",
        denormalized_static_value: Any = True,
    ) -> None:
        self.use_denormalized_metadata = use_denormalized_metadata
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
        warn_non_denormalized: bool = False,
        incoming: bool = False,
        is_denormalized: bool = False,
    ) -> set[Edge]:
        """Extract edges from the metadata based on declared edges."""

        edges = set()
        for source_key, target_key in self.edges:
            if incoming:
                source_key = target_key

            value = metadata.get(source_key, SENTINEL)
            if isinstance(value, BASIC_TYPES):
                edges.add(Edge(target_key, value, is_denormalized=is_denormalized))
            elif isinstance(value, Iterable):
                # Note: `str` and `bytes` are in `BASIC_TYPES` so no need to
                # guard against.
                if warn_non_denormalized:
                    warnings.warn(f"Non-denormalized value {value} in '{source_key}'")
                else:
                    for item in value:
                        if isinstance(item, BASIC_TYPES):
                            edges.add(
                                Edge(target_key, item, is_denormalized=is_denormalized)
                            )
                        else:
                            raise ValueError(
                                f"Unsupported item value {item} in '{source_key}'"
                            )
            elif value is not SENTINEL:
                raise ValueError(f"Unsupported value {value} in '{source_key}'")
        return edges

    def _normalize_metadata(
        self, denormalized_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in denormalized_metadata.items():
            if value != self.denormalized_static_value:
                continue

            split = key.split(self.denormalized_path_delimiter, 2)
            if len(split) == 2 and len(split[1]) > 0:
                normalized.setdefault(split[0], set()).add(split[1])
        return normalized

    def get_incoming_outgoing(
        self, metadata: dict[str, Any]
    ) -> tuple[set[Edge], set[Edge]]:
        warn_non_denormalized = self.use_denormalized_metadata
        outgoing_edges = self._edges_from_dict(
            metadata, warn_non_denormalized=warn_non_denormalized
        )
        incoming_edges = self._edges_from_dict(
            metadata, incoming=True, warn_non_denormalized=warn_non_denormalized
        )

        if self.use_denormalized_metadata:
            normalized = self._normalize_metadata(metadata)

            outgoing_edges.update(self._edges_from_dict(normalized))
            incoming_edges.update(self._edges_from_dict(normalized, incoming=True))

        return (incoming_edges, outgoing_edges)

    def get_metadata_filter(
        self,
        base_filter: dict[str, Any] | None = None,
        edge: Edge | None = None,
    ) -> dict[str, Any]:
        """Builds a metadata filter to search for documents

        Args:
            base_filter: Any metadata that should be used for hybrid search
            edge: An optional outgoing edge to add to the search
        """
        metadata_filter = {**(base_filter or {})}
        if edge is None:
            metadata_filter
        elif edge.is_denormalized:
            metadata_filter[
                f"{edge.key}{self.denormalized_path_delimiter}{edge.value}"
            ] = self.denormalized_static_value
        else:
            metadata_filter[edge.key] = edge.value

        return metadata_filter
