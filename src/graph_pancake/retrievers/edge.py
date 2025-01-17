from typing import NamedTuple, Any

class Edge(NamedTuple):
    """Represents an edge to all nodes with the given key/value incoming."""
    key: str
    value: Any
