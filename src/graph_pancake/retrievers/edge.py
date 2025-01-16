from typing import Any


class Edge:
    """Represents an edge to all nodes with the given key/value incoming."""

    key: str
    value: Any

    def __init__(self, key: str, value: Any) -> None:
        self.key = key
        self.value = value

    def __str__(self) -> str:
        return f"Edge({self.key}->{self.value}"

    def __repr__(self) -> str:
        return f"Edge(key={self.key}, value={self.value}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.key == other.key and self.value == other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return (self.__str__()) < (other.__str__())

    def __hash__(self) -> int:
        return hash((self.key, self.value))
