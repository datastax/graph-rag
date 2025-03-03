import dataclasses

from graph_retriever.types import Node


def test_node_compare():
    """Only the node ID should be hashed/compared."""
    for field in dataclasses.fields(Node):
        assert field.hash is None, f"{field.name} should have default hash"
        if field.name == "id":
            assert field.compare, "id should be part of comparisons"
        else:
            assert not field.compare, f"{field.name} should not be part of comparisons"


def test_node_hash():
    node5a = Node(id=5, content="a", depth=0, embedding=[])
    node5b = Node(id=5, content="b", depth=0, embedding=[])
    node6 = Node(id=6, content="a", depth=0, embedding=[])

    assert hash(node5a) == hash(node5b)
    assert hash(node5a) != hash(node6)
