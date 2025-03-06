from typing import Any

import pytest
from graph_retriever import Node
from graph_retriever.edges import Edges, IdEdge, MetadataEdge, MetadataEdgeFunction


def mk_node(metadata: dict[str, Any]) -> Node:
    return Node(
        id="id",
        content="testing",
        metadata=metadata,
        depth=0,
        embedding=[],
        similarity_score=0.2,
    )


def test_initialization():
    edge_function = MetadataEdgeFunction([("a", "a"), ("b", "c"), ("b", "b")])
    assert edge_function.edges == [("a", "a"), ("b", "c"), ("b", "b")]


def test_edge_function():
    edge_function = MetadataEdgeFunction([("href", "url")])
    assert edge_function(mk_node({"href": "a", "url": "b"})) == Edges(
        {MetadataEdge("url", "b")},
        {MetadataEdge("url", "a")},
    )

    assert edge_function(mk_node({"href": ["a", "c"], "url": "b"})) == Edges(
        {MetadataEdge("url", "b")},
        {MetadataEdge("url", "a"), MetadataEdge("url", "c")},
    )

    assert edge_function(mk_node({"href": ["a", "c"], "url": ["b", "d"]})) == Edges(
        {MetadataEdge("url", "b"), MetadataEdge("url", "d")},
        {MetadataEdge("url", "a"), MetadataEdge("url", "c")},
    )


def test_nested_edge():
    edge_function = MetadataEdgeFunction([("a.b", "b.c")])
    assert edge_function(mk_node({"a": {"b": 5}, "b": {"c": 7}})) == Edges(
        {MetadataEdge("b.c", 7)},
        {MetadataEdge("b.c", 5)},
    )


def test_link_to_id():
    edge_function = MetadataEdgeFunction([("mentions", "$id")])
    result = edge_function(mk_node({"mentions": ["a", "c"]}))

    assert result.incoming == {IdEdge("id")}
    assert result.outgoing == {IdEdge("a"), IdEdge("c")}


def test_link_from_id():
    edge_function = MetadataEdgeFunction([("$id", "mentions")])
    result = edge_function(mk_node({"mentions": ["a", "c"]}))

    assert result.incoming == {
        MetadataEdge("mentions", "a"),
        MetadataEdge("mentions", "c"),
    }
    assert result.outgoing == {MetadataEdge("mentions", "id")}


def test_unsupported_values():
    edge_function = MetadataEdgeFunction([("href", "url")])

    # Unsupported value
    with pytest.warns(UserWarning, match=r"Unsupported value .* in 'href'"):
        edge_function(mk_node({"href": None}))

    # Unsupported item value
    with pytest.warns(UserWarning, match=r"Unsupported item value .* in 'href'"):
        edge_function(mk_node({"href": [None]}))
