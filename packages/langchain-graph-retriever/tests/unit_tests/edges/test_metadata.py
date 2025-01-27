from typing import Any

import pytest
from langchain_graph_retriever.edges.metadata import MetadataEdgeFunction
from langchain_graph_retriever.types import Edges, IdEdge, MetadataEdge, Node


def mk_node(metadata: dict[str, Any]) -> Node:
    return Node(
        id="id",
        metadata=metadata,
        depth=0,
        embedding=[],
    )


def test_initialization():
    edge_function = MetadataEdgeFunction(["a", ("b", "c"), "b"])
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

def test_link_to_id():
    edge_function = MetadataEdgeFunction([("mentions", "$id")])
    assert edge_function(mk_node({"mentions": ["a", "c"]})) == Edges(
        {IdEdge("id")},
        {IdEdge("a"), IdEdge("c")},
    )


def test_unsupported_values():
    edge_function = MetadataEdgeFunction([("href", "url")])

    # Unsupported value
    with pytest.warns(UserWarning, match=r"Unsupported value .* in 'href'"):
        edge_function(mk_node({"href": None}))

    # Unsupported item value
    with pytest.warns(UserWarning, match=r"Unsupported item value .* in 'href'"):
        edge_function(mk_node({"href": [None]}))
